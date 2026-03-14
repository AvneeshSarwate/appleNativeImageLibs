"""Detailed per-stage profiling of the CoreML pipeline.

Measures every step independently to identify bottlenecks:
- Frame decode
- YOLO preprocessing (resize, letterbox, normalize) — both ultralytics wrapper and raw
- YOLO CoreML inference (raw, no wrapper)
- YOLO postprocessing (parse detections, mask decode)
- Person crop + RTMPose preprocess
- RTMPose CoreML inference (raw)
- RTMPose postprocess (SimCC decode)
- Mask operations (unpad, resize, contour)
- Data format conversions (numpy ↔ PIL, tensor copies)
"""

import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import coremltools as ct
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VIDEO = "foot_dance.mp4"
N_FRAMES = 50
SIMCC_SPLIT_RATIO = 2.0

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("Loading models...")

# Raw CoreML models
yolo_cml_raw = ct.models.MLModel("yolo11s-seg.mlpackage", compute_units=ct.ComputeUnit.ALL)
yolo_spec = yolo_cml_raw.get_spec()
yolo_input_name = yolo_spec.description.input[0].name
yolo_out_names = [o.name for o in yolo_spec.description.output]
print(f"  YOLO input: {yolo_input_name} (expects PIL Image)")
print(f"  YOLO outputs: {yolo_out_names}")

rtmpose_cml = ct.models.MLModel("rtmpose.mlpackage", compute_units=ct.ComputeUnit.ALL)
rtmpose_spec = rtmpose_cml.get_spec()
rtmpose_input = rtmpose_spec.description.input[0].name
rtmpose_out_names = sorted([o.name for o in rtmpose_spec.description.output])
print(f"  RTMPose input: {rtmpose_input}")
print(f"  RTMPose outputs: {rtmpose_out_names}")

# Ultralytics wrapper for comparison
yolo_wrapper = YOLO("yolo11s-seg.mlpackage")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def letterbox_pil(img_bgr, size=640):
    """BGR numpy → letterboxed PIL image for YOLO CoreML."""
    h, w = img_bgr.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    return pil_img, scale, pad_left, pad_top


def letterbox_numpy(img_bgr, size=640):
    """BGR numpy → letterboxed float32 NCHW tensor for YOLO."""
    h, w = img_bgr.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    tensor = canvas.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]
    return tensor, scale, pad_left, pad_top


def parse_yolo_raw(outputs, conf_thresh=0.25, person_class=0):
    """Parse raw YOLO-seg outputs: find top person detection.

    Raw output shapes (from ultralytics CoreML export):
    - detection tensor: (1, 116, 8400) — 4 box + 80 class + 32 mask coeffs per anchor
    - proto tensor: (1, 32, 160, 160) — mask prototypes
    """
    # Identify which output is which by shape
    det_tensor = proto_tensor = None
    for name, val in outputs.items():
        if len(val.shape) == 3:  # (1, 116, 8400)
            det_tensor = val
        elif len(val.shape) == 4:  # (1, 32, 160, 160)
            proto_tensor = val

    if det_tensor is None:
        return None, None, None, None

    # det_tensor: (1, 116, 8400) → transpose to (8400, 116)
    det = det_tensor[0].T  # (8400, 116)

    # Split: box (4), classes (80), mask_coeffs (32)
    boxes_xywh = det[:, :4]
    class_scores = det[:, 4:84]
    mask_coeffs = det[:, 84:]

    # Person class scores
    person_scores = class_scores[:, person_class]
    mask = person_scores > conf_thresh
    if not mask.any():
        return None, None, None, None

    # Top detection
    idx = person_scores[mask].argmax()
    valid_indices = np.where(mask)[0]
    best_idx = valid_indices[idx]

    # Convert xywh → xyxy
    cx, cy, w, h = boxes_xywh[best_idx]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    bbox = np.array([x1, y1, x2, y2])
    conf = person_scores[best_idx]
    coeffs = mask_coeffs[best_idx]

    return bbox, conf, coeffs, proto_tensor


def decode_mask(coeffs, proto, bbox_640, img_h, img_w, scale, pad_left, pad_top):
    """Decode mask from coefficients and prototypes."""
    # coeffs: (32,), proto: (1, 32, 160, 160)
    proto_2d = proto[0]  # (32, 160, 160)
    raw_mask = np.einsum("c,chw->hw", coeffs, proto_2d)  # (160, 160)
    raw_mask = 1 / (1 + np.exp(-raw_mask))  # sigmoid

    # Upsample to 640x640
    mask_640 = cv2.resize(raw_mask, (640, 640))

    # Remove letterbox padding
    new_h = int(img_h * scale)
    new_w = int(img_w * scale)
    mask_content = mask_640[pad_top:pad_top + new_h, pad_left:pad_left + new_w]

    # Resize to original image size
    mask_full = cv2.resize(mask_content, (img_w, img_h))
    return (mask_full > 0.5).astype(np.uint8)


def preprocess_rtmpose(img_bgr, bbox, target_h=384, target_w=288):
    """Crop person, resize, normalize for RTMPose."""
    x1, y1, x2, y2 = bbox
    h, w = img_bgr.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    crop_scale = max(bw / target_w, bh / target_h) * 1.25
    nw, nh = target_w * crop_scale, target_h * crop_scale
    x1c = max(0, int(cx - nw / 2))
    y1c = max(0, int(cy - nh / 2))
    x2c = min(w, int(cx + nw / 2))
    y2c = min(h, int(cy + nh / 2))
    crop = img_bgr[y1c:y2c, x1c:x2c]
    resized = cv2.resize(crop, (target_w, target_h))
    blob = resized.astype(np.float32)
    blob[:, :, 0] = (blob[:, :, 0] - 123.675) / 58.395
    blob[:, :, 1] = (blob[:, :, 1] - 116.28) / 57.12
    blob[:, :, 2] = (blob[:, :, 2] - 103.53) / 57.375
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis].astype(np.float32)
    return blob, {"x1c": x1c, "y1c": y1c, "scale": crop_scale}


def decode_simcc(simcc_x, simcc_y, transform):
    """Decode SimCC logits → image-space keypoints."""
    kpt_x = np.argmax(simcc_x[0], axis=-1) / SIMCC_SPLIT_RATIO
    kpt_y = np.argmax(simcc_y[0], axis=-1) / SIMCC_SPLIT_RATIO
    scale = transform["scale"]
    kpt_x_img = kpt_x * scale + transform["x1c"]
    kpt_y_img = kpt_y * scale + transform["y1c"]
    return np.stack([kpt_x_img, kpt_y_img], axis=-1)


def find_contours(mask_uint8):
    """Extract contour polygon from binary mask."""
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_indices = np.linspace(0, total_frames - 1, N_FRAMES, dtype=int)

timings = defaultdict(list)

print(f"\nProfiling {N_FRAMES} frames from {VIDEO}...\n")

# Warmup
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, warmup_frame = cap.read()
if ret:
    pil_img, s, pl, pt_ = letterbox_pil(warmup_frame)
    for _ in range(5):
        yolo_cml_raw.predict({yolo_input_name: pil_img})
    dummy_input = np.random.randn(1, 3, 384, 288).astype(np.float32)
    for _ in range(5):
        rtmpose_cml.predict({rtmpose_input: dummy_input})

for i, frame_idx in enumerate(sample_indices):
    # --- Frame decode ---
    t = time.perf_counter()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    timings["1_frame_decode"].append(time.perf_counter() - t)
    if not ret:
        continue
    h, w = frame.shape[:2]

    # --- YOLO preprocess (raw: letterbox + PIL) ---
    t = time.perf_counter()
    pil_img, scale, pad_left, pad_top = letterbox_pil(frame)
    timings["2_yolo_preprocess_raw"].append(time.perf_counter() - t)

    # --- YOLO inference (raw CoreML) ---
    t = time.perf_counter()
    yolo_raw_outs = yolo_cml_raw.predict({yolo_input_name: pil_img})
    timings["3_yolo_inference_raw"].append(time.perf_counter() - t)

    # --- YOLO postprocess (parse detections + mask decode) ---
    t = time.perf_counter()
    bbox_640, conf, coeffs, proto = parse_yolo_raw(yolo_raw_outs)
    timings["4_yolo_postprocess_parse"].append(time.perf_counter() - t)

    if bbox_640 is None:
        continue

    # Unscale bbox from 640 coords to original image coords
    bbox_img = bbox_640.copy()
    bbox_img[0] = (bbox_640[0] - pad_left) / scale
    bbox_img[1] = (bbox_640[1] - pad_top) / scale
    bbox_img[2] = (bbox_640[2] - pad_left) / scale
    bbox_img[3] = (bbox_640[3] - pad_top) / scale

    # --- Mask decode ---
    t = time.perf_counter()
    mask_full = decode_mask(coeffs, proto, bbox_640, h, w, scale, pad_left, pad_top)
    timings["5_mask_decode"].append(time.perf_counter() - t)

    # --- Contour extraction ---
    t = time.perf_counter()
    contour = find_contours(mask_full)
    timings["6_contour_extract"].append(time.perf_counter() - t)

    # --- RTMPose preprocess ---
    t = time.perf_counter()
    rtm_input, transform = preprocess_rtmpose(frame, bbox_img)
    timings["7_rtmpose_preprocess"].append(time.perf_counter() - t)

    # --- RTMPose inference ---
    t = time.perf_counter()
    rtm_outs = rtmpose_cml.predict({rtmpose_input: rtm_input})
    timings["8_rtmpose_inference"].append(time.perf_counter() - t)

    # --- RTMPose postprocess ---
    t = time.perf_counter()
    kpts = decode_simcc(rtm_outs[rtmpose_out_names[0]], rtm_outs[rtmpose_out_names[1]], transform)
    timings["9_rtmpose_postprocess"].append(time.perf_counter() - t)

    # --- For comparison: ultralytics wrapper overhead ---
    t = time.perf_counter()
    _ = yolo_wrapper.predict(frame, classes=[0], verbose=False)
    timings["X_ultralytics_wrapper_total"].append(time.perf_counter() - t)

    if (i + 1) % 25 == 0:
        print(f"  Processed {i+1}/{N_FRAMES}")

cap.release()

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Per-Stage Timing Breakdown (milliseconds)")
print("=" * 70)

total_pipeline = np.zeros(N_FRAMES)
stage_names = sorted([k for k in timings.keys() if not k.startswith("X_")])

for name in stage_names:
    arr = np.array(timings[name]) * 1000
    total_pipeline[:len(arr)] += arr
    pct = np.median(arr) / 1  # will compute pct after
    print(f"  {name:30s}  median {np.median(arr):7.2f}ms  "
          f"mean {np.mean(arr):7.2f}ms  p95 {np.percentile(arr, 95):7.2f}ms")

pipeline_median = np.median(total_pipeline)
print(f"\n  {'TOTAL_PIPELINE':30s}  median {pipeline_median:7.2f}ms  "
      f"→ {1000/pipeline_median:.1f} FPS")

# Percentage breakdown
print("\n" + "-" * 70)
print("Percentage Breakdown (by median)")
print("-" * 70)
for name in stage_names:
    arr = np.array(timings[name]) * 1000
    pct = np.median(arr) / pipeline_median * 100
    bar = "#" * int(pct / 2)
    print(f"  {name:30s}  {pct:5.1f}%  {bar}")

# Ultralytics wrapper comparison
if "X_ultralytics_wrapper_total" in timings:
    wrapper_ms = np.array(timings["X_ultralytics_wrapper_total"]) * 1000
    raw_yolo_ms = (np.array(timings["2_yolo_preprocess_raw"]) +
                   np.array(timings["3_yolo_inference_raw"]) +
                   np.array(timings["4_yolo_postprocess_parse"])) * 1000
    print(f"\n{'='*70}")
    print("Ultralytics Wrapper Overhead")
    print(f"  Raw YOLO (preprocess+infer+parse):   median {np.median(raw_yolo_ms):6.2f}ms")
    print(f"  Ultralytics wrapper (predict):       median {np.median(wrapper_ms):6.2f}ms")
    print(f"  Wrapper overhead:                    {np.median(wrapper_ms) - np.median(raw_yolo_ms):6.2f}ms")

# Summary of optimization opportunities
print(f"\n{'='*70}")
print("Optimization Opportunities")
print("=" * 70)
for name in stage_names:
    arr = np.array(timings[name]) * 1000
    med = np.median(arr)
    if med > 1.0:
        print(f"  {name}: {med:.1f}ms — worth optimizing")
    else:
        print(f"  {name}: {med:.2f}ms — negligible")
