"""Phase 4 – Optimized CoreML pipeline. Single-frame synchronous processing.

Mask + keypoints are always from the same frame. No cross-frame pipelining.

Optimizations vs profiled baseline (124ms → target <30ms):
- Sequential frame reads (no seeking) → eliminates 63ms decode
- Raw CoreML inference (no ultralytics wrapper) → saves 10ms
- Contour at 160x160 proto resolution, scale points → saves 8ms+ (no resize)
- Minimal numpy postprocessing
"""

import time
from collections import defaultdict

import cv2
import numpy as np
import coremltools as ct
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VIDEO = "foot_dance.mp4"
N_FRAMES = 200
SIMCC_SPLIT_RATIO = 2.0
YOLO_SIZE = 640
YOLO_CONF_THRESH = 0.25
PERSON_CLASS = 0

# ---------------------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------------------
print("Loading models...")
yolo_model = ct.models.MLModel("yolo11s-seg.mlpackage", compute_units=ct.ComputeUnit.ALL)
yolo_spec = yolo_model.get_spec()
yolo_input_name = yolo_spec.description.input[0].name

rtmpose_model = ct.models.MLModel("rtmpose.mlpackage", compute_units=ct.ComputeUnit.ALL)
rtmpose_spec = rtmpose_model.get_spec()
rtmpose_input_name = rtmpose_spec.description.input[0].name
rtmpose_out_names = sorted([o.name for o in rtmpose_spec.description.output])


# ---------------------------------------------------------------------------
# Precompute letterbox params for fixed input resolution
# ---------------------------------------------------------------------------
class FrameProcessor:
    """Processes frames at a fixed input resolution. Precomputes letterbox params."""

    def __init__(self, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h

        # Letterbox params (constant for fixed resolution)
        self.scale = min(YOLO_SIZE / img_h, YOLO_SIZE / img_w)
        self.new_w = int(img_w * self.scale)
        self.new_h = int(img_h * self.scale)
        self.pad_left = (YOLO_SIZE - self.new_w) // 2
        self.pad_top = (YOLO_SIZE - self.new_h) // 2

        # Proto-space params (160 = YOLO_SIZE / 4)
        proto_scale = 160 / YOLO_SIZE
        self.proto_pad_left = int(self.pad_left * proto_scale)
        self.proto_pad_top = int(self.pad_top * proto_scale)
        self.proto_content_w = int(self.new_w * proto_scale)
        self.proto_content_h = int(self.new_h * proto_scale)

        # Scale from proto content space to image space
        self.proto_to_img_x = img_w / self.proto_content_w
        self.proto_to_img_y = img_h / self.proto_content_h

        # Preallocate letterbox canvas
        self._canvas = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, dtype=np.uint8)

    def preprocess_yolo(self, frame_bgr):
        """BGR frame → letterboxed PIL image for YOLO CoreML."""
        resized = cv2.resize(frame_bgr, (self.new_w, self.new_h))
        self._canvas[:] = 114
        self._canvas[self.pad_top:self.pad_top + self.new_h,
                     self.pad_left:self.pad_left + self.new_w] = resized
        return Image.fromarray(cv2.cvtColor(self._canvas, cv2.COLOR_BGR2RGB))

    def parse_yolo(self, outputs):
        """Parse raw YOLO outputs → top person bbox (image coords), mask coeffs, protos."""
        det_tensor = proto_tensor = None
        for val in outputs.values():
            if len(val.shape) == 3:
                det_tensor = val
            elif len(val.shape) == 4:
                proto_tensor = val

        if det_tensor is None:
            return None

        det = det_tensor[0].T  # (8400, 116)
        person_scores = det[:, 4 + PERSON_CLASS]
        mask = person_scores > YOLO_CONF_THRESH
        if not mask.any():
            return None

        best = np.where(mask)[0][person_scores[mask].argmax()]

        # xywh → xyxy in 640 space
        cx, cy, w, h = det[best, :4]
        bbox_640 = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

        # Convert to image coords
        bbox_img = np.array([
            (bbox_640[0] - self.pad_left) / self.scale,
            (bbox_640[1] - self.pad_top) / self.scale,
            (bbox_640[2] - self.pad_left) / self.scale,
            (bbox_640[3] - self.pad_top) / self.scale,
        ])
        bbox_img[0] = max(0, bbox_img[0])
        bbox_img[1] = max(0, bbox_img[1])
        bbox_img[2] = min(self.img_w, bbox_img[2])
        bbox_img[3] = min(self.img_h, bbox_img[3])

        return {
            "bbox": bbox_img,
            "conf": float(person_scores[best]),
            "coeffs": det[best, 84:],
            "proto": proto_tensor,
        }

    def decode_contour(self, coeffs, proto):
        """Decode mask at proto resolution (160x160), extract contour, scale to image coords."""
        # Matmul + sigmoid at 160x160
        raw_mask = np.einsum("c,chw->hw", coeffs, proto[0])
        raw_mask = 1.0 / (1.0 + np.exp(-raw_mask))

        # Crop to content region (remove letterbox padding)
        content = raw_mask[
            self.proto_pad_top:self.proto_pad_top + self.proto_content_h,
            self.proto_pad_left:self.proto_pad_left + self.proto_content_w,
        ]

        # Threshold and find contours at proto resolution
        binary = (content > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Take largest contour, scale points to image coords
        contour = max(contours, key=cv2.contourArea)
        scaled = contour.astype(np.float32).copy()
        scaled[:, :, 0] *= self.proto_to_img_x
        scaled[:, :, 1] *= self.proto_to_img_y
        return scaled

    def preprocess_rtmpose(self, frame_bgr, bbox):
        """Crop person from frame, resize+normalize for RTMPose."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        crop_scale = max(bw / 288, bh / 384) * 1.25
        nw, nh = 288 * crop_scale, 384 * crop_scale

        x1c = max(0, int(cx - nw / 2))
        y1c = max(0, int(cy - nh / 2))
        x2c = min(self.img_w, int(cx + nw / 2))
        y2c = min(self.img_h, int(cy + nh / 2))

        crop = frame_bgr[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return None, None

        resized = cv2.resize(crop, (288, 384))
        blob = resized.astype(np.float32)
        blob[:, :, 0] = (blob[:, :, 0] - 123.675) / 58.395
        blob[:, :, 1] = (blob[:, :, 1] - 116.28) / 57.12
        blob[:, :, 2] = (blob[:, :, 2] - 103.53) / 57.375
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis]

        return blob, {"x1c": x1c, "y1c": y1c, "scale": crop_scale}

    @staticmethod
    def decode_simcc(simcc_x, simcc_y, transform):
        """Decode SimCC logits → (133, 2) keypoints in image coords + confidence."""
        kpt_x = np.argmax(simcc_x[0], axis=-1) / SIMCC_SPLIT_RATIO
        kpt_y = np.argmax(simcc_y[0], axis=-1) / SIMCC_SPLIT_RATIO
        conf = np.minimum(np.max(simcc_x[0], axis=-1), np.max(simcc_y[0], axis=-1))

        s = transform["scale"]
        kpts = np.stack([
            kpt_x * s + transform["x1c"],
            kpt_y * s + transform["y1c"],
        ], axis=-1)
        return kpts, conf


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO)
assert cap.isOpened()

img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {img_w}x{img_h} @ {fps_video:.0f}fps")

proc = FrameProcessor(img_w, img_h)

# Warmup
ret, warmup = cap.read()
if ret:
    pil = proc.preprocess_yolo(warmup)
    for _ in range(5):
        yolo_model.predict({yolo_input_name: pil})
    dummy = np.random.randn(1, 3, 384, 288).astype(np.float32)
    for _ in range(5):
        rtmpose_model.predict({rtmpose_input_name: dummy})

# Reset to start, read sequentially
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

timings = defaultdict(list)
results_count = 0

print(f"\nProcessing {N_FRAMES} sequential frames...\n")

for i in range(N_FRAMES):
    # --- Read frame (sequential, no seeking) ---
    t0 = time.perf_counter()
    ret, frame = cap.read()
    t_read = time.perf_counter()
    if not ret:
        break

    # --- YOLO preprocess ---
    pil_img = proc.preprocess_yolo(frame)
    t_yolo_pre = time.perf_counter()

    # --- YOLO inference ---
    yolo_out = yolo_model.predict({yolo_input_name: pil_img})
    t_yolo_inf = time.perf_counter()

    # --- YOLO parse ---
    detection = proc.parse_yolo(yolo_out)
    t_yolo_post = time.perf_counter()

    if detection is None:
        timings["total"].append(time.perf_counter() - t0)
        continue

    # --- Contour at proto resolution ---
    contour = proc.decode_contour(detection["coeffs"], detection["proto"])
    t_contour = time.perf_counter()

    # --- RTMPose preprocess ---
    rtm_input, transform = proc.preprocess_rtmpose(frame, detection["bbox"])
    t_rtm_pre = time.perf_counter()

    if rtm_input is None:
        timings["total"].append(time.perf_counter() - t0)
        continue

    # --- RTMPose inference ---
    rtm_out = rtmpose_model.predict({rtmpose_input_name: rtm_input})
    t_rtm_inf = time.perf_counter()

    # --- RTMPose decode ---
    kpts, conf = proc.decode_simcc(
        rtm_out[rtmpose_out_names[0]], rtm_out[rtmpose_out_names[1]], transform
    )
    t_rtm_post = time.perf_counter()

    # Record timings
    timings["1_read"].append(t_read - t0)
    timings["2_yolo_pre"].append(t_yolo_pre - t_read)
    timings["3_yolo_inf"].append(t_yolo_inf - t_yolo_pre)
    timings["4_yolo_post"].append(t_yolo_post - t_yolo_inf)
    timings["5_contour"].append(t_contour - t_yolo_post)
    timings["6_rtm_pre"].append(t_rtm_pre - t_contour)
    timings["7_rtm_inf"].append(t_rtm_inf - t_rtm_pre)
    timings["8_rtm_post"].append(t_rtm_post - t_rtm_inf)
    timings["total"].append(t_rtm_post - t0)
    results_count += 1

    if (i + 1) % 50 == 0:
        med = np.median(timings["total"]) * 1000
        print(f"  Frame {i+1}/{N_FRAMES}  —  median {med:.1f}ms  ({1000/med:.1f} FPS)")

cap.release()

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"Optimized Pipeline — {results_count}/{N_FRAMES} frames with detections")
print(f"{'='*70}")

stages = sorted([k for k in timings.keys() if k != "total"])
total_med = np.median(timings["total"]) * 1000

for name in stages:
    arr = np.array(timings[name]) * 1000
    med = np.median(arr)
    pct = med / total_med * 100
    bar = "#" * int(pct / 2)
    print(f"  {name:20s}  {med:6.2f}ms  ({pct:4.1f}%)  {bar}")

print(f"\n  {'total':20s}  {total_med:6.2f}ms  → {1000/total_med:.1f} FPS")
print(f"  {'':20s}  mean {np.mean(timings['total'])*1000:6.2f}ms  "
      f"p95 {np.percentile(np.array(timings['total'])*1000, 95):6.2f}ms")

# Compare to baseline
print(f"\n{'='*70}")
print("vs CPU baseline: 503.6ms (2.0 FPS)")
print(f"vs profiled unoptimized: 124.3ms (8.0 FPS)")
print(f"optimized: {total_med:.1f}ms ({1000/total_med:.1f} FPS)")
print(f"speedup vs CPU: {503.6/total_med:.1f}x")
print(f"speedup vs unoptimized: {124.3/total_med:.1f}x")
