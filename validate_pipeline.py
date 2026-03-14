"""Phase 3.3 – Validate end-to-end CoreML pipeline.

Compares CoreML RTMPose vs ONNX RTMPose using the SAME preprocessing (same
YOLO-seg bbox, same crop). This isolates model accuracy from preprocessing
differences vs rtmlib.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import coremltools as ct
from ultralytics import YOLO

TEST_DIR = Path("test_images")
SIMCC_SPLIT_RATIO = 2.0
RTMPOSE_ONNX = "rtmpose_fixed.onnx"

image_paths = sorted(TEST_DIR.glob("*.jpg"))
print(f"Validating end-to-end pipeline on {len(image_paths)} images\n")

# Load models
yolo_cml = YOLO("yolo11s-seg.mlpackage")
rtmpose_cml = ct.models.MLModel("rtmpose.mlpackage", compute_units=ct.ComputeUnit.ALL)
rtmpose_spec = rtmpose_cml.get_spec()
rtmpose_input = rtmpose_spec.description.input[0].name
rtmpose_out_names = sorted([o.name for o in rtmpose_spec.description.output])

rtmpose_onnx = ort.InferenceSession(RTMPOSE_ONNX, providers=["CPUExecutionProvider"])


def preprocess_crop(img, bbox, target_h=384, target_w=288):
    """Crop person, resize, normalize for RTMPose."""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    scale = max(bw / target_w, bh / target_h) * 1.25

    new_w = target_w * scale
    new_h = target_h * scale
    x1c = max(0, int(cx - new_w / 2))
    y1c = max(0, int(cy - new_h / 2))
    x2c = min(w, int(cx + new_w / 2))
    y2c = min(h, int(cy + new_h / 2))

    crop = img[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None, None

    resized = cv2.resize(crop, (target_w, target_h))
    blob = resized.astype(np.float32)
    blob[:, :, 0] = (blob[:, :, 0] - 123.675) / 58.395
    blob[:, :, 1] = (blob[:, :, 1] - 116.28) / 57.12
    blob[:, :, 2] = (blob[:, :, 2] - 103.53) / 57.375
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis].astype(np.float32)

    transform = {"x1c": x1c, "y1c": y1c, "scale": scale}
    return blob, transform


def decode_simcc(simcc_x, simcc_y, transform):
    """Decode SimCC logits to image-space keypoints."""
    kpt_x = np.argmax(simcc_x[0], axis=-1) / SIMCC_SPLIT_RATIO
    kpt_y = np.argmax(simcc_y[0], axis=-1) / SIMCC_SPLIT_RATIO
    conf_x = np.max(simcc_x[0], axis=-1)
    conf_y = np.max(simcc_y[0], axis=-1)
    confidence = np.minimum(conf_x, conf_y)

    scale = transform["scale"]
    kpt_x_img = kpt_x * scale + transform["x1c"]
    kpt_y_img = kpt_y * scale + transform["y1c"]

    return np.stack([kpt_x_img, kpt_y_img], axis=-1), confidence


def unpad_mask(mask, img_h, img_w, model_size=640):
    """Remove letterbox padding from YOLO CoreML mask."""
    scale = min(model_size / img_h, model_size / img_w)
    new_h, new_w = int(img_h * scale), int(img_w * scale)
    pad_top = (model_size - new_h) // 2
    pad_left = (model_size - new_w) // 2
    cropped = mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    return cv2.resize(cropped.astype(np.float32), (img_w, img_h)) > 0.5


results = []
pipeline_times = []

for img_path in image_paths:
    stem = img_path.stem
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    t0 = time.perf_counter()

    # Step 1: YOLO-seg CoreML
    yolo_r = yolo_cml.predict(img, classes=[0], verbose=False)[0]
    if len(yolo_r.boxes) == 0:
        print(f"  {stem}: no person detected")
        continue

    idx = yolo_r.boxes.conf.argmax().item()
    bbox = yolo_r.boxes.xyxy[idx].cpu().numpy()

    # Step 2: Crop + preprocess
    input_tensor, transform = preprocess_crop(img, bbox)
    if input_tensor is None:
        continue

    # Step 3: RTMPose CoreML
    cml_outs = rtmpose_cml.predict({rtmpose_input: input_tensor})
    simcc_x_cml = cml_outs[rtmpose_out_names[0]]
    simcc_y_cml = cml_outs[rtmpose_out_names[1]]

    # Step 4: Decode
    cml_kpts, cml_conf = decode_simcc(simcc_x_cml, simcc_y_cml, transform)

    t1 = time.perf_counter()
    pipeline_times.append(t1 - t0)

    # Step 5: Run ONNX RTMPose with same input for comparison
    onnx_outs = rtmpose_onnx.run(None, {"input": input_tensor})
    onnx_kpts, onnx_conf = decode_simcc(onnx_outs[0], onnx_outs[1], transform)

    # Compare CoreML vs ONNX (same preprocessing!)
    kpt_diff = np.abs(cml_kpts - onnx_kpts)
    max_px_err = kpt_diff.max()
    mean_px_err = kpt_diff.mean()

    # Mask info
    has_mask = yolo_r.masks is not None
    if has_mask:
        mask_raw = yolo_r.masks.data[idx].cpu().numpy()
        if mask_raw.shape[0] == mask_raw.shape[1]:
            mask_full = unpad_mask(mask_raw, h, w)
        else:
            mask_full = cv2.resize(mask_raw.astype(np.float32), (w, h)) > 0.5
        mask_pix = mask_full.sum()
    else:
        mask_pix = 0

    results.append((stem, max_px_err, mean_px_err, cml_conf.mean(), mask_pix > 0))
    print(f"  {stem}: max_err={max_px_err:.2f}px  mean_err={mean_px_err:.2f}px  "
          f"conf={cml_conf.mean():.3f}  mask={'yes' if mask_pix > 0 else 'no'}  "
          f"time={pipeline_times[-1]*1000:.0f}ms")

# Summary
print("\n" + "=" * 60)
print("End-to-End Pipeline Validation (CoreML vs ONNX, same preprocess)")
print("=" * 60)
all_pass = True
for stem, max_e, mean_e, conf, has_m in results:
    status = "PASS" if max_e < 2.0 else "WARN" if max_e < 5.0 else "FAIL"
    if status != "PASS":
        all_pass = False
    print(f"  {stem}: {status}  max={max_e:.2f}px  mean={mean_e:.2f}px  mask={has_m}")

max_errs = [r[1] for r in results]
print(f"\nMax pixel error across all images: {max(max_errs):.2f}px")
print(f"Pipeline latency (median): {np.median(pipeline_times)*1000:.0f}ms")
print(f"Overall: {'ALL PASSED' if all_pass else 'CHECK ABOVE'}")
