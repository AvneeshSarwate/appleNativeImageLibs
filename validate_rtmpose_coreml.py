"""Phase 2.2 – Validate CoreML RTMPose outputs against ONNX reference."""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import coremltools as ct

BASELINE_DIR = Path("baselines")
TEST_DIR = Path("test_images")
RTMPOSE_ONNX = "/Users/avneeshsarwate/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"

image_paths = sorted(TEST_DIR.glob("*.jpg"))
print(f"Validating RTMPose CoreML on {len(image_paths)} images\n")

# Load ONNX model for reference
onnx_sess = ort.InferenceSession(RTMPOSE_ONNX, providers=["CPUExecutionProvider"])
onnx_input_name = onnx_sess.get_inputs()[0].name
onnx_input_shape = onnx_sess.get_inputs()[0].shape  # (1, 3, 288, 384)
print(f"ONNX input: {onnx_input_name}, shape: {onnx_input_shape}")
print(f"ONNX outputs: {[o.name for o in onnx_sess.get_outputs()]}")

# Load CoreML model
cml_model = ct.models.MLModel("rtmpose.mlpackage", compute_units=ct.ComputeUnit.ALL)
spec = cml_model.get_spec()
cml_input_name = spec.description.input[0].name
print(f"CoreML input: {cml_input_name}")
print(f"CoreML outputs: {[o.name for o in spec.description.output]}\n")


def preprocess_person_crop(img, bbox, target_h=384, target_w=288):
    """Crop and preprocess a person region for RTMPose.

    Mimics rtmlib preprocessing: crop, resize, normalize.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Add padding (10%)
    pad_w = int((x2 - x1) * 0.1)
    pad_h = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img.shape[1], x2 + pad_w)
    y2 = min(img.shape[0], y2 + pad_h)

    crop = img[y1:y2, x1:x2]
    resized = cv2.resize(crop, (target_w, target_h))

    # Normalize: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    blob = resized.astype(np.float32)
    blob[:, :, 0] = (blob[:, :, 0] - 123.675) / 58.395
    blob[:, :, 1] = (blob[:, :, 1] - 116.28) / 57.12
    blob[:, :, 2] = (blob[:, :, 2] - 103.53) / 57.375

    # HWC → NCHW
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis].astype(np.float32)
    return blob


# For each test image, get the YOLO bbox, crop, run both models, compare
from ultralytics import YOLO
yolo = YOLO("yolo11s-seg.pt")

results = []

for img_path in image_paths:
    stem = img_path.stem
    img = cv2.imread(str(img_path))

    # Get person bbox from YOLO
    yolo_r = yolo.predict(img, classes=[0], verbose=False)[0]
    if len(yolo_r.boxes) == 0:
        print(f"  {stem}: no person detected, skipping")
        continue

    # Use top-confidence person
    idx = yolo_r.boxes.conf.argmax().item()
    bbox = yolo_r.boxes.xyxy[idx].cpu().numpy()

    # Preprocess crop
    input_tensor = preprocess_person_crop(img, bbox)

    # Run ONNX
    onnx_outs = onnx_sess.run(None, {onnx_input_name: input_tensor})

    # Run CoreML
    cml_outs = cml_model.predict({cml_input_name: input_tensor})

    # Compare outputs
    onnx_names = [o.name for o in onnx_sess.get_outputs()]
    cml_names = sorted(cml_outs.keys())

    max_abs_errors = []
    for i, oname in enumerate(onnx_names):
        onnx_val = onnx_outs[i]
        # Find matching CoreML output
        cml_val = cml_outs[cml_names[i]] if i < len(cml_names) else None
        if cml_val is None:
            continue
        err = np.abs(onnx_val - cml_val).max()
        max_abs_errors.append((oname, err, onnx_val.shape))

    # SimCC decode: argmax along last dim for x and y logits
    # Typically outputs are (1, K, Wx) and (1, K, Wy)
    if len(onnx_outs) >= 2:
        onnx_x = np.argmax(onnx_outs[0][0], axis=-1)  # (K,)
        onnx_y = np.argmax(onnx_outs[1][0], axis=-1)  # (K,)

        cml_out0 = cml_outs[cml_names[0]]
        cml_out1 = cml_outs[cml_names[1]]
        cml_x = np.argmax(cml_out0[0], axis=-1)
        cml_y = np.argmax(cml_out1[0], axis=-1)

        kpt_x_err = np.abs(onnx_x.astype(float) - cml_x.astype(float)).max()
        kpt_y_err = np.abs(onnx_y.astype(float) - cml_y.astype(float)).max()
    else:
        kpt_x_err = kpt_y_err = -1

    errors_str = "  ".join(f"{n}: {e:.5f}" for n, e, _ in max_abs_errors)
    print(f"  {stem}: {errors_str}  kpt_err=({kpt_x_err:.0f}, {kpt_y_err:.0f})px")
    results.append((stem, max_abs_errors, kpt_x_err, kpt_y_err))

# Summary
print("\n" + "=" * 60)
print("RTMPose Validation Summary")
print("=" * 60)
all_pass = True
for stem, errs, kx, ky in results:
    raw_ok = all(e < 0.05 for _, e, _ in errs)
    kpt_ok = kx <= 2 and ky <= 2
    status = "PASS" if (raw_ok and kpt_ok) else ("WARN" if kpt_ok else "FAIL")
    if status != "PASS":
        all_pass = False
    max_err = max(e for _, e, _ in errs) if errs else 0
    print(f"  {stem}: {status}  max_raw_err={max_err:.5f}  kpt_err=({kx:.0f},{ky:.0f})")

print(f"\nOverall: {'ALL PASSED' if all_pass else 'CHECK ABOVE'}")
