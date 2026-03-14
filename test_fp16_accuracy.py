"""Compare RTMPose fp16 vs fp32 CoreML on real images.

Focus on final keypoint positions (post-argmax), not raw logits.
Report error as % of crop dimensions and absolute pixels.
Only compare high-confidence keypoints.
"""

import os
import numpy as np
import cv2
import coremltools as ct
import onnxruntime as ort
from ultralytics import YOLO

# --- Convert fp16 model if needed ---
FP16_PATH = "rtmpose_fp16.mlpackage"
if not os.path.exists(FP16_PATH):
    print("Converting RTMPose to fp16...")
    import torch
    from onnx2torch import convert as onnx2torch_convert

    onnx_path = "rtmpose_fixed.onnx"
    torch_model = onnx2torch_convert(onnx_path)
    torch_model.eval()
    dummy = torch.randn(1, 3, 384, 288)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, dummy)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=(1, 3, 384, 288))],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.save(FP16_PATH)
    print(f"Saved {FP16_PATH}")

# --- Load models ---
print("Loading models...")
fp32_model = ct.models.MLModel("rtmpose.mlpackage", compute_units=ct.ComputeUnit.ALL)
fp16_model = ct.models.MLModel(FP16_PATH, compute_units=ct.ComputeUnit.ALL)
yolo = YOLO("yolo11s-seg.pt")
onnx_sess = ort.InferenceSession("rtmpose_fixed.onnx", providers=["CPUExecutionProvider"])

fp32_spec = fp32_model.get_spec()
fp32_in = fp32_spec.description.input[0].name
fp32_outs = sorted([o.name for o in fp32_spec.description.output])

fp16_spec = fp16_model.get_spec()
fp16_in = fp16_spec.description.input[0].name
fp16_outs = sorted([o.name for o in fp16_spec.description.output])

SIMCC_SPLIT = 2.0

def preprocess(img, bbox):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1
    s = max(bw/288, bh/384) * 1.25
    x1c = max(0, int(cx - 288*s/2))
    y1c = max(0, int(cy - 384*s/2))
    x2c = min(w, int(cx + 288*s/2))
    y2c = min(h, int(cy + 384*s/2))
    crop = img[y1c:y2c, x1c:x2c]
    resized = cv2.resize(crop, (288, 384))
    blob = resized.astype(np.float32)
    blob[:,:,0] = (blob[:,:,0] - 123.675) / 58.395
    blob[:,:,1] = (blob[:,:,1] - 116.28) / 57.12
    blob[:,:,2] = (blob[:,:,2] - 103.53) / 57.375
    blob = np.transpose(blob, (2,0,1))[np.newaxis].astype(np.float32)
    return blob, s, x1c, y1c

def decode(simcc_x, simcc_y, scale, x1c, y1c):
    kx = np.argmax(simcc_x[0], axis=-1) / SIMCC_SPLIT
    ky = np.argmax(simcc_y[0], axis=-1) / SIMCC_SPLIT
    conf_x = np.max(simcc_x[0], axis=-1)
    conf_y = np.max(simcc_y[0], axis=-1)
    conf = np.minimum(conf_x, conf_y)
    kx_img = kx * scale + x1c
    ky_img = ky * scale + y1c
    return np.stack([kx_img, ky_img], axis=-1), conf

# --- Run comparison ---
from pathlib import Path
image_paths = sorted(Path("test_images").glob("*.jpg"))
print(f"\nComparing fp16 vs fp32 on {len(image_paths)} images")
print(f"(Only high-confidence keypoints, conf > 0.3)\n")

all_errors_px = []
all_errors_pct = []
all_kpt_match = []

for img_path in image_paths:
    stem = img_path.stem
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    r = yolo.predict(img, classes=[0], verbose=False)[0]
    if len(r.boxes) == 0:
        continue
    idx = r.boxes.conf.argmax().item()
    bbox = r.boxes.xyxy[idx].cpu().numpy()

    blob, scale, x1c, y1c = preprocess(img, bbox)

    # ONNX reference
    onnx_outs = onnx_sess.run(None, {"input": blob})
    kpts_onnx, conf_onnx = decode(onnx_outs[0], onnx_outs[1], scale, x1c, y1c)

    # fp32 CoreML
    fp32_out = fp32_model.predict({fp32_in: blob})
    kpts_fp32, conf_fp32 = decode(fp32_out[fp32_outs[0]], fp32_out[fp32_outs[1]], scale, x1c, y1c)

    # fp16 CoreML
    fp16_out = fp16_model.predict({fp16_in: blob})
    kpts_fp16, conf_fp16 = decode(fp16_out[fp16_outs[0]], fp16_out[fp16_outs[1]], scale, x1c, y1c)

    # Compare: fp16 vs ONNX (ground truth), only high-conf keypoints
    valid = conf_onnx > 0.3
    n_valid = valid.sum()

    if n_valid == 0:
        continue

    # Pixel error
    diff = kpts_fp16[valid] - kpts_onnx[valid]
    px_err = np.sqrt((diff**2).sum(axis=-1))

    # As % of image diagonal
    diag = np.sqrt(w**2 + h**2)
    pct_err = px_err / diag * 100

    # How many keypoints have identical argmax (0px error)
    exact_match = (px_err < 0.01).sum()

    # Also check fp32 vs ONNX for reference
    diff32 = kpts_fp32[valid] - kpts_onnx[valid]
    px_err32 = np.sqrt((diff32**2).sum(axis=-1))

    all_errors_px.extend(px_err.tolist())
    all_errors_pct.extend(pct_err.tolist())
    all_kpt_match.append((exact_match, n_valid))

    print(f"  {stem}: {n_valid} valid kpts")
    print(f"    fp16 vs ONNX:  mean={px_err.mean():.1f}px  max={px_err.max():.1f}px  "
          f"median={np.median(px_err):.1f}px  exact_match={exact_match}/{n_valid}")
    print(f"    fp32 vs ONNX:  mean={px_err32.mean():.1f}px  max={px_err32.max():.1f}px  "
          f"(reference)")
    print(f"    fp16 as %diag: mean={pct_err.mean():.3f}%  max={pct_err.max():.3f}%")

# --- Summary ---
all_px = np.array(all_errors_px)
all_pct = np.array(all_errors_pct)
total_exact = sum(m for m, _ in all_kpt_match)
total_valid = sum(n for _, n in all_kpt_match)

print(f"\n{'='*60}")
print(f"Summary: fp16 vs ONNX (high-confidence keypoints only)")
print(f"{'='*60}")
print(f"  Total keypoints compared: {len(all_px)}")
print(f"  Exact match (0px error):  {total_exact}/{total_valid} ({100*total_exact/total_valid:.1f}%)")
print(f"  Pixel error:  mean={all_px.mean():.1f}px  median={np.median(all_px):.1f}px  "
      f"max={all_px.max():.1f}px  p95={np.percentile(all_px, 95):.1f}px")
print(f"  As % of diag: mean={all_pct.mean():.3f}%  median={np.median(all_pct):.3f}%  "
      f"max={all_pct.max():.3f}%  p95={np.percentile(all_pct, 95):.3f}%")
print(f"\n  Within 1%:  {(all_pct < 1.0).sum()}/{len(all_pct)} ({100*(all_pct < 1.0).mean():.1f}%)")
print(f"  Within 2%:  {(all_pct < 2.0).sum()}/{len(all_pct)} ({100*(all_pct < 2.0).mean():.1f}%)")

# Distribution
print(f"\n  Error distribution (pixels):")
for thresh in [0, 1, 2, 5, 10, 20, 50, 100]:
    count = (all_px <= thresh).sum()
    print(f"    ≤{thresh:3d}px: {count:5d}/{len(all_px)} ({100*count/len(all_px):5.1f}%)")
