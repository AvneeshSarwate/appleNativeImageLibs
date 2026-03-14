"""Compare RTMPose fp16 vs fp32 CoreML on real images.

Tests both ALL and CPU+NE compute units for fp16 to see if
ANE execution produces different numerical behavior than GPU.
"""

import os
import numpy as np
import cv2
import coremltools as ct
import onnxruntime as ort
from ultralytics import YOLO
from pathlib import Path

SIMCC_SPLIT = 2.0

# Load models
print("Loading models...")
onnx_sess = ort.InferenceSession("rtmpose_fixed.onnx", providers=["CPUExecutionProvider"])
fp16_all = ct.models.MLModel("rtmpose_fp16.mlpackage", compute_units=ct.ComputeUnit.ALL)
fp16_cne = ct.models.MLModel("rtmpose_fp16.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
fp32_all = ct.models.MLModel("rtmpose.mlpackage", compute_units=ct.ComputeUnit.ALL)
yolo = YOLO("yolo11s-seg.pt")

fp16_spec = fp16_all.get_spec()
fp16_in = fp16_spec.description.input[0].name
fp16_outs = sorted([o.name for o in fp16_spec.description.output])

fp32_spec = fp32_all.get_spec()
fp32_in = fp32_spec.description.input[0].name
fp32_outs = sorted([o.name for o in fp32_spec.description.output])


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
    conf = np.minimum(np.max(simcc_x[0], axis=-1), np.max(simcc_y[0], axis=-1))
    kx_img = kx * scale + x1c
    ky_img = ky * scale + y1c
    return np.stack([kx_img, ky_img], axis=-1), conf


image_paths = sorted(Path("test_images").glob("*.jpg"))
print(f"Comparing on {len(image_paths)} images (high-confidence keypoints, conf > 0.3)\n")

# Collect per-variant errors
variants = {
    "fp16 (ALL/GPU)": (fp16_all, fp16_in, fp16_outs),
    "fp16 (CPU+NE)":  (fp16_cne, fp16_in, fp16_outs),
    "fp32 (ALL)":     (fp32_all, fp32_in, fp32_outs),
}

all_errors = {name: [] for name in variants}
all_exact = {name: [0, 0] for name in variants}  # [exact, total]

for img_path in image_paths:
    stem = img_path.stem
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    diag = np.sqrt(w**2 + h**2)

    r = yolo.predict(img, classes=[0], verbose=False)[0]
    if len(r.boxes) == 0:
        continue
    idx = r.boxes.conf.argmax().item()
    bbox = r.boxes.xyxy[idx].cpu().numpy()
    blob, scale, x1c, y1c = preprocess(img, bbox)

    # ONNX reference
    onnx_outs = onnx_sess.run(None, {"input": blob})
    kpts_ref, conf_ref = decode(onnx_outs[0], onnx_outs[1], scale, x1c, y1c)
    valid = conf_ref > 0.3
    n_valid = valid.sum()
    if n_valid == 0:
        continue

    print(f"  {stem} ({n_valid} valid kpts):")

    for vname, (model, in_name, out_names) in variants.items():
        out = model.predict({in_name: blob})
        kpts, conf = decode(out[out_names[0]], out[out_names[1]], scale, x1c, y1c)

        diff = kpts[valid] - kpts_ref[valid]
        px_err = np.sqrt((diff**2).sum(axis=-1))
        exact = (px_err < 0.01).sum()

        all_errors[vname].extend(px_err.tolist())
        all_exact[vname][0] += exact
        all_exact[vname][1] += n_valid

        print(f"    {vname:20s}  mean={px_err.mean():6.1f}px  max={px_err.max():7.1f}px  "
              f"exact={exact}/{n_valid}")

# Summary
print(f"\n{'='*70}")
print(f"Summary (all images, high-confidence keypoints)")
print(f"{'='*70}\n")

for vname in variants:
    errs = np.array(all_errors[vname])
    ex, tot = all_exact[vname]
    print(f"  {vname}:")
    print(f"    Exact match:  {ex}/{tot} ({100*ex/tot:.1f}%)")
    print(f"    Pixel error:  mean={errs.mean():.1f}px  median={np.median(errs):.1f}px  "
          f"max={errs.max():.1f}px  p95={np.percentile(errs, 95):.1f}px")
    print(f"    Error distribution:")
    for thresh in [0, 2, 5, 10, 20, 50]:
        count = (errs <= thresh).sum()
        print(f"      ≤{thresh:3d}px: {count:5d}/{len(errs)} ({100*count/len(errs):5.1f}%)")
    print()
