"""Compare RTMPose variants on real images: fp32, fp16, and mixed precision."""

import numpy as np
import cv2
import coremltools as ct
import onnxruntime as ort
from ultralytics import YOLO
from pathlib import Path

SIMCC_SPLIT = 2.0

print("Loading models...")
onnx_sess = ort.InferenceSession("rtmpose_fixed.onnx", providers=["CPUExecutionProvider"])
yolo = YOLO("yolo11s-seg.pt")

variants = {
    "fp32 (ALL)":      ct.models.MLModel("rtmpose.mlpackage", compute_units=ct.ComputeUnit.ALL),
    "fp16 (ALL)":      ct.models.MLModel("rtmpose_fp16.mlpackage", compute_units=ct.ComputeUnit.ALL),
    "mixed (ALL)":     ct.models.MLModel("rtmpose_mixed.mlpackage", compute_units=ct.ComputeUnit.ALL),
    "mixed (CPU+NE)":  ct.models.MLModel("rtmpose_mixed.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE),
}

# Get input/output names for each
model_io = {}
for vname, model in variants.items():
    spec = model.get_spec()
    model_io[vname] = (
        spec.description.input[0].name,
        sorted([o.name for o in spec.description.output])
    )


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
    return np.transpose(blob, (2,0,1))[np.newaxis].astype(np.float32)


def decode(simcc_x, simcc_y):
    return np.argmax(simcc_x[0], axis=-1), np.argmax(simcc_y[0], axis=-1)


image_paths = sorted(Path("test_images").glob("*.jpg"))
print(f"Comparing on {len(image_paths)} images (high-confidence keypoints, conf > 0.3)\n")

all_errors = {v: [] for v in variants}

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    r = yolo.predict(img, classes=[0], verbose=False)[0]
    if len(r.boxes) == 0:
        continue
    idx = r.boxes.conf.argmax().item()
    bbox = r.boxes.xyxy[idx].cpu().numpy()
    blob = preprocess(img, bbox)

    onnx_outs = onnx_sess.run(None, {"input": blob})
    ref_x, ref_y = decode(onnx_outs[0], onnx_outs[1])
    conf = np.minimum(np.max(onnx_outs[0][0], axis=-1), np.max(onnx_outs[1][0], axis=-1))
    valid = conf > 0.3

    for vname, model in variants.items():
        in_name, out_names = model_io[vname]
        preds = model.predict({in_name: blob})
        pred_x, pred_y = decode(preds[out_names[0]], preds[out_names[1]])
        err_x = np.abs(pred_x[valid].astype(float) - ref_x[valid].astype(float))
        err_y = np.abs(pred_y[valid].astype(float) - ref_y[valid].astype(float))
        px_err = np.sqrt(err_x**2 + err_y**2) / SIMCC_SPLIT  # in pixel coords
        all_errors[vname].extend(px_err.tolist())

# Summary
print(f"{'='*70}")
print(f"Results (all images, high-confidence keypoints)")
print(f"{'='*70}\n")

for vname in variants:
    errs = np.array(all_errors[vname])
    exact = (errs < 0.01).sum()
    print(f"  {vname}:")
    print(f"    Exact: {exact}/{len(errs)} ({100*exact/len(errs):.1f}%)  "
          f"mean={errs.mean():.1f}px  max={errs.max():.1f}px  p95={np.percentile(errs, 95):.1f}px")
    for t in [0, 1, 2, 5, 50]:
        c = (errs <= t).sum()
        print(f"    ≤{t:2d}px: {c}/{len(errs)} ({100*c/len(errs):.1f}%)")
    print()
