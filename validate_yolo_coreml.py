"""Phase 1.2 – Validate CoreML YOLO-seg outputs against PyTorch reference."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TEST_DIR = Path("test_images")
image_paths = sorted(TEST_DIR.glob("*.jpg"))
print(f"Validating on {len(image_paths)} images\n")

yolo_pt = YOLO("yolo11s-seg.pt")
yolo_cml = YOLO("yolo11s-seg.mlpackage")


def unpad_mask(mask, img_h, img_w, model_size=640):
    """Remove letterbox padding from a mask and resize to original image dims."""
    scale = min(model_size / img_h, model_size / img_w)
    new_h, new_w = int(img_h * scale), int(img_w * scale)
    pad_top = (model_size - new_h) // 2
    pad_left = (model_size - new_w) // 2
    cropped = mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    return cv2.resize(cropped.astype(np.float32), (img_w, img_h)) > 0.5


results = []

for img_path in image_paths:
    stem = img_path.stem
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    r_pt = yolo_pt.predict(img, classes=[0], verbose=False)[0]
    r_cml = yolo_cml.predict(img, classes=[0], verbose=False)[0]

    n_pt, n_cml = len(r_pt.boxes), len(r_cml.boxes)

    if n_pt == 0 and n_cml == 0:
        results.append((stem, 0, 0, 1.0, 0.0))
        print(f"  {stem}: no detections — OK")
        continue

    if n_pt == 0 or n_cml == 0:
        results.append((stem, n_pt, n_cml, 0.0, 999.0))
        print(f"  {stem}: pt={n_pt} cml={n_cml} — count mismatch")
        continue

    # Match top-confidence detections
    pt_idx = r_pt.boxes.conf.argmax().item()
    cml_idx = r_cml.boxes.conf.argmax().item()

    box_err = float(np.abs(
        r_pt.boxes.xyxy[pt_idx].cpu().numpy() -
        r_cml.boxes.xyxy[cml_idx].cpu().numpy()
    ).max())

    mask_iou = 0.0
    if r_pt.masks is not None and r_cml.masks is not None:
        pt_m = r_pt.masks.data[pt_idx].cpu().numpy()
        cml_m = r_cml.masks.data[cml_idx].cpu().numpy()

        # PT mask is already unpadded (384x640), resize directly
        pt_full = cv2.resize(pt_m.astype(np.float32), (w, h)) > 0.5

        # CML mask is 640x640 with padding — unpad first
        if cml_m.shape[0] == cml_m.shape[1]:  # square = needs unpadding
            cml_full = unpad_mask(cml_m, h, w)
        else:
            cml_full = cv2.resize(cml_m.astype(np.float32), (w, h)) > 0.5

        inter = (pt_full & cml_full).sum()
        union = (pt_full | cml_full).sum()
        mask_iou = float(inter / union) if union > 0 else 1.0

    results.append((stem, n_pt, n_cml, mask_iou, box_err))
    print(f"  {stem}: pt={n_pt} cml={n_cml}  box_err={box_err:.1f}px  mask_iou={mask_iou:.4f}")

# Summary
print("\n" + "=" * 60)
print("Validation Summary")
print("=" * 60)
all_pass = True
for stem, n_pt, n_cml, iou, berr in results:
    box_ok = berr < 10
    iou_ok = iou > 0.85
    status = "PASS" if (box_ok and iou_ok) else "FAIL"
    if status != "PASS":
        all_pass = False
    print(f"  {stem}: {status}  IoU={iou:.4f}  box_err={berr:.1f}px")

ious = [r[3] for r in results]
print(f"\nMean IoU: {np.mean(ious):.4f}  Min IoU: {np.min(ious):.4f}")
print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILURES'}")
