"""Debug mask shape and content differences between PT and CoreML."""

import cv2
import numpy as np
from ultralytics import YOLO

img = cv2.imread("test_images/frame_055.jpg")
h, w = img.shape[:2]
print(f"Image: {w}x{h}")

pt = YOLO("yolo11s-seg.pt")
cml = YOLO("yolo11s-seg.mlpackage")

r_pt = pt.predict(img, classes=[0], verbose=False)[0]
r_cml = cml.predict(img, classes=[0], verbose=False)[0]

print(f"PT mask raw shape: {r_pt.masks.data.shape}, range: {r_pt.masks.data.min():.3f}-{r_pt.masks.data.max():.3f}")
print(f"CML mask raw shape: {r_cml.masks.data.shape}, range: {r_cml.masks.data.min():.3f}-{r_cml.masks.data.max():.3f}")

pt_m = r_pt.masks.data[0].cpu().numpy()
cml_m = r_cml.masks.data[0].cpu().numpy()
print(f"PT mask >0.5 pixels: {(pt_m > 0.5).sum()} / {pt_m.size}")
print(f"CML mask >0.5 pixels: {(cml_m > 0.5).sum()} / {cml_m.size}")

print(f"PT masks.xy[0] polygon points: {len(r_pt.masks.xy[0])}")
print(f"CML masks.xy[0] polygon points: {len(r_cml.masks.xy[0])}")
print(f"PT orig_shape: {r_pt.masks.orig_shape}")
print(f"CML orig_shape: {r_cml.masks.orig_shape}")

# Resize both masks to image dims and compute IoU
pt_full = cv2.resize(pt_m.astype(np.float32), (w, h)) > 0.5
cml_full = cv2.resize(cml_m.astype(np.float32), (w, h)) > 0.5

inter = (pt_full & cml_full).sum()
union = (pt_full | cml_full).sum()
print(f"\nResized to {w}x{h} — IoU: {inter/union:.4f}")
print(f"  PT pixels: {pt_full.sum()}, CML pixels: {cml_full.sum()}, intersection: {inter}")

# Check if CML mask has padding from 640x640 square (letterboxing artifact)
# PT outputs 384x640 (already unpadded), CML outputs 640x640 (square)
# The CML mask likely includes padding area — need to crop it before resize
if cml_m.shape[0] == 640 and cml_m.shape[1] == 640:
    # Calculate letterbox padding for this image aspect ratio
    scale = min(640 / h, 640 / w)
    new_h, new_w = int(h * scale), int(w * scale)
    pad_top = (640 - new_h) // 2
    pad_left = (640 - new_w) // 2
    print(f"\nLetterbox: scale={scale:.3f}, new={new_w}x{new_h}, pad=({pad_left},{pad_top})")

    # Crop CML mask to content area (remove padding)
    cml_cropped = cml_m[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    cml_full2 = cv2.resize(cml_cropped.astype(np.float32), (w, h)) > 0.5

    inter2 = (pt_full & cml_full2).sum()
    union2 = (pt_full | cml_full2).sum()
    print(f"  After unpad — IoU: {inter2/union2:.4f}")
    print(f"  PT pixels: {pt_full.sum()}, CML pixels: {cml_full2.sum()}, intersection: {inter2}")
