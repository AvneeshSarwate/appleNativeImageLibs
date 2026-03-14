"""Test DWPose-m accuracy on full-frame input (no person crop) vs cropped input."""

import numpy as np
import cv2
import coremltools as ct
from ultralytics import YOLO
from pathlib import Path

SIMCC_SPLIT = 2.0

print("Loading models...")
yolo = YOLO("yolo11s-seg.pt")
pose = ct.models.MLModel("dwpose_m.mlpackage", compute_units=ct.ComputeUnit.ALL)
spec = pose.get_spec()
pose_in = spec.description.input[0].name
pose_outs = sorted([o.name for o in spec.description.output])

# RTMPose input: (1, 3, 256, 192)
TARGET_H, TARGET_W = 256, 192


def preprocess_crop(img, bbox):
    """Standard: crop person, resize to 192x256, normalize."""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1
    s = max(bw/TARGET_W, bh/TARGET_H) * 1.25
    x1c = max(0, int(cx - TARGET_W*s/2))
    y1c = max(0, int(cy - TARGET_H*s/2))
    x2c = min(w, int(cx + TARGET_W*s/2))
    y2c = min(h, int(cy + TARGET_H*s/2))
    crop = img[y1c:y2c, x1c:x2c]
    resized = cv2.resize(crop, (TARGET_W, TARGET_H))
    blob = resized.astype(np.float32)
    blob[:,:,0] = (blob[:,:,0] - 123.675) / 58.395
    blob[:,:,1] = (blob[:,:,1] - 116.28) / 57.12
    blob[:,:,2] = (blob[:,:,2] - 103.53) / 57.375
    blob = np.transpose(blob, (2,0,1))[np.newaxis].astype(np.float32)
    return blob, s, x1c, y1c


def preprocess_fullframe(img):
    """Full frame: letterbox to 192x256, normalize."""
    h, w = img.shape[:2]
    scale = min(TARGET_H / h, TARGET_W / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((TARGET_H, TARGET_W, 3), 114, dtype=np.uint8)
    pad_top = (TARGET_H - new_h) // 2
    pad_left = (TARGET_W - new_w) // 2
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    blob = canvas.astype(np.float32)
    blob[:,:,0] = (blob[:,:,0] - 123.675) / 58.395
    blob[:,:,1] = (blob[:,:,1] - 116.28) / 57.12
    blob[:,:,2] = (blob[:,:,2] - 103.53) / 57.375
    blob = np.transpose(blob, (2,0,1))[np.newaxis].astype(np.float32)
    return blob, scale, pad_left, pad_top


def decode_crop(simcc_x, simcc_y, scale, x1c, y1c):
    kx = np.argmax(simcc_x[0], axis=-1) / SIMCC_SPLIT
    ky = np.argmax(simcc_y[0], axis=-1) / SIMCC_SPLIT
    conf = np.minimum(np.max(simcc_x[0], axis=-1), np.max(simcc_y[0], axis=-1))
    return np.stack([kx * scale + x1c, ky * scale + y1c], axis=-1), conf


def decode_fullframe(simcc_x, simcc_y, img_h, img_w, scale, pad_left, pad_top):
    kx = np.argmax(simcc_x[0], axis=-1) / SIMCC_SPLIT
    ky = np.argmax(simcc_y[0], axis=-1) / SIMCC_SPLIT
    conf = np.minimum(np.max(simcc_x[0], axis=-1), np.max(simcc_y[0], axis=-1))
    kx_img = (kx - pad_left) / scale
    ky_img = (ky - pad_top) / scale
    return np.stack([kx_img, ky_img], axis=-1), conf


image_paths = sorted(Path("test_images").glob("*.jpg"))
print(f"Testing on {len(image_paths)} images\n")

LOWER_BODY = [11, 12, 13, 14, 15, 16] + list(range(17, 23))
UPPER_BODY = [5, 6, 7, 8, 9, 10]
HANDS = list(range(91, 133))
YOUR_KPTS = LOWER_BODY + UPPER_BODY + HANDS
NO_FACE = [i for i in range(133) if i not in range(23, 91)]

for img_path in image_paths:
    stem = img_path.stem
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Get bbox from YOLO for cropped reference
    r = yolo.predict(img, classes=[0], verbose=False)[0]
    if len(r.boxes) == 0:
        print(f"  {stem}: no detection, skipping")
        continue
    idx = r.boxes.conf.argmax().item()
    bbox = r.boxes.xyxy[idx].cpu().numpy()

    # Cropped inference (reference)
    blob_crop, sc, x1c, y1c = preprocess_crop(img, bbox)
    out_crop = pose.predict({pose_in: blob_crop})
    kpts_crop, conf_crop = decode_crop(
        out_crop[pose_outs[0]], out_crop[pose_outs[1]], sc, x1c, y1c)

    # Full-frame inference
    blob_full, sf, pl, pt_ = preprocess_fullframe(img)
    out_full = pose.predict({pose_in: blob_full})
    kpts_full, conf_full = decode_fullframe(
        out_full[pose_outs[0]], out_full[pose_outs[1]], h, w, sf, pl, pt_)

    # Compare (only high-confidence keypoints from cropped reference)
    valid = conf_crop > 0.3

    for group_name, indices in [("Lower body+feet", LOWER_BODY),
                                 ("Upper body", UPPER_BODY),
                                 ("Hands", HANDS),
                                 ("All (no face)", NO_FACE)]:
        mask = np.zeros(133, dtype=bool)
        mask[indices] = True
        both = valid & mask
        if both.sum() == 0:
            continue
        diff = kpts_full[both] - kpts_crop[both]
        px_err = np.sqrt((diff**2).sum(axis=-1))
        within5 = (px_err <= 5).sum()
        within20 = (px_err <= 20).sum()
        print(f"  {stem} {group_name:20s}: {both.sum():3d} kpts  "
              f"mean={px_err.mean():6.1f}px  max={px_err.max():6.1f}px  "
              f"<=5px={within5}/{both.sum()}  <=20px={within20}/{both.sum()}")

    print()

# Summary: visualize where keypoints land on a full-frame input
print("=" * 70)
print("Full-frame keypoint positions on last image (sanity check)")
print("=" * 70)
print(f"Image: {w}x{h}")
print(f"Letterbox scale: {sf:.4f}, pad: ({pl}, {pt_})")
n_in_frame = ((kpts_full[:, 0] > 0) & (kpts_full[:, 0] < w) &
              (kpts_full[:, 1] > 0) & (kpts_full[:, 1] < h)).sum()
print(f"Keypoints within frame bounds: {n_in_frame}/133")
print(f"Full-frame confidence: mean={conf_full.mean():.3f}  "
      f"min={conf_full.min():.3f}  max={conf_full.max():.3f}")
print(f"Cropped confidence:    mean={conf_crop.mean():.3f}  "
      f"min={conf_crop.min():.3f}  max={conf_crop.max():.3f}")
