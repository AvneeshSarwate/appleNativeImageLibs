"""Phase 0.2 – Generate CPU reference outputs for YOLO-seg and RTMPose."""

import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_DIR = Path("test_images")
OUT_DIR = Path("baselines")
OUT_DIR.mkdir(exist_ok=True)

image_paths = sorted(TEST_DIR.glob("*.jpg"))
assert len(image_paths) > 0, "No test images found in test_images/"
print(f"Found {len(image_paths)} test images")

# ---------------------------------------------------------------------------
# 1. YOLO-seg baselines
# ---------------------------------------------------------------------------
from ultralytics import YOLO

yolo = YOLO("yolo11s-seg.pt")  # downloads on first run

yolo_times = []
for img_path in image_paths:
    stem = img_path.stem
    img = cv2.imread(str(img_path))

    t0 = time.perf_counter()
    results = yolo.predict(img, classes=[0], verbose=False)
    t1 = time.perf_counter()
    yolo_times.append(t1 - t0)

    r = results[0]
    # Save detection outputs
    np.save(OUT_DIR / f"{stem}_yolo_boxes.npy", r.boxes.xyxy.cpu().numpy())
    np.save(OUT_DIR / f"{stem}_yolo_confs.npy", r.boxes.conf.cpu().numpy())

    if r.masks is not None:
        np.save(OUT_DIR / f"{stem}_yolo_masks.npy", r.masks.data.cpu().numpy())
    else:
        np.save(OUT_DIR / f"{stem}_yolo_masks.npy", np.array([]))

    print(f"  YOLO {stem}: {len(r.boxes)}  detections, {(t1-t0)*1000:.1f}ms")

print(f"\nYOLO-seg avg: {np.mean(yolo_times)*1000:.1f}ms  "
      f"(median {np.median(yolo_times)*1000:.1f}ms)\n")

# ---------------------------------------------------------------------------
# 2. RTMPose baselines (via rtmlib)
# ---------------------------------------------------------------------------
from rtmlib import Wholebody

# rtmlib Wholebody with ONNX backend on CPU
pose = Wholebody(mode="performance", backend="onnxruntime", device="cpu",
                 to_openpose=False)

pose_times = []
for img_path in image_paths:
    stem = img_path.stem
    img = cv2.imread(str(img_path))

    t0 = time.perf_counter()
    keypoints, scores = pose(img)
    t1 = time.perf_counter()
    pose_times.append(t1 - t0)

    np.save(OUT_DIR / f"{stem}_pose_keypoints.npy", keypoints)
    np.save(OUT_DIR / f"{stem}_pose_scores.npy", scores)

    n_persons = keypoints.shape[0] if keypoints.ndim == 3 else 0
    print(f"  Pose {stem}: {n_persons} person(s), {(t1-t0)*1000:.1f}ms")

print(f"\nRTMPose avg: {np.mean(pose_times)*1000:.1f}ms  "
      f"(median {np.median(pose_times)*1000:.1f}ms)\n")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 50)
print("Baselines saved to:", OUT_DIR)
print(f"YOLO-seg  – {len(image_paths)} images, median {np.median(yolo_times)*1000:.1f}ms")
print(f"RTMPose   – {len(image_paths)} images, median {np.median(pose_times)*1000:.1f}ms")
print(f"Baseline files: {len(list(OUT_DIR.glob('*.npy')))}")
