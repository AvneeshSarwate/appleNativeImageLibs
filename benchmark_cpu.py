"""Phase 0.3 – Measure CPU pipeline latency on video frames."""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from rtmlib import Wholebody

VIDEO = "foot_dance.mp4"
N_FRAMES = 100

cap = cv2.VideoCapture(VIDEO)
assert cap.isOpened(), f"Cannot open {VIDEO}"

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames @ {fps_video:.1f}fps")

# Sample N_FRAMES evenly
sample_indices = np.linspace(0, total_frames - 1, N_FRAMES, dtype=int)

# Load models
yolo = YOLO("yolo11s-seg.pt")
pose = Wholebody(mode="performance", backend="onnxruntime", device="cpu",
                 to_openpose=False)

timings = {"capture": [], "yolo": [], "pose": [], "total": []}

for i, frame_idx in enumerate(sample_indices):
    # Capture
    t_cap0 = time.perf_counter()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    t_cap1 = time.perf_counter()
    if not ret:
        continue

    # YOLO-seg
    t_yolo0 = time.perf_counter()
    results = yolo.predict(frame, classes=[0], verbose=False)
    t_yolo1 = time.perf_counter()

    # RTMPose (runs on all detected persons internally)
    t_pose0 = time.perf_counter()
    keypoints, scores = pose(frame)
    t_pose1 = time.perf_counter()

    t_total = t_pose1 - t_cap0

    timings["capture"].append(t_cap1 - t_cap0)
    timings["yolo"].append(t_yolo1 - t_yolo0)
    timings["pose"].append(t_pose1 - t_pose0)
    timings["total"].append(t_total)

    if (i + 1) % 25 == 0:
        print(f"  Processed {i+1}/{N_FRAMES} frames...")

cap.release()

# Report
print("\n" + "=" * 55)
print("CPU Pipeline Benchmark (per-frame, milliseconds)")
print("=" * 55)
for key in ["capture", "yolo", "pose", "total"]:
    arr = np.array(timings[key]) * 1000
    print(f"  {key:>8s}:  median {np.median(arr):7.1f}ms  "
          f"mean {np.mean(arr):7.1f}ms  "
          f"p95 {np.percentile(arr, 95):7.1f}ms")

total_ms = np.array(timings["total"]) * 1000
achievable_fps = 1000.0 / np.median(total_ms)
print(f"\nAchievable FPS (median): {achievable_fps:.1f}")
print(f"Achievable FPS (mean):   {1000.0 / np.mean(total_ms):.1f}")
