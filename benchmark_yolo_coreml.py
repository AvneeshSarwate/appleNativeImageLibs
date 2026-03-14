"""Phase 1.5 – Benchmark CoreML YOLO-seg latency."""

import time

import cv2
import numpy as np
import coremltools as ct
from PIL import Image

N_ITERS = 200
IMG_PATH = "test_images/frame_055.jpg"

img = cv2.imread(IMG_PATH)
h, w = img.shape[:2]

# Prepare letterboxed PIL image (640x640) — matches what ultralytics feeds CoreML
scale = min(640 / h, 640 / w)
new_h, new_w = int(h * scale), int(w * scale)
resized = cv2.resize(img, (new_w, new_h))
canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
pad_top = (640 - new_h) // 2
pad_left = (640 - new_w) // 2
canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def bench(compute_units, label):
    model = ct.models.MLModel(
        "yolo11s-seg.mlpackage",
        compute_units=compute_units,
    )
    spec = model.get_spec()
    input_name = spec.description.input[0].name

    # Warmup
    for _ in range(10):
        model.predict({input_name: pil_img})

    times = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        model.predict({input_name: pil_img})
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_ms = np.array(times) * 1000
    print(f"  {label:>15s}: median {np.median(times_ms):6.2f}ms  "
          f"mean {np.mean(times_ms):6.2f}ms  "
          f"p95 {np.percentile(times_ms, 95):6.2f}ms  "
          f"({1000/np.median(times_ms):.0f} FPS)")


print(f"Benchmarking YOLO-seg CoreML ({N_ITERS} iterations)\n")
bench(ct.ComputeUnit.ALL, "ALL (ANE+GPU+CPU)")
bench(ct.ComputeUnit.CPU_AND_GPU, "CPU+GPU")
bench(ct.ComputeUnit.CPU_ONLY, "CPU only")
