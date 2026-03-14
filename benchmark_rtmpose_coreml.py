"""Phase 2.5 – Benchmark CoreML RTMPose latency."""

import time

import cv2
import numpy as np
import coremltools as ct

N_ITERS = 200

# Prepare input: random (1, 3, 384, 288) float32 tensor
np.random.seed(42)
test_input = np.random.randn(1, 3, 384, 288).astype(np.float32)


def bench(compute_units, label):
    model = ct.models.MLModel("rtmpose.mlpackage", compute_units=compute_units)
    spec = model.get_spec()
    input_name = spec.description.input[0].name

    # Warmup
    for _ in range(10):
        model.predict({input_name: test_input})

    times = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        model.predict({input_name: test_input})
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_ms = np.array(times) * 1000
    print(f"  {label:>15s}: median {np.median(times_ms):6.2f}ms  "
          f"mean {np.mean(times_ms):6.2f}ms  "
          f"p95 {np.percentile(times_ms, 95):6.2f}ms  "
          f"({1000/np.median(times_ms):.0f} FPS)")


print(f"Benchmarking RTMPose CoreML ({N_ITERS} iterations)\n")
bench(ct.ComputeUnit.ALL, "ALL (ANE+GPU+CPU)")
bench(ct.ComputeUnit.CPU_AND_GPU, "CPU+GPU")
bench(ct.ComputeUnit.CPU_ONLY, "CPU only")
