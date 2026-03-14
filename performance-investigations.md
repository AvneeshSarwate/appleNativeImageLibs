# Performance Investigations

## Hardware
- Apple M1 Max MacBook Pro
- macOS 14.8.4 (Sonoma)
- Xcode 16.2 / Swift 6.0.3 / macOS 15.2 SDK

## Models
- **YOLO11s-seg**: person segmentation, 640x640 input, fp16 CoreML (exported via ultralytics)
- **RTMPose RTMW-x-l**: 133 whole-body keypoints, 384x288 input, fp32 CoreML (converted via onnx2torch + coremltools)

## Input
- 1080p video (`foot_dance.mp4`), mostly single person
- ffmpeg pipes raw BGRA frames to simulate camera CVPixelBuffer delivery

---

## Phase 0: CPU Baselines

ONNX Runtime on CPU, 100 frames from video.

| Stage | Median |
|-------|--------|
| Frame decode (seeking) | 62.1ms |
| YOLO-seg | 71.5ms |
| RTMPose (rtmlib, includes YOLOX detector) | 371.8ms |
| **Total** | **503.6ms (2.0 FPS)** |

---

## Phase 1-3: CoreML Conversion and Validation

### YOLO-seg CoreML
- Direct ultralytics export (`model.export(format='coreml')`) initially failed due to numpy 2.4 incompatibility with coremltools. Fixed by pinning `numpy<2.4`.
- Mask validation required unpadding letterbox from 640x640 to content region. After fix, IoU ~0.87 vs PyTorch (boundary pixel differences from fp16).
- Standalone inference: **12.5ms (ALL), 21.2ms (CPU+GPU), 109ms (CPU only)**

### RTMPose CoreML
- coremltools 9.0 dropped ONNX converter support (removed in coremltools 6.0). Used `onnx2torch` to go ONNX → PyTorch → CoreML.
- onnx2torch failed on Clip ops with empty-string max inputs. Fixed by patching ONNX graph to add explicit float-max constants before conversion.
- **fp16 conversion caused catastrophic accuracy loss** — see dedicated section below. Converted with fp32 precision instead. Result: 0.00px error vs ONNX reference across all test images.
- YOLOX detector conversion failed (`less` op in NMS not supported by coremltools). Not needed since YOLO11-seg handles person detection.
- Standalone inference: **15.6ms (ALL), 15.2ms (CPU+GPU), 94.6ms (CPU only)**

### End-to-end validation
- CoreML pipeline (YOLO-seg → crop → RTMPose) vs ONNX pipeline: **0.00px keypoint error** across 12 test images when using identical preprocessing.

---

## Profiling: Where Time Goes

### Python unoptimized pipeline (124ms, 8 FPS)

| Stage | Median | % |
|-------|--------|---|
| Frame decode (seeking) | 63.1ms | 50.7% |
| YOLO preprocess | 1.4ms | 1.1% |
| YOLO inference | 13.2ms | 10.6% |
| YOLO postprocess | 0.08ms | 0.1% |
| Mask decode + resize to full res | 8.0ms | 6.4% |
| Contour extraction (full res) | 1.4ms | 1.1% |
| RTMPose preprocess | 1.3ms | 1.0% |
| RTMPose inference | 32.3ms | 26.0% |
| RTMPose postprocess | 0.14ms | 0.1% |
| Ultralytics wrapper overhead | +9.7ms | (comparison) |

Key findings:
- Frame decode was 50% of time due to random seeking in .mp4
- Mask decode did two unnecessary resizes (160→640→1920x1080) just to run findContours
- Ultralytics wrapper added ~10ms of pure Python overhead
- RTMPose inference was ~2x its standalone benchmark (contention, see below)

### Optimizations applied

1. **Sequential frame reads** instead of seeking: 63ms → 4.5ms
2. **Contour at proto resolution** (160x90): extract at native model output resolution, scale contour points to image coords. Eliminated two cv2.resize calls. 9.4ms → 0.28ms
3. **No ultralytics wrapper** at inference time: raw CoreML + lightweight postprocessing
4. **Preallocated letterbox canvas**: reuse between frames

### Python optimized pipeline (52.8ms, 18.9 FPS)

| Stage | Median | % |
|-------|--------|---|
| Frame read (sequential) | 4.5ms | 8.5% |
| YOLO preprocess | 0.8ms | 1.5% |
| YOLO inference | 17.1ms | 32.3% |
| YOLO postprocess | 0.08ms | 0.1% |
| Contour (160px) | 0.28ms | 0.5% |
| RTMPose preprocess | 0.8ms | 1.5% |
| RTMPose inference | 30.9ms | 58.6% |
| RTMPose postprocess | 0.18ms | 0.3% |

90% of time is now model inference. RTMPose at 31ms is the dominant cost.

---

## Swift Pipeline

Ported to Swift for lower overhead: vImage for resize, direct CoreML API, no Python/numpy.

### Swift sync (Xcode 14.3, macOS 13 SDK): 46.7ms, 21.4 FPS

| Stage | Median | % |
|-------|--------|---|
| Frame wait (pipelined) | 0.0ms | 0% |
| YOLO preprocess | 2.8ms | 5.9% |
| YOLO inference | 11.7ms | 25.1% |
| Contour (160px) | 0.67ms | 1.4% |
| RTMPose total | 31.4ms | 67.2% |

### Swift async (Xcode 16.2, macOS 15 SDK): 43.1ms, 23.2 FPS

| Stage | Median | % |
|-------|--------|---|
| Frame wait (pipelined) | 0.0ms | 0% |
| YOLO preprocess | 2.6ms | 6.0% |
| YOLO inference | 8.0ms | 18.6% |
| Contour (160px) | 0.72ms | 1.7% |
| RTMPose total | 31.5ms | 73.0% |

YOLO inference improved from 11.7ms to 8.0ms with the newer SDK. RTMPose unchanged.

---

## Model Switching Penalty Investigation

### The problem

Running YOLO then RTMPose sequentially costs ~10ms more than their standalone sum:
- Standalone: YOLO 7.9ms + RTMPose 15.2ms = **23.1ms**
- Alternating: YOLO 8.4ms + RTMPose 24.8ms = **33.2ms**
- The penalty is always on RTMPose (the second model). YOLO is unaffected.

### What we tested

| Configuration | YOLO | RTMPose | Total | Penalty |
|---------------|------|---------|-------|---------|
| Standalone sum | 7.9ms | 15.2ms | 23.1ms | — |
| Alternating (both ALL) | 8.4ms | 24.8ms | 33.2ms | +10.1ms |
| RTMPose pinned to CPU+GPU | 8.3ms | 24.6ms | 32.9ms | +9.9ms |
| YOLO(CPU+NE) + RTMPose(CPU+GPU) | 11.3ms | 25.9ms | 37.2ms | +11.4ms |
| Both CPU+NE | 11.4ms | 87.9ms | 99.3ms | N/A |
| outputBackings (preallocated) | 10.9ms | 26.0ms | 36.8ms | +11.4ms |
| Async prediction API (macOS 14+) | 8.7ms | 25.8ms | 34.6ms | +10.6ms |

**Nothing eliminated the penalty.** It persists across all compute unit assignments, API modes, and even process boundaries.

### Two-process split test

Tested running YOLO in a parent process and RTMPose in a separate child process (communicating via pipes) to determine if the penalty is CoreML's internal serialization or a hardware-level cost.

| Config | YOLO | RTMPose | Total |
|--------|------|---------|-------|
| Single-process alternating | 10.3ms | 26.3ms | 36.6ms |
| Two-process sequential | 9.6ms | 25.9ms | 35.3ms |

**Result: no improvement.** RTMPose is 25.9ms in the child process — essentially the same as 26.3ms in-process. The penalty persists even across process boundaries, confirming it is a hardware-level cost, not CoreML's internal serialization.

IPC overhead was negligible (-0.2ms, within noise).

### Root cause

Based on research (machinethink.net, hollance/neural-engine, WWDC sessions, Apple forums) and the two-process test:

1. **Hardware-level cost, not software**: The penalty persists across process boundaries, ruling out CoreML internal fences/barriers as the cause. It's in the Metal driver or silicon.

2. **YOLO runs on ANE (fp16), RTMPose runs on GPU (fp32)**. They use different hardware, but there is a ~10ms stall when transitioning from ANE activity to GPU activity — likely unified memory bus contention or GPU pipeline warmup after ANE workload completion.

3. **GPU command buffer serialization**: CoreML encodes each model into a separate Metal command buffer via MPS/MPSGraph. Each `prediction()` call is a full synchronous CPU→GPU round-trip: encode, submit, wait.

4. **The async API doesn't help** because it only moves the blocking off the calling thread. The underlying Metal submission is still serialized. The async API helps throughput for concurrent predictions of the *same* model, not alternating different models.

5. **Apple Silicon GPUs have only 2x concurrency across separate command queues** (per metal-benchmarks). Two GPU workloads cannot truly run in parallel.

---

## fp16 RTMPose Accuracy Investigation

### The problem

Converting RTMPose to fp16 precision causes ~5% of keypoints to have catastrophic position errors (hundreds of pixels off).

### Results (12 test images, 1594 high-confidence keypoints)

| Variant | Exact match | ≤2px | ≤5px | ≤50px | Max error |
|---------|-------------|------|------|-------|-----------|
| fp32 (ALL) | 100% | 100% | 100% | 100% | 0.0px |
| fp16 (ALL/GPU) | 62.5% | 80.8% | 91.1% | 94.8% | 1487px |
| fp16 (CPU+NE/ANE) | 58.2% | 76.5% | 88.7% | 94.2% | 1487px |

### Root cause analysis (deep dive)

The error distribution is **bimodal**: most keypoints are exact or very close, but ~5% have catastrophic errors up to 1487px (nearly the full image width).

**Mechanism**: fp16 quantization introduces ~0.07 max absolute error in logit values. SimCC logit distributions have tiny margins between 1st and 2nd place (median margin ~0.002). When margin < ~0.07 (the fp16 noise floor), the argmax flips to a random position — typically position 0 or the max index. **Every single failure (100/100 tested) had min_margin < 0.1.**

**No clean filtering threshold exists**: 72.6% of *correct* keypoints also have margin < 0.1 — they just happened not to flip. The distributions of good vs bad keypoints completely overlap. There is no confidence or margin threshold that catches all failures without flagging >70% of good keypoints. Post-processing cannot reliably distinguish failures from correct predictions.

**Axis breakdown**: 64% of failures affect only the X axis, 34% only Y axis, 2% both.

Running on ANE (CPU+NE) vs GPU produces similar results — slightly worse on ANE. The problem is in the model weights/computation precision, not the hardware.

### Per-body-part failure rates (RTMW-x-l, fp16)

| Group | Failure rate (>50px) | ≤1px | Notes |
|-------|---------------------|------|-------|
| Face (68 kpts) | 5.4% | 89.4% | Worst — but irrelevant for videos without faces |
| Body (17 kpts) | 7.8% | — | Hips/shoulders affected |
| Left hand (21 kpts) | 2.8% | 95.2% | |
| Right hand (21 kpts) | 2.4% | 92.5% | |
| Feet (6 kpts) | **0.0%** | **100%** | Never fail — sharp, unambiguous peaks |
| Lower body (hips/knees/ankles) | **0.0%** (>50px) | 81.9% | Worst case 35px, no catastrophic errors |

**For video content focused on lower body / feet (e.g., dance)**: feet and lower body have zero catastrophic failures. The failures concentrate in face (irrelevant if not in frame), hands (2-3%), and upper body (1.4%). A simple temporal filter ("if keypoint jumps >30px between frames, hold previous value") would catch the rare non-face failures.

### fp16 with no switching penalty

### fp16 with no switching penalty

Despite the accuracy issues, fp16 on CPU+NE **does eliminate the switching penalty**:

| Config | YOLO | RTMPose | Total |
|--------|------|---------|-------|
| fp32(ALL) alternating | 8.4ms | 24.8ms | 33.2ms |
| **fp16(CPU+NE) alternating** | **9.7ms** | **16.2ms** | **25.9ms** |

RTMPose actually gets faster in alternation (16.2ms vs 20.5ms standalone) because the ANE is warmed up from YOLO.

---

## Mixed Precision Investigation

### Approach

coremltools supports per-op precision control via `FP16ComputePrecision(op_selector=...)`. The `op_selector` callback receives each MIL op and returns `True` (fp16) or `False` (fp32).

### Finding the precision boundary

Tested multiple configurations on random input (133 keypoints × 2 axes = 266 measurements):

| Configuration | Exact match | Max error |
|---------------|-------------|-----------|
| Full fp32 | 90.6% | 2px |
| Full fp16 | 82.3% | 385px |
| fp16 matmul only fp32 | 82.7% | 386px |
| fp16 head math ops fp32 | 82.3% | 385px |
| **fp16 conv only, everything else fp32** | **92.5%** | **2px** |
| fp16 except last 50 ops fp32 | 83.8% | 386px |

**Key finding**: keeping only the 122 conv ops in fp16 (and everything else fp32) achieves the same accuracy as full fp32. The conv layers are the backbone — they produce large tensors where fp16 rounding is harmless. All other ops (mul, sigmoid, div, matmul, etc.) need fp32 because they participate in the SimCC head's normalization and projection, where small rounding errors cascade into argmax flips.

Keeping the head ops fp32 while the backbone convs are fp16 does NOT help — the fp16 rounding happens upstream in the backbone features and propagates through the fp32 head unchanged. Only keeping the convs (and nothing else) in fp16 works because the cast from fp16→fp32 at the conv output boundary preserves enough precision.

### Real image accuracy (12 images, 1594 high-confidence keypoints)

| Variant | Exact | ≤1px | ≤2px | Max error |
|---------|-------|------|------|-----------|
| fp32 (ALL) | 100% | 100% | 100% | 0px |
| **mixed fp16-conv (ALL)** | **84.5%** | **99.2%** | **99.3%** | **287px** |
| mixed fp16-conv (CPU+NE) | 73.8% | 95.7% | 96.5% | 383px |
| fp16 (ALL) | 62.5% | 90.5% | 92.2% | 383px |

Mixed precision is a big improvement over full fp16 (99.2% vs 90.5% within 1px), but still has rare outliers (0.4% of keypoints >5px off, max 287px).

### Speed impact

| Model | Standalone | Notes |
|-------|-----------|-------|
| fp32 (ALL) | 14.7ms | GPU |
| mixed (ALL) | 14.8ms | GPU — fp32 ops prevent ANE |
| mixed (CPU+NE) | 81.5ms | ANE can't run fp32 ops efficiently |
| fp16 (ALL) | 32.7ms | GPU (slower — likely memory layout overhead) |
| fp16 (CPU+NE) | 20.5ms | ANE |

**The mixed model doesn't improve speed.** The fp32 non-conv ops keep it on GPU, so it runs at the same 14.8ms as full fp32. The switching penalty remains (~10ms when alternating with YOLO).

### Core tension

The ANE requires all-fp16, but all-fp16 breaks SimCC accuracy. Mixed precision (fp16 conv + fp32 rest) preserves accuracy but stays on GPU. There is no way to get partial-fp32 ops onto the ANE — it's all-or-nothing.

### Conclusion

Mixed precision is useful if you can tolerate 0.4% outlier keypoints (99.2% within 1px is good for most applications). But it provides no speed benefit over full fp32 because both end up on GPU with the same switching penalty.

---

## All-GPU Configuration (Switching Penalty Fix)

### Discovery

The ~10ms switching penalty was specifically an **ANE→GPU transition cost**. When YOLO ran on ANE (fp16, `.all` compute units) and RTMPose ran on GPU (fp32), the GPU stalled for ~10ms at the transition.

Tested running both models on GPU only (`.cpuAndGPU`, no ANE):

| Config | YOLO | RTMPose | Total | Penalty |
|--------|------|---------|-------|---------|
| YOLO(ALL/ANE) → Pose(ALL/GPU) | 8.1ms | 24.0ms | 32.1ms | +10.1ms |
| **YOLO(CPU+GPU) → Pose(CPU+GPU)** | **9.4ms** | **14.8ms** | **24.2ms** | **~0ms** |

**The switching penalty is eliminated.** Two GPU models alternate freely. YOLO is slower on GPU (9.4ms vs 8.1ms on ANE), but RTMPose drops from 24.0ms to 14.8ms (its standalone speed). Net result: **24.2ms vs 32.1ms — 25% faster**.

### Full pipeline results

| Stage | ANE+GPU (old) | All-GPU (new) |
|-------|---------------|---------------|
| Frame wait (pipelined) | 0.0ms | 0.0ms |
| YOLO preprocess | 2.6ms | 2.6ms |
| YOLO inference | 8.0ms | 13.6ms |
| Contour (160px) | 0.7ms | 0.6ms |
| RTMPose total | 31.5ms | 23.6ms |
| **Total** | **43.1ms (23.2 FPS)** | **40.6ms (24.6 FPS)** |

---

## Performance Summary

| Pipeline | Per-frame | FPS | Speedup vs CPU |
|----------|-----------|-----|----------------|
| Python CPU (ONNX) | 503.6ms | 2.0 | 1x |
| Python CoreML (optimized) | 52.8ms | 18.9 | 9.5x |
| Swift sync (Xcode 14, ANE+GPU seq.) | 46.7ms | 21.4 | 10.8x |
| Swift async (Xcode 16, ANE+GPU seq.) | 43.1ms | 23.2 | 11.7x |
| Swift async (Xcode 16, all-GPU seq.) | 40.6ms | 24.6 | 12.4x |
| **Swift parallel (ANE+GPU, DWPose-m, 1f lag)** | **15.2ms** | **65.7** | **33x** |

---

## Ruled Out

| Approach | Result |
|----------|--------|
| Pinning models to different compute units (ANE vs GPU) | No effect on switching penalty |
| outputBackings (preallocated buffers) | No effect on switching penalty |
| Async prediction API (macOS 14+) | No effect on switching penalty (but minor overall improvement) |
| Two-process split (separate processes) | No effect — confirmed hardware-level ANE→GPU transition cost |
| RTMPose fp16 (full model) | Eliminates switching penalty but 5% of keypoints catastrophically wrong |
| RTMPose fp16 on ANE vs GPU | Same accuracy problems on both |
| Mixed precision (fp16 conv + fp32 rest) | Good accuracy (99.2% ≤1px) but same speed as fp32 — fp32 ops keep it on GPU, switching penalty remains |

## Remaining Optimization Avenues

Ranked by estimated impact-to-effort ratio:

1. **Smaller RTMPose variant** (RTMPose-s or RTMPose-m instead of RTMW-x-l): smaller model = faster inference. Easiest change — just swap the ONNX file.
2. **Skip-frame YOLO**: run YOLO every 2-3 frames, reuse bbox, run RTMPose every frame. No switching penalty on the all-GPU setup anyway, but saves YOLO inference time on skipped frames.
3. **CoreML Pipeline model**: combine YOLO + RTMPose into single .mlpackage. Less relevant now that all-GPU eliminated the switching penalty, but could still reduce per-prediction overhead.
4. **MPSGraph direct**: bypass CoreML entirely. Full control over command buffers. High implementation effort. Less motivated now that all-GPU works well.

## Current Best Configuration

**Parallel: YOLO(ANE) + DWPose-m(GPU), 1-frame bbox lag.** Both models run concurrently on different hardware. DWPose-m crops the current frame using the previous frame's YOLO bounding box. Frame time = max(YOLO, Pose) ≈ 15.2ms.

- Full pipeline: **15.2ms (65.7 FPS)**, p95: 16.0ms
- GPU usage: ~7ms per frame (DWPose-m only — GPU is mostly free for other work)
- Accuracy: **100% exact** keypoint match vs ONNX reference
- 133 whole-body keypoints (DWPose-m, AP 60.6 vs 70.1 for large model)
- 1-frame bbox lag acceptable for slow/moderate dancer movement
- **33x speedup** vs original Python CPU baseline

---

## Other Tested Configurations

### Large Model (RTMW-x-l)

### Maximum speed (both GPU, no ANE)

**fp32 RTMPose (CPU+GPU) + YOLO-seg (CPU+GPU)**, async predictions, 1-frame pipelined frame reading. Both models on GPU, no ANE — eliminates the ANE→GPU switching penalty.

- Per-frame: **40.6ms (24.6 FPS)** in full pipeline with preprocessing
- Pure inference: **24.2ms** (9.4ms YOLO + 14.8ms RTMPose, zero switching penalty)
- Accuracy: **100% exact** keypoint match vs ONNX reference
- GPU usage: ~24ms per frame (both models)
- Bottleneck: RTMPose inference (58% of frame time)

### GPU-light (YOLO on ANE, RTMPose on GPU)

**fp32 RTMPose (ALL/GPU) + YOLO-seg (ALL/ANE)**. YOLO offloads to ANE, freeing ~9ms of GPU time per frame. Includes the ~10ms ANE→GPU switching penalty.

- Per-frame: ~43ms (~23 FPS) in full pipeline
- Pure inference: **33.2ms** (8.4ms YOLO on ANE + 24.8ms RTMPose on GPU, includes 10ms switching penalty)
- Accuracy: **100% exact** keypoint match vs ONNX reference
- GPU usage: ~15ms per frame (RTMPose only)
- Still ~30 FPS at the inference level — viable when running alongside GPU-heavy applications like TouchDesigner
- Note: the 10ms penalty is GPU pipeline setup latency, not sustained GPU compute — TD rendering is only competing with the ~15ms of actual RTMPose GPU work

### GPU-light with smaller model (YOLO on ANE, DWPose-m on GPU)

**fp32 DWPose-m (CPU+GPU) + YOLO-seg (ALL/ANE)**. DWPose-m is 8x fewer FLOPs than RTMW-x-l (2.2G vs 17.7G), 133 keypoints, 256x192 input. The ANE→GPU switching penalty is only ~3ms (vs 10ms for the large model).

- Pure inference: **26.9ms** (16.6ms YOLO on ANE + 10.3ms DWPose-m on GPU, includes ~3ms switching penalty)
- Accuracy: **100% exact** keypoint match vs ONNX reference (lower overall AP than large model: 60.6 vs 70.1)
- GPU usage: ~7ms per frame (DWPose-m only — half the GPU time of the large model)
- DWPose-m standalone on GPU: 7.0ms

### Zero-GPU with smaller model (YOLO on ANE, DWPose-m on CPU)

**fp32 DWPose-m (CPU only) + YOLO-seg (CPU+NE/ANE)**. No GPU usage at all — GPU is entirely free for TouchDesigner or other rendering.

- Pure inference: **26.0ms** (9.3ms YOLO on ANE + 16.7ms DWPose-m on CPU, zero switching penalty)
- Accuracy: **100% exact** keypoint match vs ONNX reference
- GPU usage: **zero**

### Parallel with 1-frame bbox lag (YOLO on ANE, DWPose-m on GPU, concurrent)

**fp32 DWPose-m (CPU+GPU) + YOLO-seg (ALL/ANE), running in parallel with 1-frame delayed bounding box.** YOLO and DWPose-m run on different hardware (ANE vs GPU) simultaneously. DWPose-m uses the previous frame's YOLO bbox to crop the current frame — acceptable for slow/moderate movement.

- Full pipeline: **15.2ms (65.7 FPS)** — frame time = max(YOLO, Pose) instead of sum
- YOLO total (ANE): 13.6ms
- DWPose-m total (GPU): 15.0ms
- Parallel savings: 13.3ms vs sequential (28.5ms)
- p95: 16.0ms — very stable
- Accuracy: **100% exact** keypoint match vs ONNX reference
- GPU usage: ~7ms per frame (DWPose-m only)
- **33x speedup** vs original Python CPU baseline (503.6ms → 15.2ms)
