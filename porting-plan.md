# Porting RTMPose + YOLO-seg to CoreML on Apple Silicon

## Goal

Run RTMPose-m (133 keypoints, whole-body) and YOLO11-seg (person segmentation) at 60fps on an M1 Max MacBook Pro, targeting the ANE/GPU via CoreML instead of CPU-only ONNX Runtime.

## Architecture Summary

**RTMPose (RTMW-m, 133 keypoints)**
- Backbone: CSPNeXt (conv2d, batchnorm, SiLU, concat, maxpool)
- Head: SimCC — two fully-connected layers → 1D softmax per axis per keypoint
- Input: cropped person image, 256×192 or 384×288
- Output: (K, 2) keypoint coordinates + (K,) confidence scores
- Weights: ~14M params, ~56MB fp32
- Source ONNX: download from OpenMMLab model zoo via rtmlib

**YOLO11-seg (person instance segmentation)**
- Backbone: YOLO11 CNN (conv2d, batchnorm, SiLU, concat, upsample)
- Detection head: boxes + class scores + 32 mask coefficients per detection
- Proto head: small CNN producing 32 prototype masks at (1, 32, 160, 160)
- Post-processing: NMS → matmul(coefficients, prototypes) → sigmoid → upsample → threshold
- Input: 640×640
- Output: bounding boxes, class ids, binary masks
- Source: `ultralytics` Python package, exportable to ONNX

All ops in both models are standard (conv2d, bn, silu, matmul, upsample, sigmoid, softmax, concat, maxpool). No custom kernels, no dynamic control flow. Fully ANE-compatible in theory.

---

## Phase 0: Establish CPU Baselines

### 0.1 Set up reference environment

```
pip install rtmlib onnxruntime opencv-python-headless ultralytics numpy
```

### 0.2 Generate reference outputs

Write a script `generate_baselines.py` that:
1. Loads 10-20 test images covering varied poses, lighting, occlusion, scale
2. For each image, runs:
   - rtmlib `Wholebody(mode='performance', backend='onnxruntime', device='cpu')` → saves `keypoints` and `scores` arrays as .npy
   - ultralytics `YOLO("yolo11s-seg.pt").predict(img, classes=[0])` → saves `masks.data` tensor (as numpy), `masks.xy` polygon list, `boxes.xyxy`, and raw model output tensors (detection tensor + proto tensor) as .npy
3. Also saves the preprocessed input tensors (the exact float arrays fed to each model after resize/normalize) as .npy — these become the inputs for the CoreML models during validation
4. Measures and records per-image inference time for each model on CPU

### 0.3 Measure CPU pipeline latency

Write `benchmark_cpu.py`:
1. Captures 100 frames from a 1080p video file (not webcam — deterministic)
2. Runs the full pipeline: resize → YOLO-seg → crop person → RTMPose → post-process
3. Reports per-frame timing breakdown: capture, YOLO inference, RTMPose inference, post-processing, total
4. Reports achievable FPS

**Exit criteria:** You have .npy reference files for all test images and know the CPU baseline FPS.

---

## Phase 1: Convert YOLO-seg to CoreML

YOLO-seg goes first because Ultralytics has built-in CoreML export. This is the lower-risk conversion.

### 1.1 Export via Ultralytics

```python
from ultralytics import YOLO
model = YOLO("yolo11s-seg.pt")
model.export(format="coreml", imgsz=640, half=True, nms=False)
# produces yolo11s-seg.mlpackage
```

Export with `nms=False` to get raw outputs. We'll do NMS ourselves (simpler for single-person: just take top detection).

### 1.2 Validate CoreML YOLO-seg outputs

Write `validate_yolo_coreml.py`:
1. Load the .mlpackage with `coremltools`
2. For each test image, feed the saved preprocessed input tensor
3. Compare CoreML raw outputs (detection tensor + proto tensor) against saved ONNX reference outputs
4. Metric: max absolute error per tensor, plus a functional check — do the top-1 person detection boxes match within 2px? Do the thresholded binary masks have IoU > 0.98?
5. If CoreML defaults to float16: expect small numerical differences, set tolerance accordingly

**Known risk:** There was a reported issue with YOLO-seg CoreML export producing all-zero mask prototypes. If this happens:
- Check ultralytics version (upgrade to latest)
- Try exporting with `half=False`
- If still broken, fall back to Phase 1B (manual ONNX→CoreML conversion)

### 1.3 (Fallback) Phase 1B: Manual ONNX→CoreML conversion

Only if 1.1/1.2 fail.

```python
import coremltools as ct
# Export YOLO to ONNX first
model.export(format="onnx", imgsz=640, simplify=True)
# Convert ONNX → CoreML
mlmodel = ct.convert(
    "yolo11s-seg.onnx",
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS13,
    compute_precision=ct.precision.FLOAT16,
)
mlmodel.save("yolo11s-seg.mlpackage")
```

Re-run validation from 1.2.

### 1.4 (Fallback) Phase 1C: Build CoreML model from ONNX graph manually

Only if 1B also fails (unsupported op error).

1. Open the ONNX file in Netron, identify the failing op
2. Use `coremltools.converters.mil.builder` to construct the graph op-by-op, loading weights from the ONNX file
3. This is tedious but has zero risk — all ops are standard MIL ops

### 1.5 Benchmark CoreML YOLO-seg

Write `benchmark_yolo_coreml.py`:
1. Load model with `MLComputeUnits.ALL` (ANE+GPU+CPU)
2. Run 200 inferences, report median latency
3. Also test with `MLComputeUnits.CPU_AND_GPU` and `MLComputeUnits.CPU_ONLY` to confirm ANE is actually being used (should see significant speedup with ALL vs CPU_ONLY)
4. Target: <8ms per inference for YOLO11s-seg

**Exit criteria:** YOLO-seg CoreML model produces correct outputs and runs significantly faster than CPU ONNX Runtime.

---

## Phase 2: Convert RTMPose to CoreML

Higher risk — no existing conversion pipeline. Two approaches in order of preference.

### 2.1 Attempt ONNX→CoreML via coremltools

rtmlib auto-downloads ONNX models. Locate the RTMW model file (check `~/.cache/` or rtmlib source for download path).

```python
import coremltools as ct
mlmodel = ct.convert(
    "rtmw-m_simcc-cocktail14_270e-256x192.onnx",  # or whatever the filename is
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS13,
    compute_precision=ct.precision.FLOAT16,
    inputs=[ct.TensorType(shape=(1, 3, 192, 256))],  # confirm exact shape from ONNX
)
mlmodel.save("rtmw-m.mlpackage")
```

### 2.2 Validate RTMPose CoreML outputs

Write `validate_rtmpose_coreml.py`:
1. Load the .mlpackage
2. For each test image, feed the saved preprocessed input tensor (the exact tensor rtmlib sends to the ONNX model)
3. Compare CoreML SimCC logits (pre-argmax) against ONNX reference — max absolute error
4. Compare final keypoint coordinates (post-argmax) — should be identical or within 1 pixel
5. Compare confidence scores — relative error < 0.01

### 2.3 (Fallback) PyTorch trace → CoreML

If ONNX conversion fails on an unsupported op:

1. Install MMPose, load the RTMPose-m model in PyTorch
2. Trace it: `traced = torch.jit.trace(model, dummy_input)`
3. Convert: `ct.convert(traced, inputs=[ct.TensorType(shape=...)])`

This requires installing the full MMPose stack temporarily just for the conversion step. The resulting CoreML model is standalone.

### 2.4 (Fallback) Manual MIL graph construction

If both ONNX and trace conversion fail:

1. Parse the ONNX graph programmatically with the `onnx` Python package
2. Walk the graph, build equivalent MIL ops using `coremltools.converters.mil.builder`
3. Load weight tensors from ONNX initializers
4. RTMPose-m has ~200 ops — significant work but mechanical and deterministic

A coding agent is well-suited for this: read op, emit MIL equivalent, repeat. The ONNX→MIL op mapping for standard ops is 1:1.

### 2.5 Benchmark RTMPose CoreML

Same pattern as 1.5:
1. Run 200 inferences with MLComputeUnits.ALL
2. Compare against CPU_ONLY to confirm acceleration
3. Target: <4ms per inference for RTMPose-m

**Exit criteria:** RTMPose CoreML model produces correct keypoints and runs significantly faster than CPU ONNX Runtime.

---

## Phase 3: Implement Post-Processing

The CoreML models output raw tensors. Post-processing runs on CPU and must be fast (<2ms).

### 3.1 YOLO-seg post-processing

Implement in Python (or Swift if building a native app):

```
1. Parse detection tensor → filter to person class (id=0)
2. Take highest-confidence detection (single-person assumption)
3. Extract bbox (x1, y1, x2, y2) and 32 mask coefficients
4. Matmul: coefficients @ prototypes → (160, 160) raw mask
5. Sigmoid → threshold at 0.5 → binary mask
6. Bilinear upsample to original image resolution
7. Crop to bbox
8. findContours → vector polygon
```

### 3.2 RTMPose post-processing

```
1. Parse SimCC output: two tensors of shape (K, Wx) and (K, Wy)
2. Argmax along last dimension for each → (K,) x-indices and (K,) y-indices
3. Divide by simcc_split_ratio (default 2.0) to get pixel coordinates
4. Scale back to original image coordinates using the crop/resize transform
5. Confidence = max value of each 1D distribution (or softmax then max)
```

### 3.3 Validate end-to-end pipeline

Write `validate_pipeline.py`:
1. For each test image, run the full CoreML pipeline (YOLO-seg → crop → RTMPose → post-process)
2. Compare final keypoint coordinates against the rtmlib CPU reference — max pixel error
3. Compare final mask polygon against ultralytics CPU reference — mask IoU
4. Acceptable thresholds: keypoints within 2px, mask IoU > 0.95

---

## Phase 4: Integration and 60fps Benchmark

### 4.1 Build the real-time pipeline

Write `realtime_pipeline.py`:

```
loop:
  capture frame (1080p)
  resize to 640×640 for YOLO-seg
  run YOLO-seg CoreML → get person bbox + mask
  crop frame to person bbox, resize for RTMPose
  run RTMPose CoreML → get 133 keypoints
  post-process both outputs
  (optional) render visualization
```

### 4.2 Pipeline optimizations

- **Skip-frame detection:** Run YOLO-seg every N frames (e.g., every 3), track the bbox between frames using simple momentum/Kalman filter. Run RTMPose every frame within the tracked bbox. This is how MediaPipe achieves high FPS — detect rarely, track always.
- **Double-buffering:** Prepare next frame's input tensor while current frame is on ANE.
- **Async dispatch:** CoreML supports async prediction. Dispatch YOLO-seg, start preparing RTMPose input, wait for YOLO result, dispatch RTMPose.

### 4.3 Final benchmark

1. Run on 1080p 60fps video file, measure:
   - Frames actually processed per second
   - Per-frame latency breakdown (capture, YOLO, RTMPose, post-process, render)
   - Sustained performance over 60 seconds (thermal throttling?)
2. Compare against Phase 0 CPU baseline
3. Profile in Xcode Instruments to confirm ANE utilization (optional but recommended)

**Exit criteria:** Sustained 60fps (or close to it) on M1 Max with both models running per frame.

---

## Verification Loop (for coding agent)

At every phase transition, the agent runs validation before proceeding. The loop is:

```
for each phase:
    attempt conversion/implementation
    run validation script
    if validation passes (outputs match reference within tolerance):
        record benchmark numbers
        proceed to next phase
    else:
        inspect error:
            if unsupported op → try fallback path (B → C)
            if numerical divergence > tolerance → check precision settings, try float32
            if crash/load failure → check coremltools version, check model shapes
        retry with fix
        if 3 retries fail on same phase → stop and report blocker
```

### Validation scripts summary

| Script | Phase | What it checks |
|--------|-------|----------------|
| `generate_baselines.py` | 0 | Produces .npy reference files from CPU inference |
| `benchmark_cpu.py` | 0 | Measures CPU baseline FPS |
| `validate_yolo_coreml.py` | 1 | Compares CoreML YOLO outputs vs ONNX reference |
| `benchmark_yolo_coreml.py` | 1 | Measures CoreML YOLO latency + confirms ANE use |
| `validate_rtmpose_coreml.py` | 2 | Compares CoreML RTMPose outputs vs ONNX reference |
| `benchmark_rtmpose_coreml.py` | 2 | Measures CoreML RTMPose latency + confirms ANE use |
| `validate_pipeline.py` | 3 | End-to-end output comparison (keypoints + masks) |
| `realtime_pipeline.py` | 4 | Full integration with FPS measurement |

### Tolerance thresholds

| Check | Threshold | Notes |
|-------|-----------|-------|
| Raw tensor max abs error | < 0.05 | fp16 vs fp32 accumulation differences |
| Keypoint pixel position | < 2px | After argmax, fp16 rounding rarely shifts index |
| Detection bbox | < 3px | Small coordinate rounding from fp16 |
| Mask IoU (thresholded binary) | > 0.95 | Boundary pixels may differ |
| Top-1 detection class match | Exact | Must detect person |

### Test image selection

Include at minimum:
- Standing front-facing, well-lit (easy case)
- Sitting with partial occlusion
- Arms raised / yoga pose
- Low light
- Person at edge of frame
- Small person (far from camera)
- Profile view
- Fast motion blur (if targeting video)

---

## Dependencies

**Conversion machine (can be the M1 Max itself):**
- Python 3.9-3.11 (coremltools compatibility)
- `coremltools >= 8.0`
- `onnx`
- `onnxruntime` (CPU, for baselines)
- `rtmlib`
- `ultralytics`
- `numpy`, `opencv-python`
- `torch` (only needed if fallback to PyTorch trace path)

**Runtime (M1 Max):**
- macOS 13+ (for mlprogram format)
- Python with `coremltools` (for Python inference) or Swift/ObjC CoreML APIs
- `opencv-python` (for frame capture and contour extraction)
- `numpy`

---

## Risk Summary

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| YOLO-seg CoreML export produces zero masks | Medium | Blocks Phase 1 | Fallback to ONNX→CoreML or manual MIL |
| RTMPose ONNX has unsupported op in coremltools | Low-Medium | Blocks Phase 2 | All ops are standard, but coremltools ONNX converter is deprecated. Fallback to PyTorch trace or manual MIL |
| CoreML silently falls back to CPU (no ANE speedup) | Low | Defeats purpose | Benchmark with different compute_units settings to detect. Profile in Instruments. |
| fp16 precision causes keypoint jitter | Low | Quality regression | Convert with float32, accept slight speed penalty |
| Total pipeline latency still >16.7ms | Low | Can't hit 60fps | Use skip-frame detection strategy to amortize YOLO cost |
| Thermal throttling on sustained load | Medium | FPS drops over time | Benchmark over 60+ seconds. May need to use smaller model variants. |
