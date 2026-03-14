# CoreML Pose + Segmentation Pipeline

Real-time person segmentation (YOLO11-seg) and whole-body pose estimation (RTMPose, 133 keypoints) on Apple Silicon via CoreML.

## Requirements

### Python (model conversion and baselines)
- Python 3.12 (managed by `uv`)
- `uv` package manager

```bash
uv sync
```

### Swift (optimized pipeline)
- **macOS 14 (Sonoma)** or later
- **Xcode 16.2** — required for macOS 15 SDK, Swift 6, and async CoreML prediction APIs
  - Download via `brew install xcodes && xcodes install 16.2` or from [developer.apple.com/download/all](https://developer.apple.com/download/all/)
  - Ensure it's selected: `sudo xcode-select -s /Applications/Xcode-16.2.0.app/Contents/Developer`
- Swift Package Manager (included with Xcode)

Verify your toolchain:
```bash
swift --version           # Apple Swift version 6.0.3+
xcrun --show-sdk-version  # 15.2+
```

## Setup

### 1. Generate models

```bash
# Convert YOLO-seg to CoreML (via ultralytics)
uv run python -c "
from ultralytics import YOLO
YOLO('yolo11s-seg.pt').export(format='coreml', imgsz=640, half=False, nms=False)
"

# Convert RTMPose to CoreML (ONNX -> PyTorch -> CoreML)
uv run python convert_rtmpose_coreml.py
```

### 2. Generate baselines and test images

Extract test frames from a video (requires `ffmpeg`):
```bash
mkdir -p test_images
for i in 0 5 12 20 30 45 55 65 80 95 110 130; do
  ffmpeg -y -ss "$i" -i <your-video> -frames:v 1 -vf "scale=-1:1080" -q:v 2 "test_images/frame_$(printf '%03d' $i).jpg"
done
```

Run baseline generation:
```bash
uv run python generate_baselines.py
```

### 3. Build and run Swift pipeline

```bash
swift build -c release --package-path swift-pipeline
swift-pipeline/.build/release/CoreMLPipeline <video-path> [num-frames]
swift-pipeline/.build/release/CoreMLPipeline --bench  # standalone model benchmarks
```

## Performance (M1 Max, 1080p input)

| Pipeline | Per-frame | FPS |
|----------|-----------|-----|
| Python CPU (ONNX) | 503.6ms | 2.0 |
| Python CoreML | 52.8ms | 18.9 |
| Swift async CoreML | 43.1ms | 23.2 |

## Architecture

- **YOLO11s-seg** - person bounding box + segmentation mask (CoreML, ANE/GPU)
- **RTMPose RTMW-x-l** - 133 whole-body keypoints via SimCC (CoreML, GPU, float32)
- Mask contour extracted at proto resolution (160x90) - no upsampling
- 1-frame pipelined frame reading (ffmpeg pipe simulates camera CVPixelBuffer input)
- Mask + keypoints always from the same frame (no cross-frame drift)

## Key files

| File | Purpose |
|------|---------|
| `convert_rtmpose_coreml.py` | ONNX -> PyTorch -> CoreML conversion for RTMPose |
| `generate_baselines.py` | CPU reference outputs for validation |
| `validate_yolo_coreml.py` | YOLO CoreML accuracy vs PyTorch |
| `validate_rtmpose_coreml.py` | RTMPose CoreML accuracy vs ONNX |
| `validate_pipeline.py` | End-to-end pipeline validation |
| `profile_pipeline.py` | Per-stage timing breakdown |
| `swift-pipeline/` | Optimized Swift pipeline with async CoreML |

## Notes

- RTMPose requires **float32** precision. fp16 causes ~5% of keypoints to have catastrophic errors (argmax instability in SimCC logits).
- There is a ~10ms hardware-level penalty when alternating between two CoreML models, regardless of compute unit assignment or sync/async API. Standalone inference sum is ~25ms but alternating costs ~35ms.
