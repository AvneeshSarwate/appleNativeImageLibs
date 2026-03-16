# VisionApp

Real-time camera app using Apple's native Vision framework for person segmentation, body/hand pose, and face landmarks. Alternative to the CoreML pipeline (`CameraApp`) that uses YOLO11-seg + DWPose.

## Build & Run

```bash
cd swift-pipeline
swift build -c release
.build/release/VisionApp
```

No model files needed — Vision uses system-provided models.

## What It Does

Runs 4 Vision requests per camera frame and renders results as overlays:

| Request | Output | Visualization |
|---|---|---|
| `VNGeneratePersonSegmentationRequest` | Grayscale mask | Green semi-transparent overlay |
| `VNDetectHumanBodyPoseRequest` | 19 body joints | Colored dots + white skeleton bones |
| `VNDetectHumanHandPoseRequest` | 21 joints/hand (up to 2) | Cyan/mint dots + bones |
| `VNDetectFaceLandmarksRequest` | 76 face landmarks | Purple dots |

## UI Controls

- **Left panel**: Toggle each layer on/off, segmentation quality (Fast/Balanced/Accurate), batch mode
- **Right panel**: Rolling-average FPS, total frame time, per-request timing breakdown
- **Batch mode**: Runs all requests in a single `perform()` call (faster, but no per-request timing)

## Architecture

```
VisionApp.swift              — @main app shell, SwiftUI view with controls
VisionPipelineEngine.swift   — AVCaptureSession + Vision request orchestration
VisionResultTypes.swift      — FrameResult, BodyPoseData, HandPoseData, FaceLandmarkData
VisionOverlayView.swift      — Canvas-based rendering (mask, skeleton, hands, face)
SkeletonDefinitions.swift    — Joint connection pairs for body + hand bone drawing
```

### Request Handler Strategy

Each request type uses a specific handler for correctness:

| Request | Handler | Why |
|---|---|---|
| Segmentation | `VNSequenceRequestHandler` | Temporal smoothing improves mask stability across frames |
| Body pose | `VNSequenceRequestHandler` | Reduces inter-frame jitter |
| **Hand pose** | **`VNImageRequestHandler`** | `VNSequenceRequestHandler` produces invalid confidence values (>1.0, up to 255) on macOS 14 / M1 Max — a system-level quirk not documented by Apple |
| Face landmarks | `VNSequenceRequestHandler` | Temporal smoothing works correctly |

The overlay filters joints to confidence range [0.3, 1.0] as a safety net.

### Thread Model

- Camera frames arrive on a dedicated `DispatchQueue` (`.userInitiated` QoS for GPU access)
- Vision `perform()` calls are synchronous on that queue
- Results dispatched to `@MainActor` via `Task` for SwiftUI updates
- UI toggle state synced to the video queue via a lock-protected `PipelineConfig` object

## Hardware Profile (M1 Max, Xcode Instruments)

Profiled with all 4 requests enabled, segmentation quality `.balanced`, 10 seconds steady-state:

| Metric | Value |
|---|---|
| FPS | ~15 (all on), ~30 (seg + body only) |
| GPU total | ~1.4 ms/frame (~3% of M1 Max budget) |
| ANE | Primary compute — continuous activity |
| CPU | Moderate (coordination, result extraction) |

**Vision is mostly ANE + CPU with minimal GPU usage.** Good for coexistence with GPU-heavy apps like TouchDesigner.

## Comparison to CoreML Pipeline (CameraApp)

| | VisionApp | CameraApp (CoreML) |
|---|---|---|
| FPS | ~15-30 | ~60 |
| GPU/frame | ~1.4 ms | ~7 ms (DWPose on GPU) |
| Foot keypoints | None (ankles only) | 6 (big toe, small toe, heel x2) |
| Model files | None (system) | YOLO + DWPose .mlpackage |
| Compute unit control | None | Full (`MLComputeUnits`) |
| Code complexity | ~500 lines | ~1500 lines + Shared |
| Preprocessing | None (Vision handles it) | Letterbox, crop, normalize |

Use VisionApp when: simpler setup, no foot keypoints needed, GPU headroom is priority.
Use CameraApp when: higher FPS needed, foot keypoints required, explicit hardware control matters.
