# VisionApp

Real-time camera app using Apple's native Vision framework for person segmentation, body/hand pose, and face landmarks. Alternative to the CoreML pipeline (`CameraApp`) that uses YOLO11-seg + DWPose.

## Build & Run

For local development, run the SwiftPM product directly:

```bash
cd swift-pipeline
swift run VisionApp
```

For an optimized local build:

```bash
cd swift-pipeline
swift build -c release --product VisionApp
.build/release/VisionApp
```

No model files needed — Vision uses system-provided models.

Do not use `swift-pipeline/dist/VisionApp` for local development unless you
have explicitly rebuilt the distribution bundle. That binary is a packaged copy
and can be stale relative to `Sources/VisionApp/`.

## What It Does

Runs 4 Vision requests per camera frame and renders results as overlays:

| Request | Output | Visualization |
|---|---|---|
| `VNGeneratePersonSegmentationRequest` | Grayscale mask | Green semi-transparent overlay |
| `VNDetectHumanBodyPoseRequest` | 19 body joints | Colored dots + white skeleton bones |
| `VNDetectHumanHandPoseRequest` | 21 joints/hand (up to 2) | Cyan/mint dots + bones |
| `VNDetectFaceLandmarksRequest` | 76 face landmarks | Purple dots |

## UI Controls

- **Input source**: `Camera` or internal `Camera via Syphon` loopback
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

The hand overlay intentionally uses only a lower confidence bound (`conf > 0.3`).
On the affected Sonoma 14.8.4 setup, Vision sometimes returns valid hand joint
positions with garbage confidence values above 1.0. Adding an upper bound hides
otherwise usable joints. WebSocket hand bounding-box streaming still filters
`conf > 0 && conf <= 1.0` to avoid garbage-confidence joints in bbox math.

### Hand Pose Startup Workaround

`Hands` defaults to on. This is intentional.

On the affected Sonoma 14.8.4 / M1 Max setup, toggling hand pose on while hands
are already visible can reliably trigger either `com.apple.Vision Code=9`
(`-[__NSArrayM insertObject:atIndex:]: object cannot be nil`) or bad confidence
values. Starting VisionApp with hand inference already enabled, letting it see a
few no-hand frames, and then bringing hands into frame has been reliable across
repeated manual runs.

For the most stable hand tracking:

```bash
cd swift-pipeline
swift run VisionApp
```

Launch with hands out of frame, then bring them in. Avoid turning `Hands` off and
back on while hands are visible unless you are trying to reproduce the bug.

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
