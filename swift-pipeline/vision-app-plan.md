# Vision Framework App -- Implementation Plan

A new executable target ("VisionApp") in the existing Swift package that replaces
the custom CoreML pipeline (YOLO11-seg + DWPose/RTMPose) with Apple's built-in
Vision framework requests. Same core visualization: person mask, body skeleton,
hand keypoints, face landmarks.

---

## 1. Vision API Mapping

### 1A. Person Segmentation

| Existing | Vision replacement |
|---|---|
| YOLO11-seg instance mask (32-coeff proto decode, ~160x160 output) | `VNGeneratePersonSegmentationRequest` |

**API details:**
- Available since macOS 12 / iOS 15.
- `qualityLevel` property controls the accuracy/speed tradeoff:
  - `.fast` -- designed for real-time video; mask resolution is low (around 64x64
    to 128x128 depending on input). Runs almost entirely on the ANE.
  - `.balanced` -- still usable at video rates on M1; output around 256x256 to
    384x512. Good middle ground.
  - `.accurate` -- high-quality mask, roughly matching input resolution. Too slow
    for live video on most devices (hundreds of ms per frame).
- Output format: a `CVPixelBuffer` in `kCVPixelFormatType_OneComponent8` (single
  channel, 0-255 grayscale matte). Access via
  `observation.pixelBuffer` on the `VNPixelBufferObservation` result.
- The request is **stateful** -- Apple recommends reusing the same request
  instance across frames for temporal consistency (it uses past frames to smooth
  the mask).
- Unlike the YOLO pipeline there is no contour or bounding box output. Those
  would have to be derived from the mask if needed (threshold + march-squares or
  `findContours` from CoreImage/Accelerate).

**Recommendation:** Start with `.balanced` for the live-camera path. It gives a
reasonable mask at video frame rates on M1 Max. The user can toggle to `.fast` if
GPU headroom is needed for TouchDesigner.

### 1B. Body Keypoints

| Existing | Vision replacement |
|---|---|
| DWPose/RTMPose 133-keypoint whole-body (17 body + 6 feet + 68 face + 42 hands) | `VNDetectHumanBodyPoseRequest` (19 body joints only) |

**The 19 body joints** (VNHumanBodyPoseObservation.JointName):

| Group | Joints |
|---|---|
| Head | nose, leftEye, rightEye, leftEar, rightEar |
| Torso | neck, leftShoulder, rightShoulder, root (center hip), leftHip, rightHip |
| Arms | leftElbow, leftWrist, rightElbow, rightWrist |
| Legs | leftKnee, leftAnkle, rightKnee, rightAnkle |

Joint groups for batch access (VNHumanBodyPoseObservation.JointsGroupName):
`face`, `torso`, `leftArm`, `rightArm`, `leftLeg`, `rightLeg`, `all`.

**Gap:** DWPose provides 133 points. Vision body pose gives only 19. The 114
missing points are covered (partially) by the separate hand and face requests
described below. There is no foot-keypoint equivalent in Vision at all (DWPose
gives 6 foot points: left/right big-toe, small-toe, heel). This is an
unavoidable loss with Vision.

**Coordinate system:** Points are `VNRecognizedPoint` with `.location` in
**Vision normalized coordinates** (0..1, origin at bottom-left). To convert to
pixel coordinates for a frame of size (W, H):
```swift
let pixelX = point.location.x * CGFloat(imageWidth)
let pixelY = (1.0 - point.location.y) * CGFloat(imageHeight)  // flip Y
```
Each point also carries a `confidence` value (0..1).

The request supports detecting multiple people simultaneously (returns an array
of `VNHumanBodyPoseObservation`).

Available since macOS 11 / iOS 14.

### 1C. Hand Keypoints

| Existing | Vision replacement |
|---|---|
| DWPose hand subset (21 points per hand embedded in the 133-point output) | `VNDetectHumanHandPoseRequest` (21 joints per hand, up to 4 hands) |

**The 21 hand joints** (VNHumanHandPoseObservation.JointName):

| Group | Joints |
|---|---|
| Wrist | wrist |
| Thumb (4) | thumbCMC, thumbMP, thumbIP, thumbTip |
| Index (4) | indexMCP, indexPIP, indexDIP, indexTip |
| Middle (4) | middleMCP, middlePIP, middleDIP, middleTip |
| Ring (4) | ringMCP, ringPIP, ringDIP, ringTip |
| Little (4) | littleMCP, littlePIP, littleDIP, littleTip |

Joint groups: `thumb`, `indexFinger`, `middleFinger`, `ringFinger`,
`littleFinger`, `all`.

Set `maximumHandCount` to control how many hands to detect (default 2, max 4).
Each detected hand is a separate `VNHumanHandPoseObservation`.

**Same coordinate system** as body pose (normalized, bottom-left origin).

**Note:** Unlike DWPose, Vision hand detection is a wholly separate request.
There is no automatic association between a detected hand and a detected body.
If you need to know "which body does this hand belong to," you must do spatial
matching yourself (e.g., check if the hand wrist point is near a body wrist
point).

Available since macOS 11 / iOS 14.

### 1D. Face Landmarks

| Existing | Vision replacement |
|---|---|
| DWPose face subset (68 points embedded in 133-point output) | `VNDetectFaceLandmarksRequest` (up to 76 points per face) |

This is a two-stage process in Vision:
1. `VNDetectFaceRectanglesRequest` (or the landmarks request handles face
   detection implicitly when needed).
2. `VNDetectFaceLandmarksRequest` -- returns `VNFaceObservation` with a
   `.landmarks` property of type `VNFaceLandmarks2D`.

**Face landmark regions** (VNFaceLandmarks2D properties):

| Region | Approximate point count (76-pt constellation) |
|---|---|
| faceContour | 17 |
| leftEye | 8 |
| rightEye | 8 |
| leftEyebrow | 7 |
| rightEyebrow | 7 |
| nose | 5 |
| noseCrest | 3 |
| medianLine | 3 |
| outerLips | 11 |
| innerLips | 5 |
| leftPupil | 1 |
| rightPupil | 1 |
| **allPoints** | **76** |

To get 76 points (vs the older 65-point constellation), set the request's
constellation property:
```swift
let faceLandmarksReq = VNDetectFaceLandmarksRequest()
faceLandmarksReq.constellation = .constellation76Points
```

**Coordinate system for face landmarks:** Points from
`VNFaceLandmarkRegion2D.normalizedPoints` are normalized relative to the
**face bounding box** (not the full image). To get image-normalized coordinates
you must transform through the face observation's `boundingBox`:
```swift
let faceBBox = faceObservation.boundingBox  // in image-normalized coords
let ptInImage = CGPoint(
    x: faceBBox.origin.x + landmark.x * faceBBox.width,
    y: faceBBox.origin.y + landmark.y * faceBBox.height
)
// Then convert from Vision coords (bottom-left origin) to pixel coords
let pixelX = ptInImage.x * CGFloat(imageWidth)
let pixelY = (1.0 - ptInImage.y) * CGFloat(imageHeight)
```

Available since macOS 10.13 / iOS 11 (76-point constellation from revision 3,
macOS 12+).

### 1E. Summary: Keypoint Count Comparison

| Component | DWPose/RTMPose | Vision framework |
|---|---|---|
| Body | 17 | 19 (adds neck, root; drops nothing) |
| Feet | 6 | 0 (no equivalent) |
| Face | 68 | 76 (comparable, different topology) |
| Hands (2) | 42 (21 x 2) | 42 (21 x 2, separate request) |
| **Total** | **133** | **137** (body+face+hands, but no feet and from 3 separate requests) |

The total raw keypoint count is actually slightly higher with Vision, but they
come from three independent detectors with no cross-association and no foot
points.

---

## 2. Architecture

### 2A. Package.swift Changes

Add a new executable target `VisionApp` alongside the existing `CameraApp`:

```swift
.executableTarget(
    name: "VisionApp",
    path: "Sources/VisionApp",
    swiftSettings: [.swiftLanguageMode(.v5)]
)
```

No dependency on `Shared` is required. The Vision-based app does not use any
CoreML model loading, YOLO parsing, SimCC decoding, letterboxing, or pose
normalization code from Shared. A clean separation avoids accidental coupling.

The VisionApp target will live under `Sources/VisionApp/`.

### 2B. File Structure

```
Sources/VisionApp/
    VisionApp.swift          -- @main App, WindowGroup, top-level view
    VisionPipelineEngine.swift -- camera capture + Vision request orchestration
    VisionResultTypes.swift  -- result structs (VisionFrameResult, etc.)
    VisionOverlayView.swift  -- Canvas-based visualization
    SkeletonDefinitions.swift -- joint connection pairs for drawing bones
```

### 2C. Running Multiple Vision Requests Efficiently

`VNImageRequestHandler` accepts an **array** of requests in a single `perform()`
call. The framework handles scheduling internally (it may parallelize across
ANE/GPU/CPU as it sees fit). This is the recommended approach:

```swift
let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
try handler.perform([
    segmentationRequest,
    bodyPoseRequest,
    handPoseRequest,
    faceLandmarksRequest
])
```

All four requests run on the same frame with one handler call. Results are
accessed from each request's `results` property after `perform()` returns.

**Important:** `perform()` is synchronous and blocking. It should be called on a
background queue (the video-capture dispatch queue is ideal). Results are then
dispatched to `@MainActor` for UI updates.

### 2D. Camera Capture

The camera setup is identical to the existing CameraApp:
- `AVCaptureSession` with `.hd1920x1080` preset
- `AVCaptureVideoDataOutput` with BGRA pixel format
- `alwaysDiscardsLateVideoFrames = true`
- Delegate on a dedicated `DispatchQueue`

This code is simple enough to duplicate rather than factor into Shared. It is
about 20 lines and avoids adding a framework dependency.

### 2E. Request Lifecycle

The four Vision requests should be **created once** and reused across frames.
This is especially important for `VNGeneratePersonSegmentationRequest` which is
stateful and uses temporal information. For the pose and face requests, reuse
avoids repeated internal setup costs.

```swift
// Created once at init
private let segRequest: VNGeneratePersonSegmentationRequest
private let bodyPoseRequest: VNDetectHumanBodyPoseRequest
private let handPoseRequest: VNDetectHumanHandPoseRequest
private let faceLandmarksRequest: VNDetectFaceLandmarksRequest

init() {
    segRequest = VNGeneratePersonSegmentationRequest()
    segRequest.qualityLevel = .balanced
    segRequest.outputPixelFormat = kCVPixelFormatType_OneComponent8

    bodyPoseRequest = VNDetectHumanBodyPoseRequest()

    handPoseRequest = VNDetectHumanHandPoseRequest()
    handPoseRequest.maximumHandCount = 2

    faceLandmarksRequest = VNDetectFaceLandmarksRequest()
    faceLandmarksRequest.constellation = .constellation76Points
}
```

---

## 3. Performance Considerations

### 3A. Segmentation Quality Tradeoffs

| Quality | Approx mask size | Approx latency (M1 Max) | Suitable for |
|---|---|---|---|
| `.fast` | ~64x64 to 128x128 | <5 ms | Real-time video, minimal GPU use |
| `.balanced` | ~256x256 to 384x512 | ~8-15 ms | Real-time video, good quality |
| `.accurate` | near input resolution | ~50-200 ms | Still images only |

For TouchDesigner coexistence, `.fast` may be the right default since it leaves
more GPU/ANE headroom. The mask is lower resolution but still usable as an
overlay at 960x540 display size.

### 3B. Parallelism Within Vision

When you pass multiple requests to a single `perform()` call, Vision can
internally parallelize across compute units. Apple's implementation distributes
work across ANE, GPU, and CPU as it sees fit. You do not control which unit
each request runs on (there is no equivalent of `MLComputeUnits` for Vision
requests).

This is both a strength (zero configuration) and a limitation (no ability to
pin segmentation to ANE and leave GPU free for TouchDesigner).

### 3C. Expected Performance vs Custom CoreML Pipeline

The existing pipeline achieves ~30 FPS with YOLO+DWPose in parallel mode on
M1 Max. Expected Vision performance:

- **Segmentation** (`.balanced`): ~10-15 ms per frame
- **Body pose**: ~3-5 ms per frame
- **Hand pose** (2 hands): ~5-8 ms per frame
- **Face landmarks**: ~3-5 ms per frame
- **Total (sequential within perform)**: ~20-35 ms per frame, i.e., 28-50 FPS

The Vision pipeline should achieve comparable or slightly better throughput than
the CoreML pipeline because:
1. No letterboxing / manual preprocessing required (Vision handles it internally)
2. No SimCC decode / manual postprocessing
3. Apple can fuse and optimize the internal models
4. The segmentation model is lighter than YOLO11-seg for the mask-only use case

However, the total CPU time in the `captureOutput` delegate is the bottleneck.
If Vision's `perform()` blocks for 30+ ms, frames will be dropped (which is fine
since `alwaysDiscardsLateVideoFrames = true`).

### 3D. GPU/ANE Considerations for TouchDesigner

Vision framework requests run on whatever compute units Apple chooses internally.
On M1 Max:
- Segmentation (`.fast` and `.balanced`) primarily uses the ANE
- Body/hand pose detection uses a mix of ANE and GPU
- Face landmarks uses mostly CPU/ANE

There is **no way to force Vision to avoid the GPU entirely**, unlike the CoreML
pipeline where you can set `computeUnits = .cpuAndNeuralEngine`. If GPU
contention with TouchDesigner is a concern:
1. Use `.fast` segmentation quality (more ANE, less GPU)
2. Reduce the camera resolution to 1280x720 (less work overall)
3. Consider dropping face landmarks if not needed (saves ~3-5 ms and some GPU)
4. Profile with Instruments to measure actual GPU overlap

### 3E. Frame Dropping Strategy

Same as existing pipeline: set `alwaysDiscardsLateVideoFrames = true` on the
video output. If Vision processing takes longer than the inter-frame interval
(33 ms at 30 FPS), late frames are silently dropped. No explicit frame skipping
logic is needed.

---

## 4. Visualization Mapping

### 4A. Segmentation Mask

The existing pipeline produces a `MaskData` struct with float values and
width/height. Vision produces a `CVPixelBuffer` in OneComponent8 format.

To render the mask as a green semi-transparent overlay (matching existing style):

```swift
func maskToImage(_ pixelBuffer: CVPixelBuffer) -> CGImage? {
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    let w = CVPixelBufferGetWidth(pixelBuffer)
    let h = CVPixelBufferGetHeight(pixelBuffer)
    let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let base = CVPixelBufferGetBaseAddress(pixelBuffer)!
        .assumingMemoryBound(to: UInt8.self)

    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    for y in 0..<h {
        for x in 0..<w {
            let alpha = base[y * rowBytes + x]
            let scaled = UInt8(min(255, Int(alpha) * 160 / 255))
            let i = (y * w + x) * 4
            rgba[i]     = 0        // R
            rgba[i + 1] = scaled   // G (premultiplied)
            rgba[i + 2] = 0        // B
            rgba[i + 3] = scaled   // A
        }
    }
    // ... create CGImage from rgba (same as existing maskToCGImage)
}
```

Because the mask pixel buffer comes directly from Vision, no letterbox-unpadding
is needed. The mask covers the full image extent and can be drawn directly into
the view (scaled to fit).

### 4B. Body Skeleton

The existing pipeline draws colored dots for keypoints but does not draw
connecting bones. The Vision app should draw both joints and bones for a
proper skeleton visualization.

**Body skeleton connections** (19 joints, ~17 bones):

```swift
static let bodyConnections: [(JointName, JointName)] = [
    (.leftEar, .leftEye),   (.rightEar, .rightEye),
    (.leftEye, .nose),      (.rightEye, .nose),
    (.nose, .neck),
    (.neck, .leftShoulder), (.neck, .rightShoulder),
    (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
    (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
    (.neck, .root),
    (.root, .leftHip),      (.root, .rightHip),
    (.leftHip, .leftKnee),  (.leftKnee, .leftAnkle),
    (.rightHip, .rightKnee),(.rightKnee, .rightAnkle),
]
```

Draw each bone as a line segment when both endpoints have confidence > 0.3.
Draw each joint as a filled circle (radius ~4pt). Color by group: head = red,
arms = orange, torso = yellow, legs = blue (matching existing color scheme
roughly).

### 4C. Hand Keypoints

**Hand skeleton connections** (21 joints, 20 bones per hand):

```swift
static let handConnections: [(JointName, JointName)] = [
    // Thumb
    (.wrist, .thumbCMC), (.thumbCMC, .thumbMP),
    (.thumbMP, .thumbIP), (.thumbIP, .thumbTip),
    // Index
    (.wrist, .indexMCP), (.indexMCP, .indexPIP),
    (.indexPIP, .indexDIP), (.indexDIP, .indexTip),
    // Middle
    (.wrist, .middleMCP), (.middleMCP, .middlePIP),
    (.middlePIP, .middleDIP), (.middleDIP, .middleTip),
    // Ring
    (.wrist, .ringMCP), (.ringMCP, .ringPIP),
    (.ringPIP, .ringDIP), (.ringDIP, .ringTip),
    // Little
    (.wrist, .littleMCP), (.littleMCP, .littlePIP),
    (.littlePIP, .littleDIP), (.littleDIP, .littleTip),
]
```

Draw hand joints smaller (radius ~2pt) in cyan/blue (matching existing color
for hand keypoints). Draw bones as thin lines.

### 4D. Face Landmarks

Face landmarks are returned as arrays of `CGPoint` per region. Draw them as
small dots (radius ~1.5pt) in a distinct color (e.g., magenta or light purple).

Optionally connect them by region (e.g., draw the faceContour as a polyline,
each eye as a closed polygon, lips as closed polygons). This is more complex
but gives a nicer visualization.

For a first pass, just draw the 76 points as dots.

### 4E. Coordinate System Conversion

All Vision points use normalized coordinates with **origin at bottom-left**.
The existing pipeline uses pixel coordinates with origin at top-left (standard
image convention). The conversion function:

```swift
/// Convert Vision normalized point to view coordinates
func visionPointToView(_ point: VNRecognizedPoint,
                       imageSize: CGSize,
                       viewSize: CGSize,
                       mirrored: Bool) -> CGPoint {
    let scaleX = viewSize.width / imageSize.width
    let scaleY = viewSize.height / imageSize.height

    // Vision: bottom-left origin, normalized 0..1
    let pixelX = point.location.x * imageSize.width
    let pixelY = (1.0 - point.location.y) * imageSize.height  // flip Y

    var viewX = pixelX * scaleX
    let viewY = pixelY * scaleY

    if mirrored {
        viewX = viewSize.width - viewX
    }
    return CGPoint(x: viewX, y: viewY)
}
```

For face landmarks specifically, an extra step transforms from face-bbox-relative
coordinates to image-normalized coordinates first (see section 1D above).

---

## 5. Implementation Steps

### Step 1: Package.swift

Add the `VisionApp` executable target. No external dependencies.

```swift
.executableTarget(
    name: "VisionApp",
    path: "Sources/VisionApp",
    swiftSettings: [.swiftLanguageMode(.v5)]
)
```

Create directory: `Sources/VisionApp/`

### Step 2: Result Types (VisionResultTypes.swift)

Define the data structures that flow from the processing pipeline to the UI:

```swift
struct VisionFrameResult {
    let maskPixelBuffer: CVPixelBuffer?       // from segmentation
    let bodyPoses: [BodyPoseData]             // from body pose detection
    let handPoses: [HandPoseData]             // from hand pose detection
    let faceLandmarks: [FaceLandmarkData]     // from face landmarks
    let frameTimeMs: Double
}

struct BodyPoseData {
    let joints: [String: (CGPoint, Float)]    // jointName -> (pixel coords, confidence)
}

struct HandPoseData {
    let joints: [String: (CGPoint, Float)]
    let chirality: Chirality                   // .left or .right (if available)
}

struct FaceLandmarkData {
    let allPoints: [(CGPoint, Float)]         // pixel coords + per-point confidence
    let boundingBox: CGRect                    // face bbox in pixel coords
}

enum Chirality { case left, right, unknown }
```

Using pixel coordinates (top-left origin) in the result types keeps the
visualization code simple and matches the existing pipeline's convention. The
coordinate conversion from Vision's normalized bottom-left system happens once
during result extraction, not in the drawing code.

### Step 3: Camera Capture + Vision Pipeline (VisionPipelineEngine.swift)

Core class, roughly 150-200 lines. Key design decisions:

- `@MainActor class VisionPipelineEngine: NSObject, ObservableObject`
- Camera setup: duplicated from CameraApp (simple, avoids Shared dependency)
- Vision requests: created once, reused across frames
- Processing: synchronous `perform()` on the video capture queue
- Result extraction: convert Vision observations to `VisionFrameResult` with
  pixel coordinates
- UI update: dispatch to `@MainActor` via `Task`

Processing flow per frame:

```
captureOutput delegate (video queue)
    |
    v
VNImageRequestHandler(cvPixelBuffer:)
    |
    v
perform([segReq, bodyReq, handReq, faceReq])  // single call, ~20-30ms
    |
    v
Extract results from each request
    |-- segReq.results?.first?.pixelBuffer
    |-- bodyReq.results  -> map to BodyPoseData[]
    |-- handReq.results  -> map to HandPoseData[]
    |-- faceReq.results  -> map to FaceLandmarkData[]
    |
    v
Construct VisionFrameResult
    |
    v
Task { @MainActor in self.latestResult = result }
```

**Key difference from existing pipeline:** No model loading, no letterboxing,
no crop-to-bbox, no SimCC decode, no proto-mask decode. Vision handles all
preprocessing and postprocessing internally. The code is dramatically simpler.

**Thread safety note:** `VNImageRequestHandler.perform()` is synchronous and
runs on the calling thread (the video capture DispatchQueue). The request
objects themselves should only be accessed from this same queue. Results are
read immediately after `perform()` returns, before any other frame can arrive
(since `alwaysDiscardsLateVideoFrames = true` means only one frame is in
flight).

### Step 4: Skeleton Definitions (SkeletonDefinitions.swift)

Static data defining bone connections for body and hand skeletons. Pure data,
no logic. Also defines color mappings per joint group. Roughly 60-80 lines.

### Step 5: Visualization Overlay (VisionOverlayView.swift)

A SwiftUI `Canvas`-based view similar to the existing `OverlayView`. Draws:

1. **Mask layer** (bottom): Convert `CVPixelBuffer` to `CGImage`, draw
   full-frame with green tint and alpha.
2. **Body skeleton**: Draw bones as lines, joints as circles. Color by group.
3. **Hand skeleton**: Draw bones as thinner lines, joints as smaller circles.
   Color: cyan.
4. **Face landmarks**: Draw 76 dots per face. Color: magenta. Optionally
   connect by region.
5. **Info overlay** (top-right): FPS counter, frame time, keypoint counts.

Coordinate conversion uses the `mirrored` flag for front-camera mirror (same
as existing pipeline). The `mapX` helper pattern from the existing OverlayView
can be reused.

Approximate size: 150-200 lines.

### Step 6: Main App Shell (VisionApp.swift)

Minimal SwiftUI app:

```swift
@main
struct VisionTestApp: App {
    var body: some Scene {
        WindowGroup {
            VisionCameraView()
                .frame(width: 960, height: 540)
        }
    }
}

struct VisionCameraView: View {
    @StateObject private var engine = VisionPipelineEngine()

    var body: some View {
        ZStack {
            Color.black.edgesIgnoringSafeArea(.all)

            VisionOverlayView(
                result: engine.latestResult,
                viewSize: CGSize(width: 960, height: 540),
                imageSize: engine.imageSize,
                mirrored: true
            )
            .frame(width: 960, height: 540)

            // FPS overlay (top-right)
            ...
        }
        .task {
            do {
                try engine.start()
            } catch {
                print("Failed to start: \(error)")
            }
        }
    }
}
```

No model directory argument needed (unlike CameraApp which needs
`modelDir` to find .mlpackage files). Vision uses system-provided models.

---

## 6. Things the Existing Pipeline Does That Vision Cannot

| Feature | Status with Vision |
|---|---|
| Foot keypoints (6 pts) | Not available. No Vision API for feet. |
| Contour polygon from mask | Must be derived manually from the mask buffer. Not provided by Vision. |
| Bounding box from YOLO | Body pose gives implicit bbox via joint extremes. Segmentation does not give a bbox. Face detection gives a face bbox. |
| Single-person detection (best confidence) | Vision returns all detected people/hands/faces. You filter to the most prominent or largest if you want single-person mode. |
| Explicit compute unit control | Not possible with Vision requests. Cannot pin to ANE-only or CPU-only. |
| 1-frame-lag parallel mode | Not needed. Vision internally parallelizes within `perform()`. No manual async pipelining required. |

---

## 7. Build and Run

```bash
cd /Users/avneeshsarwate/agentCombine/appleNativeImageLibs/swift-pipeline
swift run VisionApp
```

Or open in Xcode:
```bash
open Package.swift
# Select VisionApp scheme, run
```

Camera permission: the app needs `com.apple.security.device.camera` entitlement.
For a command-line Swift package executable, the system will prompt on first run.
If it does not, an `Info.plist` with `NSCameraUsageDescription` may need to be
embedded (same situation as the existing CameraApp).

---

## 8. Future Enhancements

1. **Contour extraction from mask:** Run a simple marching-squares or
   `CIFilter`-based contour on the segmentation mask to replicate the
   existing contour overlay.

2. **Body-hand association:** Match detected hands to bodies by comparing
   wrist positions. This lets you draw a complete "person" with body + hands
   as a unified entity.

3. **3D body pose:** `VNDetectHumanBodyPose3DRequest` (macOS 14+) provides
   17 joints in 3D world coordinates. Could be interesting for depth-aware
   visualization or sending 3D pose data to TouchDesigner.

4. **Person instance segmentation:** `VNGeneratePersonInstanceMaskRequest`
   (macOS 15+ / WWDC 2023) segments up to 4 individual people with separate
   masks. Useful if multiple people are in frame.

5. **Configurable quality preset:** Add a UI toggle or config similar to the
   existing `PipelineConfig` that lets the user switch segmentation quality
   and enable/disable hand/face detection at runtime.
