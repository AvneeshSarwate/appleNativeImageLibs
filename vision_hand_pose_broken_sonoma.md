---
name: VNDetectHumanHandPoseRequest broken on macOS 14.8.4
description: Apple Vision hand-pose API is flaky/unusable on user's Sonoma 14.8.4, affects both swift-pipeline and vision-standalone in appleNativeImageLibs
type: project
originSessionId: 5ded4992-fd47-479b-84ab-b108651a6fe3
---
`VNDetectHumanHandPoseRequest` is flaky on the user's M1 Max + macOS 14.8.4 (Xcode 16.2, Swift 6.0.3). Two distinct failure modes, both traced to the same underlying Vision bug:

1. **Hard error** — `Error Domain=com.apple.Vision Code=9 "encountered an unexpected condition: *** -[__NSArrayM insertObject:atIndex:]: object cannot be nil"` with 0 observations. Reproduces in both `appleNativeImageLibs/swift-pipeline` and `appleNativeImageLibs/vision-standalone`.
2. **Partial-success** — `perform` returns 21 joints but a subset have garbage confidences in the 191-255 range (uninitialized uint8 values leaking into the Float confidence field). Positions of those joints are still valid.

Reboot dramatically reduces frequency of both modes but does NOT eliminate them — occasional `Code=9` errors still appear post-reboot. User recalls hand pose being reliable before a macOS upgrade.

Additional repeated observation from 2026-04-22:
- If the hand model is enabled while hands are already in frame, failure is reproducible: either bad confidence values or the Code=9 exception.
- If hand inference is already enabled while no hands are in frame, and hands are brought into frame later, detection works reliably.
- Synthetic prewarming, blank live-sized primers, a 15-frame masked reveal, and camera-to-internal-Syphon loopback did not reproduce the reliable "bring hands in" behavior.

Investigated 2026-04-20 through 2026-04-22. Ruled out:
- Build / deployment target (vision-standalone is `.v12`, swift-pipeline is `.v14` — both fail identically)
- Shared-vs-fresh `VNDetectHumanHandPoseRequest` instance
- ANE/GPU vs CPU compute path (`usesCPUOnly = true` still errors)
- Camera device selection (both apps use default FaceTime cam)
- Joint name-lookup mismatch (stored keys match `HandSkeleton.connections` exactly)
- Simple camera pixel-buffer path (internal Syphon loopback still fails)
- Simple lazy-load prewarming (synthetic startup request and hand-toggle primer did not fix activation-with-hands-visible)

**Why:** internal Vision framework bug — Vision's own code catches an `NSInvalidArgumentException` from an array insert with nil and wraps it as NSError Code=9. When the partial-success path is hit instead, uninitialized memory ends up in the confidence field. Body/face/seg/contours in the same pipeline are unaffected; only hand pose.

**How to apply:**
- For VisionApp local development, run current source with:
  ```bash
  cd swift-pipeline
  swift run VisionApp
  ```
- Keep `Hands` enabled by default. Launch with hands out of frame, let Vision process a few no-hand frames, then bring hands in. Avoid toggling `Hands` on while hands are already visible unless reproducing the bug.
- When filtering hand joints by confidence, use a lower bound only (`conf > 0.3`) — do NOT add an upper bound like `conf <= 1.0`; that drops valid positions on partial-success frames. See `VisionOverlayView.swift:133,144`.
- If hand-pose reliability matters outside this default-on workflow, don't rely on toggling `VNDetectHumanHandPoseRequest` on while hands are visible. Use the DWPose-m CoreML model already present in `appleNativeImageLibs/` (133 whole-body keypoints incl. hands) as the fallback.
- If the user reports a macOS update and hand pose becoming reliable again, revisit this memory.
