# Vision Hand-Tracking Bugs

Status: intermittent / partially worked-around as of 2026-04-22.
Environment: macOS Sonoma 14.8.4 (23J319), Xcode 16.2 (SDK 15.2), Swift 6.0.3,
Apple M1 Max (10-core CPU / 32 GB).

Affects both apps in this directory (`swift-pipeline` and `vision-standalone`)
identically — it's an OS/Vision-framework-level issue, not anything in our code.

---

## 1. Hard failure — `perform` throws Code=9

**Symptom.** `VNDetectHumanHandPoseRequest.perform()` throws intermittently:

```
Error Domain=com.apple.Vision Code=9
"encountered an unexpected condition: *** -[__NSArrayM insertObject:atIndex:]:
 object cannot be nil"
```

When it throws, `request.results` stays nil/empty — zero hand observations for
that frame. Before a reboot this was firing **every** frame, effectively making
hand pose unusable.

**Root area.** Internally, Vision's hand-pose postprocessing is trying to
append something to an `NSMutableArray` that came back nil. Wrapped as an
`NSError` with domain `com.apple.Vision`, code 9. This is a bug inside Vision
itself — our code only catches it.

**Body / face / segmentation / contour requests are unaffected** in the same
pipeline. Only hand pose fails. Each Vision request type has its own weights
and postprocess path, and only the hand-pose path is broken.

## 2. Partial-success failure — garbage confidences

Even when `perform` succeeds and returns 21 joints, a subset of those joints
come back with confidence values in the `191`–`255` range — not 0–1 floats.
Example dump on a clearly-posed hand:

```
VNHLKITIP  conf=255.000   ← garbage
VNHLKMPIP  conf=255.000   ← garbage
VNHLKRDIP  conf=254.000   ← garbage
...
VNHLKTTIP  conf=0.968     ← normal
VNHLKTIP   conf=0.966     ← normal
VNHLKWRI   conf=0.948     ← normal
VNHLKPTIP  conf=0.000     ← actual zero (rejected)
```

Values in the 191–255 range are consistent with uninitialized uint8 memory
being reinterpreted as `Float32` (`Float(someUInt8)` → 0–255). The **positions**
of those joints remain valid — only the confidence field is corrupt.

This is almost certainly the same underlying bug as #1: in the "throw" case
something downstream catches the nil and raises Code=9; in the "partial" case
something else leaks garbage into the output struct. Two symptoms, one fault.

## 3. Content-dependent activation behavior

Manual testing on 2026-04-22 reproduced this consistently across about 10 runs
per state:

- If `Hands` is enabled while hands are already visible, Vision often returns
  bad confidence values or throws the Code=9 exception.
- If VisionApp starts with hand inference already enabled while no hands are in
  frame, and hands are brought into frame later, detection works reliably.

This still uses a fresh `VNDetectHumanHandPoseRequest` and fresh
`VNImageRequestHandler` per frame in the non-batch path, so the likely state is
inside Vision's process-global model/runtime/postprocessing machinery rather
than in a reused request object in our code. The failure appears tied to the
first positive hand-candidate/result-construction path after activation.

---

## What we ruled out during investigation

- **Shared vs. fresh request instance.** Switched from a single `handPoseRequest`
  reused per frame to a fresh `VNDetectHumanHandPoseRequest()` every frame —
  identical failure.
- **Compute device.** `freshHandRequest.usesCPUOnly = true` still throws
  Code=9 (plus ~3× slower), so it isn't the ANE/GPU kernel path. The fault is
  in framework-level code that runs regardless of compute device.
- **Deployment target.** `swift-pipeline`'s `Package.swift` targets
  `.macOS(.v14)`; `vision-standalone` targets `.macOS(.v12)`. Both fail
  identically on the same machine, so the `.v14` vs `.v12` binding isn't
  routing us to different internal model variants.
- **Camera.** Both apps default to the same built-in FaceTime camera.
  Switching doesn't change anything; Continuity Camera warnings in the log
  (`AVCaptureDeviceTypeExternal is deprecated...`) are unrelated.
- **Joint name-string mismatch.** Dumped `hand.joints` keys vs.
  `HandSkeleton.connections` raw values — identical sets. Rendering bugs are
  not a name-lookup issue.
- **Stale CoreML compiled-model cache.** The `com.apple.e5rt.e5bundlecache`
  under `~/Library/Containers/com.apple.mediaanalysisd/...` holds compiled
  models for the `mediaanalysisd` daemon — not for our (unsandboxed) app.
  Our app's Vision models aren't persisted to any user-writable location
  we could locate (not under `~/Library/Caches`, `~/Library/Containers`,
  `/tmp`, or `/var/folders`). Vision must be recompiling into memory on
  each launch, and the "reboot helps" effect comes from resetting the ANE
  daemon's in-memory state, not from clearing an on-disk cache.
- **Synthetic prewarming / priming.** Running a startup synthetic BGRA
  hand-pose request, running blank live-sized buffers on the hand-toggle edge,
  and feeding a 15-frame masked "window opening" sequence did not reproduce the
  reliable behavior of physically bringing hands into the camera frame.
- **Camera vs internal Syphon loopback.** Routing the camera frame through an
  internal Syphon send/receive path did not eliminate the bug, which makes a
  simple camera pixel-buffer-format issue less likely.

## What actually helps

- **Keep hand inference on from app startup.** `VisionApp` now defaults
  `Hands` to on. The reliable manual path is: launch with hands out of frame,
  let the app process a few no-hand frames, then bring hands into frame. Avoid
  turning `Hands` off and back on while hands are visible unless reproducing
  the bug.
- **Reboot.** Dramatically reduces occurrence of both symptoms — but does
  not eliminate them. Post-reboot the `Code=9` errors still fire
  intermittently, and garbage-confidence frames still appear.

## What *hasn't* been tried

- **Upgrading to macOS Sonoma 14.8.5.** Released 2026-03-25. Its security
  notes list exactly one Vision framework entry: **CVE-2026-20657 — "Parsing
  a maliciously crafted file may lead to an unexpected app termination." Fix:
  improved memory handling.** Same framework and same category (a Vision
  memory-handling bug), but framed as a file-parsing CVE, not a hand-pose
  issue. Might or might not cover our case.
- **Filing a Feedback Assistant report.** Worth doing — web searches found
  zero public reports of this exact error pattern on hand pose in Sonoma
  14.8.x, so a Radar is the only way to get an authoritative answer.

---

## Per-frame `VNImageRequestHandler` vs `VNSequenceRequestHandler`

Flagged in an earlier investigation and worth noting here:

- `segRequest` uses a long-lived `VNSequenceRequestHandler`.
- `bodyPoseRequest` uses a long-lived `VNSequenceRequestHandler`.
- `faceLandmarksRequest` uses a long-lived `VNSequenceRequestHandler`.
- **Hand pose uses a fresh `VNImageRequestHandler` every frame.**

See `swift-pipeline/Sources/VisionApp/VisionPipelineEngine.swift` — in the
non-batch branch the hand path is:

```swift
let handHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
try handHandler.perform([freshHandRequest])
```

This inconsistency existed before the Code=9 bug surfaced and wasn't
introduced to work around it. `VNSequenceRequestHandler` preserves tracking
state across frames (useful for tracking requests); hand pose is stateless
per-frame, so there's no correctness reason to require one. Still, it's the
one code-shape difference in the file and worth experimenting with if we
need another lever:

- Try running the fresh hand request through a persistent
  `VNSequenceRequestHandler` to see if it changes Code=9 frequency or the
  garbage-confidence rate. No specific reason to expect a fix (the fault is
  inside postprocessing, not in handler state), but it's a cheap test and
  brings the code into uniformity with the other Vision requests.
- If future investigation reaches for "what haven't we tried," this is on
  the list.

---

## Workarounds currently in tree

1. **Hand inference starts enabled.**
   `VisionPipelineEngine` initializes both the UI toggle and the video-queue
   `PipelineConfig` hand flag to `true`. This avoids the high-risk path where
   `Hands` is toggled on while hands are already visible. For local testing,
   launch current source with:

   ```bash
   cd swift-pipeline
   swift run VisionApp
   ```

   Keep hands out of frame during launch, then bring them in.

2. **Rendering filter relaxed to ignore the garbage upper bound.**
   `swift-pipeline/Sources/VisionApp/VisionOverlayView.swift` draws joints
   with `conf > 0.3` only — the previous `conf <= 1.0` upper bound (added
   to suppress the 191-255 junk) was rejecting joints whose positions were
   still valid, producing the "only thumb + wrist show up" symptom. Comment
   in that file flags this intentional choice.

3. **WebSocket hand-bbox streaming filters explicitly.**
   `swift-pipeline/Sources/VisionApp/ContourUDPSender.swift` `sendHand` uses
   `conf > 0 && conf <= 1.0` when computing the bounding box — excludes
   both Vision's explicit rejections (0) and the garbage uint8s (> 1.0).
   Bbox is stable across partial-success frames because the garbage-conf
   joints (whose positions are valid) are left in the pose for rendering
   but not used in the bbox computation.

4. **Per-frame Code=9 errors logged, not swallowed.**
   The hand path in `VisionPipelineEngine.swift` wraps `perform` in a
   `do/catch` and prints `[hand] perform error: ...` so we can tell a dead
   frame from "no hand in view." Keep this diagnostic until the upstream
   bug is understood or gone.

## Testing checklist before removing any workaround

- [ ] Upgrade macOS to latest Sonoma point release (14.8.5+) and re-run
      the standalone VisionApp with hands enabled. If Code=9 stops, the
      `conf <= 1.0` upper bound in the overlay filter can come back.
- [ ] Do a dev session where hand pose is the only Vision request enabled
      (seg/body/face off). Confirm whether partial-failure still happens
      in isolation — rules out interaction between the request types as
      a cause.
- [ ] Try running hand pose via a persistent `VNSequenceRequestHandler`
      just to check (see section above).
- [ ] If the default-on workaround starts failing, retest whether the crucial
      variable is physical hand motion into frame rather than simply "no-hand
      frames before first positive detection."
- [ ] File an Apple Feedback Assistant radar with a minimal reproducer
      and the current confidence dump. Link the FB number here.
