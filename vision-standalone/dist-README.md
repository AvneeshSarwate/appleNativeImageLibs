# VisionApp

Real-time camera → Vision framework → Syphon output.

## First run

macOS quarantines unsigned binaries. Clear it once:

```bash
xattr -cr /path/to/this/dist/folder
```

Then:

```bash
cd /path/to/this/dist/folder
./VisionApp
```

Grant camera access when prompted.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 12 (Monterey) or later
- Camera

## Syphon

Toggle "Syphon out" in the app. In TouchDesigner, add a **Syphon Spout In TOP** and select "VisionApp" as the sender. The output is the camera frame masked by person segmentation with adjustable threshold.

## Build

There are two build projects due to Swift runtime compatibility:

| Project | Path | macOS target | Use case |
|---|---|---|---|
| Main | `swift-pipeline/` | macOS 14+ | Full project (VisionApp + CameraApp + Benchmarks) |
| Standalone | `vision-standalone/` | macOS 12+ | VisionApp only, distributable to older macOS |

The standalone build exists because Xcode 16.2 / Swift 6.0 emits Foundation symbols that only exist on macOS 14+. The `vision-standalone/` project sets its deployment target to macOS 12, so the binary runs on Monterey and later.

**To build for distribution (macOS 12+):**

```bash
cd vision-standalone
./bundle.sh
```

Creates `dist/` with the binary + Syphon framework. Zip and send.

**To build for local dev (macOS 14+):**

```bash
cd swift-pipeline
swift run VisionApp
```

Use `swift run VisionApp` when testing current source changes. The packaged
`dist/VisionApp` binary can be stale relative to `swift-pipeline/Sources/VisionApp/`.

For an optimized local build:

```bash
cd swift-pipeline
swift build -c release --product VisionApp
.build/release/VisionApp
```

Both projects share the same VisionApp source code. When editing, modify the files in `swift-pipeline/Sources/VisionApp/` and copy them to `vision-standalone/Sources/VisionApp/` before bundling. The only difference is `.external` vs `.externalUnknown` for the camera device type (macOS 14+ renamed it).

## Hand Pose Note

On the affected Sonoma 14.8.4 / M1 Max setup, `Hands` defaults to on as a
workaround for a content-dependent Vision hand-pose bug. For best reliability,
launch VisionApp with hands out of frame, let it process a few no-hand frames,
then bring hands into frame. Toggling `Hands` on while hands are already visible
can trigger `com.apple.Vision Code=9` or garbage confidence values.
