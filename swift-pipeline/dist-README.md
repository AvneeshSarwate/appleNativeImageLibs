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
