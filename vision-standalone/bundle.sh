#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST="$SCRIPT_DIR/dist"
SYPHON_FW="$SCRIPT_DIR/../Syphon-Framework/build/Release/Syphon.framework"

# Build Syphon if needed
if [ ! -d "$SYPHON_FW" ]; then
    echo "Building Syphon framework..."
    xcodebuild -project "$SCRIPT_DIR/../Syphon-Framework/Syphon.xcodeproj" \
        -scheme Syphon -configuration Release \
        BUILD_DIR="$SCRIPT_DIR/../Syphon-Framework/build" -quiet
fi

# Build VisionApp targeting macOS 12
echo "Building VisionApp (macOS 12 compat)..."
cd "$SCRIPT_DIR"
swift build -c release -q

# Create dist
rm -rf "$DIST"
mkdir -p "$DIST/Frameworks"

cp .build/arm64-apple-macosx/release/VisionApp "$DIST/"
cp -R "$SYPHON_FW" "$DIST/Frameworks/"
cp "$SCRIPT_DIR/../swift-pipeline/dist-README.md" "$DIST/README.md" 2>/dev/null || true

echo ""
echo "Done! Distributable at: $DIST/"
echo "To run: cd $DIST && ./VisionApp"
