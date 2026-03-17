#!/bin/bash
# Creates a distributable bundle: dist/VisionApp + dist/Frameworks/Syphon.framework
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

# Build VisionApp
echo "Building VisionApp..."
cd "$SCRIPT_DIR"
swift build -c release -q

# Create dist
rm -rf "$DIST"
mkdir -p "$DIST/Frameworks"

# Copy binary and readme
cp .build/arm64-apple-macosx/release/VisionApp "$DIST/"
cp "$SCRIPT_DIR/dist-README.md" "$DIST/README.md" 2>/dev/null || true

# Copy framework
cp -R "$SYPHON_FW" "$DIST/Frameworks/"

# Fix rpath: point to Frameworks/ next to the binary
install_name_tool -delete_rpath "@loader_path/../../../../Syphon-Framework/build/Release" "$DIST/VisionApp" 2>/dev/null || true
install_name_tool -add_rpath "@loader_path/Frameworks" "$DIST/VisionApp"

echo ""
echo "Done! Distributable at: $DIST/"
echo "  $DIST/VisionApp"
echo "  $DIST/Frameworks/Syphon.framework"
echo ""
echo "To run: cd $DIST && ./VisionApp"
