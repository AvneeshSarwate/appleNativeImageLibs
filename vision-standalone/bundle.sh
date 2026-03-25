#!/bin/bash
# Builds VisionApp for macOS 12+ and creates a versioned zip for distribution.
# Syncs sources from swift-pipeline/Sources/VisionApp/ first, patching macOS 14 APIs.
#
# Usage: cd vision-standalone && ./bundle.sh
# Output: VisionApp-v<N>.zip (self-contained, ready to send)
# Version is auto-incremented from VERSION file each build.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST="$SCRIPT_DIR/dist"
MAIN_SRC="$SCRIPT_DIR/../swift-pipeline/Sources/VisionApp"
COMPAT_SRC="$SCRIPT_DIR/Sources/VisionApp"
SYPHON_FW="$SCRIPT_DIR/../Syphon-Framework/build/Release/Syphon.framework"
VERSION_FILE="$SCRIPT_DIR/VERSION"

# Read and increment version
VERSION=$(cat "$VERSION_FILE" 2>/dev/null || echo "0")
VERSION=$((VERSION + 1))
echo "$VERSION" > "$VERSION_FILE"

echo "=== Building VisionApp v$VERSION ==="

# Sync sources from main project
echo "Syncing sources from swift-pipeline..."
cp "$MAIN_SRC"/*.swift "$COMPAT_SRC/"

# Patch macOS 14+ APIs for macOS 12 compat
# .external (macOS 14+) -> .externalUnknown (macOS 11+)
sed -i '' 's/\.external\]/.externalUnknown]/g' "$COMPAT_SRC/VisionPipelineEngine.swift"

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

# Zip
ZIP_NAME="VisionApp-v${VERSION}.zip"
cd "$DIST"
zip -r -q "$SCRIPT_DIR/$ZIP_NAME" .
cd "$SCRIPT_DIR"

echo ""
echo "=== VisionApp v$VERSION ==="
echo "Dist:  $DIST/"
echo "Zip:   $SCRIPT_DIR/$ZIP_NAME"
echo "Send:  $ZIP_NAME"
