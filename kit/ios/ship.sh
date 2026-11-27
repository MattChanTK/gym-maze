#!/bin/bash
# Builds the app, installs it on the iPhone and opens it (device, team and bundle from CLAUDE.local.md).
# Prints one line, or the build errors. Ask the user before running it (CLAUDE.md).
#   kit/ios/ship.sh            build, install, launch
#   kit/ios/ship.sh --build    build only
set -uo pipefail
source "$(dirname "$0")/../tools/lib.sh"
refuse_protected_checkout || exit 1
load_local DEVICE TEAM BUNDLE
need_lane heavy "$KIT_DIR/ios/ship.sh" "$@"
out=$("$KIT_DIR/ios/xcodebuild.sh" -project "$KIT_XCODE_PROJECT" -scheme "$KIT_SCHEME" -configuration Debug -destination "id=$DEVICE" -derivedDataPath build \
      -allowProvisioningUpdates DEVELOPMENT_TEAM="$TEAM" PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE" build 2>&1)
if ! grep -q "BUILD SUCCEEDED" <<< "$out"; then grep -E "error:" <<< "$out" | sort -u | head -20; echo "BUILD FAILED"; exit 1; fi
[ "${1:-}" = "--build" ] && { echo "BUILD SUCCEEDED"; exit 0; }
xcrun devicectl device install app --device "$DEVICE" build/Build/Products/"Debug-iphoneos/$KIT_SCHEME.app" > build/install.log 2>&1 \
    || { tail -5 build/install.log; echo "INSTALL FAILED (phone locked or unplugged?)"; exit 1; }
xcrun devicectl device process launch --terminate-existing --device "$DEVICE" "$BUNDLE" > build/launch.log 2>&1 \
    || { tail -3 build/launch.log; echo "Installed, but the launch failed (unlock the phone)"; exit 1; }
echo "Built, installed and opened on the iPhone"
