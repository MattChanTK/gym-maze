#!/bin/bash
# Uploads a new TestFlight build: asks App Store Connect for the next build number (kit/ios/asc_build_number.py, read-only),
# archives with it, uploads with the App Store Connect key (.env.local: ASC_KEY_ID, ASC_ISSUER_ID; the key in
# ~/.appstoreconnect/private_keys). Commits nothing: the number lives on App Store Connect, not in the repo.
# Type yes when asked, or pass --i-confirm-upload for an approved non-interactive run.
# Do not run it from the protected main checkout. Prints the build number, or the error.
set -uo pipefail
source "$(dirname "$0")/../tools/lib.sh"
refuse_protected_checkout || exit 1

# Confirmation first, so a typo or a missing tty does not take the heavy lane.
confirm_upload=0
dry_run=0
tf_args=()
for a in "${@+"$@"}"; do
    case "$a" in
        --i-confirm-upload) confirm_upload=1 ;;
        --dry-run) dry_run=1 ;;
        --help|-h)
            echo "usage: kit/ios/testflight.sh [--i-confirm-upload] [--dry-run]"
            echo "  Type yes when asked, or pass --i-confirm-upload for an approved non-interactive run."
            echo "  --dry-run stops after confirmation and taking the heavy lane (no archive or upload)."
            exit 0
            ;;
        *) echo "testflight.sh: unknown option $a"; exit 2 ;;
    esac
done
if [ "${KIT_TESTFLIGHT_CONFIRMED:-}" = 1 ]; then
    confirm_upload=1
fi
if [ "$confirm_upload" -eq 0 ]; then
    if [ -t 0 ]; then
        printf "Type yes to upload to TestFlight: " >&2
    fi
    if ! read -r ans; then
        echo "testflight.sh: type yes to upload, or pass --i-confirm-upload for an approved non-interactive run" >&2
        exit 2
    fi
    [ "$ans" = yes ] || { echo "testflight.sh: aborted"; exit 2; }
fi
export KIT_TESTFLIGHT_CONFIRMED=1

need_lane heavy "$KIT_DIR/ios/testflight.sh" "$@"
if [ "$dry_run" -eq 1 ]; then
    echo "testflight.sh: dry-run confirmed, heavy lane held; stopping before archive"
    exit 0
fi
load_local DEVICE TEAM BUNDLE
export BUNDLE
next=$(python3 "$KIT_DIR/ios/asc_build_number.py") || exit 1
case "$next" in ''|*[!0-9]*) echo "No build number from App Store Connect: $next"; exit 1 ;; esac
mkdir -p build
cat > build/ExportOptions.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>method</key><string>app-store-connect</string><key>destination</key><string>upload</string>
<key>teamID</key><string>$TEAM</string><key>signingStyle</key><string>automatic</string>
</dict></plist>
PLIST
rm -rf "build/$KIT_SCHEME.xcarchive"
"$KIT_DIR/ios/xcodebuild.sh" -project "$KIT_XCODE_PROJECT" -scheme "$KIT_SCHEME" -configuration Release -destination 'generic/platform=iOS' -derivedDataPath build \
    -archivePath "build/$KIT_SCHEME.xcarchive" -allowProvisioningUpdates DEVELOPMENT_TEAM="$TEAM" PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE" CURRENT_PROJECT_VERSION="$next" archive \
    > build/archive.log 2>&1 || { grep -E "error:" build/archive.log | sort -u | head; echo "ARCHIVE FAILED (build $next)"; exit 1; }
"$KIT_DIR/ios/xcodebuild.sh" -exportArchive -archivePath "build/$KIT_SCHEME.xcarchive" -exportOptionsPlist build/ExportOptions.plist -exportPath build/export \
    -allowProvisioningUpdates -authenticationKeyPath ~/.appstoreconnect/private_keys/AuthKey_$ASC_KEY_ID.p8 \
    -authenticationKeyID "$ASC_KEY_ID" -authenticationKeyIssuerID "$ASC_ISSUER_ID" > build/upload.log 2>&1
if ! grep -q "EXPORT SUCCEEDED" build/upload.log; then
    grep -iE "error" build/upload.log | sort -u | head -5
    echo "UPLOAD FAILED (no network to Apple? see docs/CHEATSHEET.md, When something fails)"; exit 1
fi
version=$(sed -nE 's/.*MARKETING_VERSION = ([0-9.]+);.*/\1/p' "$KIT_XCODE_PROJECT/project.pbxproj" | head -1)
echo "Uploaded build $version ($next) to TestFlight; Apple processes it for a few minutes to an hour"
