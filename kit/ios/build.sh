#!/bin/bash
# Unsigned build of the app for a QA gate: kit/ios/build.sh Debug|Release. On a clean tree it records the pass
# (build/.kit-stamps/<config>-<tree> holds the commit) for `kit/ios/release_check.sh same-sha`.
set -uo pipefail
source "$(dirname "$0")/../tools/lib.sh"
cfg=${1:?usage: kit/ios/build.sh Debug|Release}
refuse_protected_checkout || exit 1
need_lane heavy "$KIT_DIR/ios/build.sh" "$@"
mkdir -p build/.kit-stamps
log="build/qa-$cfg.log"
"$KIT_DIR/ios/xcodebuild.sh" -project "$KIT_XCODE_PROJECT" -scheme "$KIT_SCHEME" -configuration "$cfg" \
    -destination 'generic/platform=iOS' -derivedDataPath build \
    CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO CODE_SIGN_IDENTITY="" build > "$log" 2>&1
if [ $? -ne 0 ] || ! grep -q "BUILD SUCCEEDED" "$log"; then
    grep -E "error:" "$log" | sort -u | head -20; echo "$cfg BUILD FAILED"; exit 1
fi
[ -z "$(git status --porcelain)" ] && git rev-parse HEAD > "build/.kit-stamps/$cfg-$(git rev-parse 'HEAD^{tree}')"
echo "$cfg BUILD SUCCEEDED"
