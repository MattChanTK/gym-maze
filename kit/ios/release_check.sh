#!/bin/bash
# Release-gate steps for KIT_QA_RELEASE_<NAME> (kit/tools/qa.sh release NAME). Each prints PASS or why, exits 0 or 1.
#   kit/ios/release_check.sh same-sha          Debug and Release builds both passed on HEAD (kit/ios/build.sh stamps)
#   kit/ios/release_check.sh build-number      the next build beats the last TestFlight upload (App Store Connect, read-only)
#   kit/ios/release_check.sh phones <set>      the layout set has at least KIT_LAYOUT_MIN_PHONES_<SET> phones
set -uo pipefail
source "$(dirname "$0")/../tools/lib.sh"
gate() { python3 "$KIT_DIR/ios/release_gate.py" "$@"; }
case "${1:-}" in
same-sha)
    tree=$(git rev-parse 'HEAD^{tree}')
    gate same-sha "$(git rev-parse HEAD)" "$(cat "build/.kit-stamps/Debug-$tree" 2>/dev/null)" "$(cat "build/.kit-stamps/Release-$tree" 2>/dev/null)" ;;
build-number)
    last=${KIT_LAST_TESTFLIGHT_BUILD:-$(python3 "$KIT_DIR/ios/asc_build_number.py" --latest)} || exit 1
    gate build-number "${KIT_BUILD_NUMBER:-$((last + 1))}" "$last" ;;
phones)
    set_name=${2:?usage: release_check.sh phones <set>}
    gate phones "$set_name" "$("$KIT_DIR/ios/check_layouts.sh" --phone-count --set "$set_name")" ;;
*) sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
