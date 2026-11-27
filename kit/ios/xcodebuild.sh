#!/bin/bash
# Run Apple's xcodebuild only while holding the heavy lane.
#   kit/ios/xcodebuild.sh [xcodebuild args…]
# Takes a heavy slot via kit/tools/lane.sh if this process does not already hold one.
# Scripts and docs must call this wrapper, not xcodebuild directly.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=kit/tools/lib.sh
source "$here/../tools/lib.sh"
refuse_protected_checkout || exit 1
need_lane heavy "$here/xcodebuild.sh" "$@"
exec xcodebuild "$@"
