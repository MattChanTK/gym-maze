#!/bin/bash
# Run `xcrun simctl` only while holding the heavy lane.
#   kit/ios/simctl.sh [simctl args…]
# Takes a heavy slot via kit/tools/lane.sh if this process does not already hold one.
# Scripts and docs must call this wrapper, not `xcrun simctl` or `simctl` directly.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=kit/tools/lib.sh
source "$here/../tools/lib.sh"
refuse_protected_checkout || exit 1
need_lane heavy "$here/simctl.sh" "$@"
exec xcrun simctl "$@"
