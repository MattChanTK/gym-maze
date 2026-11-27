#!/bin/bash
# Used by the Swift workflow's gate (OPS-007): prints "skip" when <commit>'s tree has a Mac pass note
# (kit/ios/mac_pass.sh) in refs/notes/mac-pass, else "run". Fetches the notes from origin first unless --local.
# Never fails the build: any error prints "run".
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
local_only=0
[ "${1:-}" = "--local" ] && { local_only=1; shift; }
commit="${1:-HEAD}"
[ $local_only -eq 1 ] || git fetch -q --depth=1 origin "+refs/notes/mac-pass:refs/notes/mac-pass" 2>/dev/null || true
tree=$(git rev-parse "$commit^{tree}" 2>/dev/null) || { echo run; exit 0; }
if note=$(git notes --ref=refs/notes/mac-pass show "$tree" 2>/dev/null) && [ -n "$note" ]; then
    echo "$note" | tail -1 >&2
    echo skip
else
    echo "No Mac pass for tree ${tree:0:12}" >&2
    echo run
fi
