#!/bin/bash
# Pushes to GitHub after listing the commits and scanning them for secrets. Ask the user before running it (CLAUDE.md).
set -uo pipefail
source "$(dirname "$0")/lib.sh"
git fetch -q origin 2>/dev/null
outgoing=$(git log --oneline "origin/${KIT_MAIN_BRANCH:-main}..HEAD")
[ -n "$outgoing" ] || { echo "Nothing to push"; exit 0; }
echo "$outgoing"
python3 "$KIT_DIR/tools/scan_secrets.py" --push || { echo "Not pushed"; exit 1; }
git push 2>&1 | tail -1
