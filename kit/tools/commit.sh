#!/bin/bash
# Stages the named files, scans them for secrets (tools/scan_secrets.py), and commits with the attribution line.
#   tools/commit.sh "Subject line" file1 file2 …       (a body: put it after a blank line in the subject, e.g. $'Subject\n\nBody')
# Never stages everything: only the files named. Prints the new commit's hash and size, or what the scan found.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
[ $# -ge 2 ] || { echo 'Usage: tools/commit.sh "message" file…'; exit 1; }
message="$1"; shift
git add -- "$@" || exit 1
if ! python3 "$KIT_DIR/tools/scan_secrets.py"; then echo "Nothing committed: unstage it (git restore --staged <file>) and fix it"; exit 1; fi
git commit -q -m "$message" -m "$ATTRIBUTION" || exit 1
echo "$(git log --oneline -1)  ($(git diff --shortstat HEAD~1 HEAD | sed 's/^ //'))"
left=$(git status --short | grep -v '^??' | wc -l | tr -d ' ')
[ "$left" != "0" ] && echo "Still changed, not committed: $left file(s)"
exit 0
