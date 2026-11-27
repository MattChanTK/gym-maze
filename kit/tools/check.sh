#!/bin/bash
# Every fast test, one line per suite plus failures. Suites: KIT_CHECKS in kit.env, one command per line.
# Always runs the kit's own tool tests. Exits 1 if any suite fails.
set -o pipefail
source "$(dirname "$0")/lib.sh"
refuse_protected_checkout || exit 1
failed=()
run() {
    local label="$1"; shift
    bash -c "$*" 2>&1 | sed "s/^/$label/" || failed+=("${label%%:*}: $*")
}
run "Kit:    " python3 -I $KIT_DIR/tools/test_scan_secrets.py
run "Kit:    " python3 -I $KIT_DIR/tools/test_scrub_report.py
run "Kit:    " python3 -I $KIT_DIR/tools/test_roll_changelog.py
[ -f docs/CHANGELOG.md ] && run "Kit:    " python3 $KIT_DIR/tools/roll_changelog.py --check
run "Kit:    " $KIT_DIR/tools/test_worktree.sh
run "Kit:    " $KIT_DIR/tools/test_lane.sh
if [ -d "$KIT_DIR/ios" ]; then
    for t in "$KIT_DIR"/ios/test_*.py; do run "Kit:    " python3 -I "$t"; done
    run "Kit:    " $KIT_DIR/ios/test_swift_gate.sh
fi
while IFS= read -r cmd; do
    [ -n "${cmd// }" ] && run "Repo:   " "$cmd"
done <<< "${KIT_CHECKS:-}"
if [ ${#failed[@]} -gt 0 ]; then
    printf 'CHECK FAILED: %s\n' "${failed[@]}"
    exit 1
fi
echo "CHECK PASS"
