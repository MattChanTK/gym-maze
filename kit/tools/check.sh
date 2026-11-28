#!/bin/bash
# Every fast test, one line per suite plus failures. Suites: KIT_CHECKS in kit.env, one command per line.
# Always runs the kit's own tool tests. Exits 1 if any suite fails.
set -o pipefail
env_before=$(compgen -e)
source "$(dirname "$0")/lib.sh"
# Kit self-tests run without the project's settings: drop every KIT_* variable and everything
# kit.env / .env.local exported, so a project value (e.g. KIT_MAIN_BRANCH=master) can't change them.
clean_env=(env)
for v in $(compgen -e); do
    if [[ $v == KIT_* ]] || ! grep -qx "$v" <<< "$env_before"; then clean_env+=(-u "$v"); fi
done
clean_env+=(KIT_NO_ENV=1)
refuse_protected_checkout || exit 1
failed=()
run() {
    local label="$1"; shift
    bash -c "$*" 2>&1 | sed "s/^/$label/" || failed+=("${label%%:*}: $*")
}
run "Kit:    " "${clean_env[@]}" python3 -I $KIT_DIR/tools/test_kit_env.py
run "Kit:    " "${clean_env[@]}" python3 -I $KIT_DIR/tools/test_scan_secrets.py
run "Kit:    " "${clean_env[@]}" python3 -I $KIT_DIR/tools/test_scrub_report.py
run "Kit:    " "${clean_env[@]}" python3 -I $KIT_DIR/tools/test_roll_changelog.py
[ -f docs/CHANGELOG.md ] && run "Kit:    " python3 $KIT_DIR/tools/roll_changelog.py --check
run "Kit:    " "${clean_env[@]}" $KIT_DIR/tools/test_worktree.sh
run "Kit:    " "${clean_env[@]}" $KIT_DIR/tools/test_lane.sh
run "Kit:    " "${clean_env[@]}" $KIT_DIR/tools/test_qa.sh
run "Kit:    " "${clean_env[@]}" $KIT_DIR/tools/test_kit.sh
run "Kit:    " "${clean_env[@]}" python3 -I $KIT_DIR/tools/test_qa_dashboard.py
run "Kit:    " "${clean_env[@]}" python3 -I $KIT_DIR/tools/test_layout_rules.py
if [ -d "$KIT_DIR/ios" ]; then
    for t in "$KIT_DIR"/ios/test_*.py; do run "Kit:    " "${clean_env[@]}" python3 -I "$t"; done
    run "Kit:    " "${clean_env[@]}" $KIT_DIR/ios/test_swift_gate.sh
    run "Kit:    " "${clean_env[@]}" $KIT_DIR/ios/test_check_layouts.sh
fi
while IFS= read -r cmd; do
    [ -n "${cmd// }" ] && run "Repo:   " "$cmd"
done <<< "${KIT_CHECKS:-}"
if [ ${#failed[@]} -gt 0 ]; then
    printf 'CHECK FAILED: %s\n' "${failed[@]}"
    exit 1
fi
echo "CHECK PASS"
