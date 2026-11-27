#!/bin/bash
# kit/tools/qa.sh against a throwaway repo with a bare origin and PR refs. No network.
set -uo pipefail
kit=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/kit-qa-test.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
unset KIT_LANE_RANK KIT_LANE_FD KIT_NO_ENV
export KIT_LANE_DIR="$tmp/lanes" KIT_QA_LOG_DIR="$tmp/logs" KIT_PROTECTED_CHECKOUT=
checks=0 fails=0
ok() { checks=$((checks+1)); }
fail() { checks=$((checks+1)); fails=$((fails+1)); echo "FAIL $1"; }
expect() { if printf '%s\n' "$2" | grep -q -- "$3"; then ok; else fail "$1: no '$3' in: $(printf '%s' "$2" | tail -5)"; fi; }
g() { git -c user.email=t@e -c user.name=t -c commit.gpgsign=false "$@"; }

git init -q --bare -b main "$tmp/origin.git"
git init -q -b main "$tmp/repo"; cd "$tmp/repo"
cp -R "$kit" kit; find kit -name __pycache__ -prune -exec rm -rf {} +
cat > kit.env <<'ENV'
KIT_PROJECT=demo
KIT_QA_SMOKE="echo smoke-ran"
KIT_QA_FULL="echo full-ran
test ! -f broken"
KIT_QA_SELECT='grep -q "^docs/" && echo "echo docs-only-ran" || echo FULL'
KIT_QA_APP_CHANGED='grep -q "^app/" && echo true || echo false'
KIT_QA_BUILDS="echo build-ran"
KIT_QA_UI_CHANGED='grep -q "^app/ui" && echo true || echo false'
KIT_QA_LAYOUT="echo layout-ran"
KIT_QA_AFTER_PASS="echo after-pass-ran"
KIT_QA_RELEASE_ALPHA="echo step-one
echo step-two"
KIT_QA_RELEASE_BAD="echo fine
false"
ENV
mkdir -p docs app; echo a > docs/a.md; echo a > app/a.txt
g add -A; g commit -qm init; git remote add origin "$tmp/origin.git"; git push -q origin main 2>/dev/null
pr() {   # pr <n> <file> <content>: a branch off main pushed as refs/pull/<n>/head
    g checkout -q -b "b$1" main; mkdir -p "$(dirname "$2")"; echo "$3" > "$2"; g add -A; g commit -qm "pr $1"
    git push -q origin "HEAD:refs/pull/$1/head" 2>/dev/null; g checkout -q main
}
pr 1 docs/b.md docs; pr 2 app/x.txt app; pr 3 app/ui/v.txt ui; pr 4 broken x; pr 5 app/x.txt other

out=$(kit/tools/qa.sh smoke 2>&1); expect "smoke runs" "$out" "smoke-ran"; expect "smoke passes" "$out" "QA smoke: PASS"
ls "$KIT_QA_LOG_DIR"/qa-demo-smoke-*.log >/dev/null 2>&1 && ok || fail "smoke log written"
out=$(kit/tools/qa.sh pr 1 2>&1); expect "docs PR selected" "$out" "docs-only-ran"; expect "docs PR no builds" "$out" "skip builds"; expect "pr 1 pass" "$out" "QA pr 1: PASS"
out=$(kit/tools/qa.sh pr 2 2>&1); expect "app PR full" "$out" "full-ran"; expect "app PR builds" "$out" "build-ran"
out=$(kit/tools/qa.sh pr 4 2>&1); expect "broken PR fails" "$out" "QA pr 4: FAIL"
out=$(kit/tools/qa.sh batch 1 3 2>&1); expect "batch layout" "$out" "layout-ran"; expect "batch after-pass" "$out" "after-pass-ran"; expect "batch pass" "$out" "QA batch 1 3: PASS"
out=$(kit/tools/qa.sh batch 1 2 2>&1); expect "batch no ui skip" "$out" "skip layout check"
out=$(kit/tools/qa.sh batch 2 5 2>&1); expect "batch conflict" "$out" "merge conflict stacking #5"; expect "conflict fails" "$out" "QA batch 2 5: FAIL"
out=$(kit/tools/qa.sh batch 1 4 2>&1); expect "batch with failing PR" "$out" "QA batch 1 4: FAIL"
printf '%s\n' "$out" | grep -q after-pass-ran && fail "no after-pass on failure" || ok
out=$(kit/tools/qa.sh alpha 2>&1); expect "release GO" "$out" "QA alpha: GO"; expect "release steps" "$out" "step-two"
out=$(kit/tools/qa.sh release bad 2>&1); expect "release NO-GO" "$out" "QA bad: NO-GO"
out=$(kit/tools/qa.sh --print-plan batch 1 2 2>&1); expect "plan batch" "$out" "PLAN batch"; expect "plan never" "$out" "never merge"
out=$(kit/tools/qa.sh pr x 2>&1); expect "bad pr arg" "$out" "one pull request number"
git worktree prune; if git worktree list | grep -qE "kit-qa-(pr|stack|pass)"; then git worktree list; fail "temp worktrees cleaned"; else ok; fi
[ $fails -eq 0 ] && echo "PASS $checks qa.sh checks" || { echo "FAIL $fails of $checks qa.sh checks"; exit 1; }
