#!/bin/bash
# Mac pass (OPS-007): runs what the Swift workflow runs (KIT_TEST_CMD and an unsigned Debug build) on this Mac,
# then records the pass for HEAD's tree as a git note (refs/notes/mac-pass) and pushes that note.
# A PR whose head has the same tree then skips the Swift workflow's Mac job, saving GitHub macOS minutes.
# GitHub runs that job only once a PR has the ready-to-merge label. Needs a clean checkout (the tree tested is the tree recorded). Pushes only the note, never a branch.
# --no-push: record the note locally only. 
set -uo pipefail
source "$(dirname "$0")/../tools/lib.sh"
push=1
[ "${1:-}" = "--no-push" ] && push=0
notes=refs/notes/mac-pass

if [ -n "$(git status --porcelain)" ]; then
    echo "Uncommitted or untracked files: commit first (the pass is recorded for a commit's tree)"; exit 1
fi
refuse_protected_checkout || exit 1
need_lane heavy "$KIT_DIR/ios/mac_pass.sh" "$@"
tree=$(git rev-parse 'HEAD^{tree}')
commit=$(git rev-parse --short HEAD)

if [ -z "${MAC_PASS_SKIP_RUN:-}" ]; then   # test hook only
    result=$(bash -c "${KIT_TEST_CMD:-tools/test.sh}" 2>&1); status=$?
    [ $status -eq 0 ] || { echo "$result" | tail -20; echo "Tests failed: no Mac pass"; exit 1; }
    echo "Tests passed"
    {
        mkdir -p build
        "$KIT_DIR/ios/xcodebuild.sh" -project "$KIT_XCODE_PROJECT" -scheme "$KIT_SCHEME" -configuration Debug \
            -destination 'generic/platform=iOS' -derivedDataPath build \
            CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO CODE_SIGN_IDENTITY="" \
            build > build/mac-pass-debug.log 2>&1
        if [ $? -ne 0 ] || ! grep -q "BUILD SUCCEEDED" build/mac-pass-debug.log; then
            grep -E "error:" build/mac-pass-debug.log | sort -u | head -20
            echo "Debug build failed: no Mac pass"; exit 1
        fi
        echo "DEBUG BUILD SUCCEEDED"
    }
fi

xcode=$("$KIT_DIR/ios/xcodebuild.sh" -version 2>/dev/null | head -1)
where="GitHub Actions"
[ -n "${GITHUB_ACTIONS:-}" ] || where="Mac"
note="Mac pass $(date -u +%Y-%m-%dT%H:%MZ) commit $commit ${xcode:-unknown Xcode} ($where): ${KIT_TEST_CMD:-tools/test.sh}, unsigned Debug build"
git notes --ref=$notes add -f -m "$note" "$tree" || exit 1
[ $push -eq 1 ] || { echo "Mac pass recorded for tree ${tree:0:12} (not pushed)"; exit 0; }

# Other sessions push notes too: merge theirs in and retry.
for try in 1 2 3 4; do
    if git fetch -q origin "+$notes:refs/notes/mac-pass-remote" 2>/dev/null; then
        git notes --ref=$notes merge -q -s cat_sort_uniq refs/notes/mac-pass-remote >/dev/null 2>&1 || true
    fi
    if git push -q origin "$notes" 2>/dev/null; then
        echo "Mac pass recorded and pushed for commit $commit (tree ${tree:0:12})"; exit 0
    fi
    sleep $((try * 2))
done
echo "Mac pass recorded locally but not pushed (GitHub will run the Swift workflow)"
[ -n "${MAC_PASS_OK_IF_PUSH_FAILS:-}" ] && exit 0
exit 1
