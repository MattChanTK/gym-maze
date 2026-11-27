#!/bin/bash
# Behavior of tools/worktree.sh: create, list, remove, prune; idempotent; keeps dirty or unmerged trees.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
wt="$root/tools/worktree.sh"
fails=0
checks=0

tmp=$(mktemp -d "${TMPDIR:-/tmp}/kit-worktree-test.XXXXXX")
tmp=$(cd "$tmp" && pwd -P)
trap 'rm -rf "$tmp"' EXIT

ok() { checks=$((checks + 1)); }
fail() { checks=$((checks + 1)); echo "FAIL $1"; fails=$((fails + 1)); }

# A tiny repo with a fake origin, so prune can see origin/main.
git init -q --bare "$tmp/origin.git"
git clone -q "$tmp/origin.git" "$tmp/repo"
git -C "$tmp/repo" config user.email test@example.com
git -C "$tmp/repo" config user.name Test
echo hello > "$tmp/repo/README"
git -C "$tmp/repo" add README
git -C "$tmp/repo" commit -q -m init
git -C "$tmp/repo" branch -M main
git -C "$tmp/repo" push -q -u origin main

export KIT_WORKTREE_ROOT="$tmp/trees"
cd "$tmp/repo"

"$wt" >/dev/null 2>&1 && fail "no args should exit non-zero" || ok
"$wt" --help >/dev/null && ok || fail "help should exit 0"

path=$("$wt" create feature-a)
if [ "$path" = "$tmp/trees/feature-a" ] && [ -d "$path" ] && [ "$(git -C "$path" branch --show-current)" = feature-a ]; then
    ok
else
    fail "create should add a worktree at the hidden root: got '$path'"
fi

again=$("$wt" create feature-a)
if [ "$again" = "$path" ]; then
    ok
else
    fail "create is not idempotent: '$again'"
fi

list=$("$wt" list)
case "$list" in
    *"$path"*feature-a*clean*unmerged*) ok ;;
    *) fail "list should show the new unmerged clean worktree: $list" ;;
esac

# Unmerged and clean: prune must keep it.
keep=$("$wt" prune)
case "$keep" in
    *keep\ unmerged\ "$path"*) ok ;;
    *) fail "prune kept an unmerged tree? $keep" ;;
esac
[ -d "$path" ] && ok || fail "prune removed an unmerged worktree"

# Dirty: prune must keep it even if we merge later; first make it dirty.
echo dirty >> "$path/README"
"$wt" prune >/dev/null
[ -d "$path" ] && ok || fail "prune removed a dirty worktree"
"$wt" remove feature-a >/dev/null 2>&1 && fail "remove of a dirty tree should fail without --force" || ok
"$wt" remove feature-a --force && ok || fail "remove --force should drop a dirty tree"
[ ! -d "$path" ] && ok || fail "force-removed worktree still exists"

# Merged + clean is removed by prune. Recreate, merge (fast-forward via a commit on main that includes the branch).
path=$("$wt" create feature-b)
echo more > "$path/FEATURE"
git -C "$path" add FEATURE
git -C "$path" commit -q -m feature
git -C "$tmp/repo" merge -q --no-edit feature-b
git -C "$tmp/repo" push -q origin main
# The worktree is still on feature-b, now an ancestor of origin/main, and clean.
out=$("$wt" prune)
case "$out" in
    *removed\ "$path"*) ok ;;
    *) fail "prune should remove a merged clean worktree: $out" ;;
esac
[ ! -d "$path" ] && ok || fail "merged worktree still exists after prune"

# Squash merge: unique commit is not an ancestor, but git cherry sees it as applied.
path=$("$wt" create feature-c)
echo squash-me > "$path/SQUASH"
git -C "$path" add SQUASH
git -C "$path" commit -q -m 'squash me'
# squash into main in the primary checkout (not the worktree)
git -C "$tmp/repo" checkout -q main
git -C "$tmp/repo" merge -q --squash feature-c
git -C "$tmp/repo" commit -q -m 'squash feature-c'
git -C "$tmp/repo" push -q origin main
out=$("$wt" prune)
case "$out" in
    *removed\ "$path"*) ok ;;
    *) fail "prune should remove a squash-merged clean worktree: $out" ;;
esac

# remove of a missing name is success (idempotent)
"$wt" remove definitely-missing && ok || fail "remove of a missing worktree should succeed"

# The hidden root stays under KIT_WORKTREE_ROOT, never under Projects/protected_checkout.
case "$(find "$tmp/trees" -mindepth 1 -maxdepth 1 2>/dev/null)$( "$wt" list )" in
    *protected_checkout*) fail "worktrees escaped into protected_checkout" ;;
    *) ok ;;
esac

[ $fails -eq 0 ] && echo "PASS $checks worktree.sh checks" || { echo "FAIL $fails of $checks worktree.sh checks"; exit 1; }
