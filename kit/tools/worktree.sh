#!/bin/bash
# Manage this repo's worktrees under one hidden directory.
#   tools/worktree.sh create <name> [start-point]
#   tools/worktree.sh list
#   tools/worktree.sh remove <name> [--force]
#   tools/worktree.sh prune
#
# Root: $KIT_WORKTREE_ROOT, or $HOME/Projects/.kit-worktrees.
# Operates on the git repo of the current directory. create is idempotent: a
# worktree already at that path is reused. prune removes a worktree only when
# its branch is merged (including squash) and the tree is clean.
set -euo pipefail
MAIN=${KIT_MAIN_BRANCH:-main}

usage() {
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
    echo "usage: tools/worktree.sh create <name> [start-point]"
    echo "       tools/worktree.sh list"
    echo "       tools/worktree.sh remove <name> [--force]"
    echo "       tools/worktree.sh prune"
}

root_dir() {
    local d=${KIT_WORKTREE_ROOT:-$HOME/Projects/.kit-worktrees}
    mkdir -p "$d"
    (cd "$d" && pwd -P)
}

repo() {
    git rev-parse --show-toplevel
}

worktree_path() {
    local name=$1
    case "$name" in
        ""|*/*|*..*|.*)
            echo "worktree.sh: invalid name '$name'" >&2
            return 2
            ;;
    esac
    echo "$(root_dir)/$name"
}

is_worktree_of_repo() {
    local path=$1 here there
    [ -d "$path" ] || return 1
    here=$(git rev-parse --path-format=absolute --git-common-dir) || return 1
    there=$(git -C "$path" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || return 1
    [ "$here" = "$there" ]
}

branch_of() {
    local path=$1
    git -C "$path" branch --show-current 2>/dev/null || true
}

is_clean() {
    local path=$1
    [ -z "$(git -C "$path" status --porcelain)" ]
}

# A worktree is merged when unique work has landed on main, not merely because
# a new branch still points at origin/$MAIN.
is_merged() {
    # A brand-new branch created from main is an ancestor of main and must
    # not be pruned. Treat it as merged only when unique work has landed:
    # squash/cherry-pick (git cherry reports applied commits), a merged
    # GitHub PR, or a fast-forward whose branch reflog has a commit.
    local path=$1
    local head main_ref cherry branch reflog
    head=$(git -C "$path" rev-parse HEAD 2>/dev/null) || return 1
    branch=$(branch_of "$path")
    if git -C "$path" rev-parse --verify -q origin/$MAIN >/dev/null; then
        main_ref=origin/$MAIN
    elif git -C "$path" rev-parse --verify -q "$MAIN" >/dev/null; then
        main_ref=$MAIN
    else
        return 1
    fi
    cherry=$(git -C "$path" cherry "$main_ref" "$head" 2>/dev/null) || cherry=""
    case "$cherry" in
        "") ;;
        *"+ "*) ;;
        *) return 0 ;;
    esac
    if git -C "$path" merge-base --is-ancestor "$head" "$main_ref"; then
        if [ -n "$branch" ]; then
            reflog=$(git -C "$path" reflog show "$branch" 2>/dev/null || true)
            case "$reflog" in
                *": commit"*) return 0 ;;
            esac
        fi
    fi
    if [ -n "$branch" ] && command -v gh >/dev/null 2>&1; then
        if [ "$(gh pr list --head "$branch" --state merged --json number --jq 'length' 2>/dev/null || echo 0)" != 0 ]; then
            return 0
        fi
    fi
    return 1
}

cmd_create() {
    local name=${1:-}
    local start=${2:-}
    [ -n "$name" ] || { echo "worktree.sh: create needs a name" >&2; return 2; }
    local path
    path=$(worktree_path "$name") || return $?
    if [ -e "$path" ]; then
        if is_worktree_of_repo "$path"; then
            echo "$path"
            return 0
        fi
        echo "worktree.sh: $path already exists and is not a worktree of this repo" >&2
        return 1
    fi
    mkdir -p "$(root_dir)"
    if git show-ref --verify --quiet "refs/heads/$name"; then
        git worktree add "$path" "$name" >&2
    elif [ -n "$start" ]; then
        git worktree add -b "$name" "$path" "$start" >&2
    elif git rev-parse --verify -q origin/$MAIN >/dev/null; then
        git worktree add -b "$name" "$path" origin/$MAIN >&2
    else
        git worktree add -b "$name" "$path" HEAD >&2
    fi
    (cd "$path" && pwd -P)
}

cmd_list() {
    local root top path branch state
    root=$(root_dir)
    top=$(repo)
    git worktree list --porcelain | awk '
        $1 == "worktree" { path=$2; branch=""; sha="" }
        $1 == "HEAD" { sha=$2 }
        $1 == "branch" { sub(/^refs\/heads\//, "", $2); branch=$2 }
        $1 == "detached" { branch="detached" }
        NF == 0 && path != "" { print path "\t" branch "\t" sha; path="" }
        END { if (path != "") print path "\t" branch "\t" sha }
    ' | while IFS=$(printf '\t') read -r path branch sha; do
        path=$(cd "$path" 2>/dev/null && pwd -P) || continue
        case "$path" in
            "$root"/*) ;;
            *) continue ;;
        esac
        if is_clean "$path"; then
            state=clean
        else
            state=dirty
        fi
        if is_merged "$path"; then
            state="$state,merged"
        else
            state="$state,unmerged"
        fi
        printf '%s\t%s\t%s\n' "$path" "${branch:-?}" "$state"
    done
}

cmd_remove() {
    local name=${1:-}
    local force=0
    [ -n "$name" ] || { echo "worktree.sh: remove needs a name" >&2; return 2; }
    shift || true
    for a in "${@+"$@"}"; do
        case "$a" in
            --force) force=1 ;;
            *) echo "worktree.sh: unknown option $a" >&2; return 2 ;;
        esac
    done
    local path
    path=$(worktree_path "$name") || return $?
    if [ ! -e "$path" ]; then
        return 0
    fi
    if ! is_worktree_of_repo "$path"; then
        echo "worktree.sh: $path is not a worktree of this repo" >&2
        return 1
    fi
    if [ "$force" -eq 0 ] && ! is_clean "$path"; then
        echo "worktree.sh: $path is dirty; pass --force to remove it anyway" >&2
        return 1
    fi
    if [ "$force" -eq 1 ]; then
        git worktree remove --force "$path"
    else
        git worktree remove "$path"
    fi
}

cmd_prune() {
    local root path name
    root=$(root_dir)
    if git remote get-url origin >/dev/null 2>&1; then
        git fetch -q origin 2>/dev/null || true
    fi
    # Snapshot paths first: removing during the walk confuses git worktree list.
    paths=
    while IFS=$(printf '\t') read -r path _ _; do
        [ -n "$path" ] || continue
        paths="$paths
$path"
    done <<LIST
$(cmd_list)
LIST
    # cmd_list already filters to the hidden root.
    printf '%s\n' "$paths" | while IFS= read -r path; do
        [ -n "$path" ] || continue
        if ! is_clean "$path"; then
            echo "keep dirty $path"
            continue
        fi
        if ! is_merged "$path"; then
            echo "keep unmerged $path"
            continue
        fi
        name=$(basename "$path")
        if git worktree remove "$path"; then
            echo "removed $path"
        else
            echo "worktree.sh: could not remove $path" >&2
        fi
    done
    git worktree prune
}

mode=${1:-}
shift || true
case "$mode" in
    create) cmd_create "${@+"$@"}" ;;
    list) cmd_list ;;
    remove) cmd_remove "${@+"$@"}" ;;
    prune) cmd_prune ;;
    --help|-h) usage; exit 0 ;;
    "") usage; exit 2 ;;
    *) echo "worktree.sh: unknown command '$mode' (try --help)" >&2; exit 2 ;;
esac
