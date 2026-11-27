#!/bin/bash
# Single QA entry point for any repo. Runs the commands kit.env names, queued through lane.sh so parallel
# sessions on one machine take turns. Never merges, pushes to main, uploads, or calls paid APIs.
#   kit/tools/qa.sh smoke              quick dev check on the current tree (dirty OK)
#   kit/tools/qa.sh pr N               one PR's checks in a temp worktree (path-selected when KIT_QA_SELECT is set)
#   kit/tools/qa.sh batch N [N…]       each PR's full checks, then stack them off main (stop on conflicts),
#                                      one layout check if UI paths changed, then KIT_QA_AFTER_PASS per PR head
#   kit/tools/qa.sh release NAME       release gate on the current tree: every KIT_QA_RELEASE_<NAME> step; GO or NO-GO
#   kit/tools/qa.sh NAME               same, when KIT_QA_RELEASE_<NAME> is set (e.g. qa.sh alpha)
#   --print-plan                       print the steps without running them
# Settings (kit.env; command lists are one command per line, run with bash from the tree under test):
#   KIT_QA_SMOKE, KIT_QA_FULL          default kit/tools/check.sh
#   KIT_QA_SELECT                      stdin: changed paths; stdout: commands to run, or FULL (default: FULL)
#   KIT_QA_APP_CHANGED + KIT_QA_BUILDS stdin paths -> true/false; builds to run when true (e.g. Debug and Release)
#   KIT_QA_UI_CHANGED + KIT_QA_LAYOUT  stdin paths -> true/false; the layout check batch runs when true
#   KIT_QA_AFTER_PASS                  run in each PR head after a batch passes (e.g. kit/ios/mac_pass.sh)
#   KIT_QA_RELEASE_<NAME>              the release gate's steps
#   KIT_QA_LOG_DIR                     logs (default ~/qa-logs; the QA dashboard tails them)
set -euo pipefail
source "$(dirname "$0")/lib.sh"
QA_ROOT=$(pwd)
refuse_protected_checkout || exit 1

print_plan=0
usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; }
args=()
for a in "$@"; do
    case "$a" in
        --print-plan) print_plan=1 ;;
        --help|-h) usage; exit 0 ;;
        --*) echo "qa.sh: unknown option $a (try --help)"; exit 2 ;;
        *) args+=("$a") ;;
    esac
done
set -- "${args[@]+"${args[@]}"}"
mode=${1:-}
[ -n "$mode" ] || { usage; exit 2; }
shift || true
MAIN=${KIT_MAIN_BRANCH:-main}

is_pr_num() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
# cmds VAR [default]: the command list in VAR, one per line, blank lines dropped.
cmds() { local v="${!1:-${2:-}}"; printf '%s\n' "$v" | sed '/^[[:space:]]*$/d'; }
release_var() { printf 'KIT_QA_RELEASE_%s' "$(printf '%s' "$1" | tr '[:lower:]-' '[:upper:]_')"; }

# Temp worktrees and refs, dropped on exit.
worktrees=() local_refs=() local_branches=()
cleanup() {
    local p
    for p in "${worktrees[@]+"${worktrees[@]}"}"; do git -C "$QA_ROOT" worktree remove --force "$p" >/dev/null 2>&1 || rm -rf "$p"; done
    for p in "${local_branches[@]+"${local_branches[@]}"}"; do git -C "$QA_ROOT" branch -D "$p" >/dev/null 2>&1 || true; done
    for p in "${local_refs[@]+"${local_refs[@]}"}"; do git -C "$QA_ROOT" update-ref -d "$p" >/dev/null 2>&1 || true; done
}
trap cleanup EXIT

# run_list VAR [default]: run each command in the current directory, with this tree's kit.env; 1 if any failed.
run_list() {
    local st=0 c
    while IFS= read -r c; do
        echo "== $c =="
        ( [ -f kit.env ] && { set -a; source kit.env; set +a; }; bash -c "$c" ) || { echo "FAILED: $c"; st=1; }
    done < <(cmds "$@")
    return $st
}
# yes_no VAR files: the true/false answer of VAR's command for those paths (false when unset).
yes_no() { local c="${!1:-}"; [ -n "$c" ] || { echo false; return; }; bash -c "$c" < "$2" 2>/dev/null || echo false; }
selected() {   # selected <files>: commands to run for those paths, or FULL
    if [ -z "${KIT_QA_SELECT:-}" ]; then echo FULL; else bash -c "$KIT_QA_SELECT" < "$1" || echo FULL; fi
}
fetch_main() { git -C "$QA_ROOT" fetch -q origin "$MAIN:refs/remotes/origin/$MAIN"; }
fetch_pr() { git -C "$QA_ROOT" fetch -q origin "pull/$1/head:refs/qa/pr/$1"; local_refs+=("refs/qa/pr/$1"); }
pr_files() { git -C "$QA_ROOT" diff --name-only "origin/$MAIN...refs/qa/pr/$1"; }
add_worktree() { git -C "$QA_ROOT" worktree add --detach "$1" "$2" >/dev/null; worktrees+=("$1"); }
new_tmp() { mktemp -d "${TMPDIR:-/tmp}/kit-qa-$1.XXXXXX"; }

# Log everything a real run prints (the dashboard tails these).
start_log() {
    local dir=${KIT_QA_LOG_DIR:-$HOME/qa-logs}
    mkdir -p "$dir"
    local log; log="$dir/qa-${KIT_PROJECT:-$(basename "$QA_ROOT")}-$1-$(date +%Y%m%d-%H%M%S).log"
    exec > >(tee -a "$log") 2>&1
    echo "log: $log"
}
verdict() {   # verdict <label> <status> [GO]
    local ok=PASS bad=FAIL
    [ -n "${3:-}" ] && { ok=GO; bad=NO-GO; }
    if [ "$2" -eq 0 ]; then echo "QA $1: $ok"; else echo "QA $1: $bad"; fi
    return "$2"
}

plan() {
    local what=$1; shift
    echo "PLAN $what"
    case "$what" in
        smoke) cmds KIT_QA_SMOKE kit/tools/check.sh | sed 's/^/- /'; echo "- dirty tree OK; light lane" ;;
        pr) echo "- fetch PR $1 into a temp worktree"
            echo "- ${KIT_QA_SELECT:+path-selected commands from KIT_QA_SELECT, else }full: $(cmds KIT_QA_FULL kit/tools/check.sh | paste -sd';' -)"
            [ -n "${KIT_QA_BUILDS:-}" ] && echo "- if app changed: $(cmds KIT_QA_BUILDS | paste -sd';' -)"
            echo "- light lane, heavy when full or builds run" ;;
        batch) echo "- fetch PRs $* and origin/$MAIN"
            echo "- each PR: full checks${KIT_QA_BUILDS:+ + builds if app changed}"
            echo "- stack on a temp branch off origin/$MAIN; stop on merge conflicts"
            [ -n "${KIT_QA_LAYOUT:-}" ] && echo "- if UI changed: $(cmds KIT_QA_LAYOUT | paste -sd';' -)"
            [ -n "${KIT_QA_AFTER_PASS:-}" ] && echo "- on pass, in each PR head: $(cmds KIT_QA_AFTER_PASS | paste -sd';' -)"
            echo "- heavy lane" ;;
        release) local var; var=$(release_var "$1"); cmds "$var" | sed 's/^/- /'; echo "- heavy lane; GO only if every step passes" ;;
    esac
    echo "- never merge, push to main, upload, or call paid APIs"
}

# checks_here <files>: the selected (or full) checks plus builds, in the current directory.
checks_here() {
    local files=$1 sel st=0
    sel=$(selected "$files")
    if [ "$sel" = FULL ]; then run_list KIT_QA_FULL kit/tools/check.sh || st=1
    elif [ -n "$sel" ]; then KIT_QA_SEL="$sel" run_list KIT_QA_SEL || st=1
    else echo "no checks selected for these paths"; fi
    if [ "$(yes_no KIT_QA_APP_CHANGED "$files")" = true ]; then run_list KIT_QA_BUILDS || st=1
    elif [ -n "${KIT_QA_BUILDS:-}" ]; then echo "skip builds (app unchanged)"; fi
    return $st
}

run_smoke() {
    [ $print_plan = 1 ] && { plan smoke; return 0; }
    need_lane light "$KIT_DIR/tools/qa.sh" smoke
    start_log smoke
    local st=0; run_list KIT_QA_SMOKE kit/tools/check.sh || st=1
    verdict smoke $st
}

run_pr() {
    local n=${1:-}
    is_pr_num "$n" && [ $# -eq 1 ] || { echo "qa.sh: pr takes one pull request number"; exit 2; }
    [ $print_plan = 1 ] && { plan pr "$n"; return 0; }
    fetch_main; fetch_pr "$n"
    local files; files=$(mktemp); pr_files "$n" > "$files"
    local lane=light
    { [ "$(selected "$files")" = FULL ] || [ "$(yes_no KIT_QA_APP_CHANGED "$files")" = true ]; } && lane=heavy
    [ "${KIT_LANE_RANK:-0}" -ge "$([ $lane = heavy ] && echo 2 || echo 1)" ] || rm -f "$files"
    need_lane "$lane" "$KIT_DIR/tools/qa.sh" pr "$n"
    start_log "pr$n"
    local wt st=0; wt=$(new_tmp "pr$n"); add_worktree "$wt" "refs/qa/pr/$n"
    ( cd "$wt" && checks_here "$files" ) || st=1
    rm -f "$files"
    verdict "pr $n" $st
}

run_batch() {
    [ $# -ge 1 ] || { echo "qa.sh: batch needs at least one pull request number"; exit 2; }
    local n; for n in "$@"; do is_pr_num "$n" || { echo "qa.sh: not a pull request number: $n"; exit 2; }; done
    [ $print_plan = 1 ] && { plan batch "$@"; return 0; }
    need_lane heavy "$KIT_DIR/tools/qa.sh" batch "$@"
    start_log "batch-$(IFS=-; echo "$*")"
    fetch_main; for n in "$@"; do fetch_pr "$n"; done
    local st=0 files wt all; all=$(mktemp)
    for n in "$@"; do
        echo "==== PR $n ===="
        files=$(mktemp); pr_files "$n" > "$files"; cat "$files" >> "$all"
        wt=$(new_tmp "pr$n"); add_worktree "$wt" "refs/qa/pr/$n"
        if ( cd "$wt" && KIT_QA_SELECT= checks_here "$files" ); then echo "PR $n: PASS"; else echo "PR $n: FAIL"; st=1; fi
        rm -f "$files"
    done
    local branch="qa/batch-$$" stack; stack=$(new_tmp stack)
    git -C "$QA_ROOT" worktree add -b "$branch" "$stack" "origin/$MAIN" >/dev/null
    worktrees+=("$stack"); local_branches+=("$branch")
    for n in "$@"; do
        if ! git -C "$stack" -c user.name="QA" -c user.email="qa@localhost" merge -q --no-edit "refs/qa/pr/$n" >/dev/null 2>&1; then
            echo "merge conflict stacking #$n:"; git -C "$stack" diff --name-only --diff-filter=U
            rm -f "$all"; verdict "batch $*" 1; return 1
        fi
    done
    echo "stack: clean"
    [ $st -eq 0 ] || { echo "skip layout check and after-pass (a PR failed)"; rm -f "$all"; verdict "batch $*" 1; return 1; }
    if [ "$(yes_no KIT_QA_UI_CHANGED "$all")" = true ]; then
        ( cd "$stack" && run_list KIT_QA_LAYOUT ) || { rm -f "$all"; verdict "batch $*" 1; return 1; }
    elif [ -n "${KIT_QA_LAYOUT:-}" ]; then echo "skip layout check (UI unchanged)"; fi
    rm -f "$all"
    if [ -n "${KIT_QA_AFTER_PASS:-}" ]; then
        for n in "$@"; do
            wt=$(new_tmp "pass$n"); add_worktree "$wt" "refs/qa/pr/$n"
            ( cd "$wt" && run_list KIT_QA_AFTER_PASS ) || st=1
        done
    fi
    verdict "batch $*" $st
}

run_release() {
    local name=${1:-} var
    [ -n "$name" ] || { echo "qa.sh: release needs a name (KIT_QA_RELEASE_<NAME> in kit.env)"; exit 2; }
    var=$(release_var "$name")
    [ -n "${!var:-}" ] || { echo "qa.sh: no $var in kit.env"; exit 2; }
    [ $print_plan = 1 ] && { plan release "$name"; return 0; }
    need_lane heavy "$KIT_DIR/tools/qa.sh" release "$name"
    start_log "release-$name"
    echo "tree: $(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo ' (dirty)')"
    local st=0; run_list "$var" || st=1
    verdict "$name" $st GO
}

case "$mode" in
    smoke) run_smoke "$@" ;;
    pr) run_pr "$@" ;;
    batch) run_batch "$@" ;;
    release) run_release "$@" ;;
    *) var=$(release_var "$mode")
       if [ -n "${!var:-}" ]; then run_release "$mode"; else usage; exit 2; fi ;;
esac
