#!/bin/bash
# Shared by the kit's tools: run from the project's repo root, load kit.env (tracked, per-repo settings) and the git-ignored .env.local.
# KIT_DIR is the vendored kit folder (kit/ in a project, or the kit/ folder of dev-kit itself).
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$(git -C "$KIT_DIR" rev-parse --show-toplevel)"

# A new worktree has no git-ignored files: copy the setup files from the main checkout. One already there is kept.
main_checkout=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)")
for setup_file in CLAUDE.local.md .env.local; do
    if [ ! -f "$setup_file" ] && [ -f "$main_checkout/$setup_file" ]; then cp -p "$main_checkout/$setup_file" "$setup_file"; fi
done

[ -z "${KIT_NO_ENV:-}" ] && [ -f kit.env ] && { set -a; source kit.env; set +a; }
[ -z "${KIT_NO_ENV:-}" ] && [ -f .env.local ] && { set -a; source .env.local; set +a; }

# load_local KEY…: export each KEY=value line from the git-ignored CLAUDE.local.md; exit if one is missing.
load_local() {
    for key in "$@"; do
        local value; value=$(sed -nE "s/^$key=([^ #]+).*/\1/p" CLAUDE.local.md 2>/dev/null | head -1)
        [ -n "$value" ] || { echo "Missing $key in CLAUDE.local.md"; exit 1; }
        export "$key=$value"
    done
}

# Every commit made by these tools ends with this line (set KIT_ATTRIBUTION in kit.env).
ATTRIBUTION="${KIT_ATTRIBUTION:-Co-Authored-By: Claude <noreply@anthropic.com>}"

# need_lane heavy|light <self> [args…]
# Re-exec <self> under tools/lane.sh unless this process already holds a
# slot of that rank or higher. Call after parsing args, before any work.
need_lane() {
    local lane=$1 self=$2
    shift 2
    local rank=0
    case "$lane" in
        light) rank=1 ;;
        heavy) rank=2 ;;
        *) echo "need_lane: unknown lane $lane" >&2; exit 2 ;;
    esac
    if [ "${KIT_LANE_RANK:-0}" -ge "$rank" ]; then
        return 0
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    exec "$here/lane.sh" "$lane" -- "$self" "$@"
}

# refuse_protected_checkout: fail if this worktree is the protected main checkout (KIT_PROTECTED_CHECKOUT in kit.env).
# Test, QA, and upload scripts call this so those runs go through a clone or worktree. Unset: no guard.
refuse_protected_checkout() {
    local here protected resolved
    here=$(pwd -P)
    protected=${KIT_PROTECTED_CHECKOUT:-}
    [ -n "$protected" ] && [ -d "$protected" ] || return 0
    resolved=$(cd "$protected" && pwd -P) || return 0
    if [ "$here" = "$resolved" ]; then
        echo "This is the protected checkout ($resolved)." >&2
        echo "Run QA, tests, and uploads from a clone or a worktree:" >&2
        echo "  kit/tools/worktree.sh create <name>" >&2
        echo "Or change KIT_PROTECTED_CHECKOUT in kit.env." >&2
        return 1
    fi
    return 0
}
