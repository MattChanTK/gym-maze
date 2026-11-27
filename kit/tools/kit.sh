#!/bin/bash
# The vendored dev-kit in this repo: kit/ at a pinned tag (kit/VERSION). Projects upgrade when they choose.
#   kit/tools/kit.sh status                 pinned version, latest tag, local edits to kit/
#   kit/tools/kit.sh update [tag]           replace kit/ with that tag (default: latest). Refuses if kit/ has local edits.
#   kit/tools/kit.sh contribute <branch>    put this repo's kit/ edits on a dev-kit branch and push it (ask first), then open a PR there
# Remote: KIT_REMOTE in kit.env (default github.com/MattChanTK/dev-kit). Never commits in this repo: commit kit/ yourself.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
remote=${KIT_REMOTE:-https://github.com/MattChanTK/dev-kit.git}
pinned=$(cat "$KIT_DIR/VERSION" 2>/dev/null || echo none)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/kit.XXXXXX"); trap 'rm -rf "$tmp"' EXIT

latest_tag() { git ls-remote --tags --refs "$remote" 'v*' | sed 's|.*refs/tags/||' | sort -V | tail -1; }
fetch_tag() { git clone -q --depth 1 --branch "$1" "$remote" "$tmp/dk" 2>/dev/null || { echo "Can't fetch $1 from $remote"; exit 1; }; }
# local_edits: files in kit/ that differ from the pinned tag (needs the tag fetched into $tmp/dk).
# copy_kit <from> <to>: replace folder <to> with <from>, without __pycache__.
copy_kit() { rm -rf "$2"; cp -R "$1" "$2"; find "$2" -name __pycache__ -prune -exec rm -rf {} +; }
local_edits() { diff -rq --exclude=__pycache__ --exclude=VERSION "$tmp/dk/kit" "$KIT_DIR" | sed "s|$tmp/dk/||; s|$(pwd)/||" || true; }

case "${1:-status}" in
status)
    echo "kit $pinned (latest $(latest_tag))"
    fetch_tag "$pinned"; edits=$(local_edits)
    [ -z "$edits" ] && echo "no local edits" || { echo "local edits (send with kit.sh contribute):"; echo "$edits"; }
    ;;
update)
    tag=${2:-$(latest_tag)}
    [ "$tag" != "$pinned" ] || { echo "already on $tag"; exit 0; }
    fetch_tag "$pinned"; edits=$(local_edits)
    if [ -n "$edits" ] && [ "${3:-}" != --force ]; then
        echo "kit/ has local edits; contribute them first (kit.sh contribute <branch>) or pass --force to drop them:"; echo "$edits"; exit 1
    fi
    rm -rf "$tmp/dk"; fetch_tag "$tag"
    copy_kit "$tmp/dk/kit" "$KIT_DIR"; echo "$tag" > "$KIT_DIR/VERSION"
    echo "kit $pinned -> $tag. What changed:"
    sed -n "/^## $tag/,/^## $pinned/p" "$tmp/dk/CHANGELOG.md" | sed '$d'
    echo "Run kit/tools/check.sh, then commit kit/."
    ;;
contribute)
    branch=${2:?usage: kit.sh contribute <branch>}
    fetch_tag "$pinned"; edits=$(local_edits)
    [ -n "$edits" ] || { echo "no local edits to contribute"; exit 0; }
    keep=$(cat "$tmp/dk/kit/VERSION"); copy_kit "$KIT_DIR" "$tmp/dk/kit"; echo "$keep" > "$tmp/dk/kit/VERSION"
    git -C "$tmp/dk" checkout -q -b "$branch"
    git -C "$tmp/dk" add -A kit
    # Scan with this repo's secret values (dev-kit git-ignores these files).
    for f in CLAUDE.local.md .env.local; do [ -f "$f" ] && cp "$f" "$tmp/dk/"; done
    (cd "$tmp/dk" && python3 kit/tools/scan_secrets.py) || { echo "Not pushed: secret found"; exit 1; }
    git -C "$tmp/dk" commit -q -m "From $(basename "$(pwd)") on $pinned: kit edits" -m "$ATTRIBUTION"
    git -C "$tmp/dk" push -q -u origin "$branch"
    echo "Pushed $branch to dev-kit (based on $pinned). Open a PR there; after merge and a new tag, run kit.sh update."
    ;;
*) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
