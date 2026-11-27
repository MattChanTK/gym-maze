#!/bin/bash
# Prints a GitHub Actions runs-on JSON array. Uses the repo's self-hosted Mac
# (labels self-hosted, macOS, $KIT_RUNNER_LABEL) when one is online; otherwise a GitHub
# fallback. Default fallback is the xcode-27 image (Swift and TestFlight). Pass
# --fallback ubuntu-latest for Linux CI. A busy-but-online runner still counts,
# so the job queues there instead of spending GitHub minutes. API errors use the
# fallback. Never prints runner names.
#   kit/ios/pick_mac_runner.sh
#   kit/ios/pick_mac_runner.sh --fallback ubuntu-latest
#   kit/ios/pick_mac_runner.sh --runners-file runners.json
#   kit/ios/pick_mac_runner.sh --repo owner/name
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
[ -f kit.env ] && { set -a; source kit.env; set +a; }
runner_label=${KIT_RUNNER_LABEL:-mac}   # your self-hosted Mac runner's custom label (kit.env)
self_hosted="[\"self-hosted\",\"macOS\",\"$runner_label\"]"
fallback_label=xcode-27
file=
repo="${GITHUB_REPOSITORY:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --runners-file) file="$2"; shift 2 ;;
        --repo) repo="$2"; shift 2 ;;
        --fallback)
            [ -n "${2:-}" ] || { echo "usage: kit/ios/pick_mac_runner.sh [--fallback xcode-27|ubuntu-latest] [--runners-file file] [--repo owner/name]" >&2; exit 2; }
            fallback_label="$2"; shift 2 ;;
        *) echo "usage: kit/ios/pick_mac_runner.sh [--fallback xcode-27|ubuntu-latest] [--runners-file file] [--repo owner/name]" >&2; exit 2 ;;
    esac
done
case "$fallback_label" in
    xcode-27|ubuntu-latest) ;;
    *) echo "kit/ios/pick_mac_runner.sh: unknown --fallback $fallback_label (want xcode-27 or ubuntu-latest)" >&2; exit 2 ;;
esac
github=$(python3 -c 'import json,sys; print(json.dumps([sys.argv[1]]))' "$fallback_label")

if [ -z "$file" ]; then
    if [ -z "$repo" ]; then
        echo "No online self-hosted Mac runner (missing repo); using GitHub ${fallback_label}." >&2
        printf '%s\n' "$github"
        exit 0
    fi
    file=$(mktemp)
    trap 'rm -f "$file"' EXIT
    if ! gh api "repos/${repo}/actions/runners" > "$file" 2>/tmp/pick-mac-runner.err; then
        echo "No online self-hosted Mac runner (could not list runners); using GitHub ${fallback_label}." >&2
        printf '%s\n' "$github"
        exit 0
    fi
fi

python3 - "$file" "$self_hosted" "$github" "$fallback_label" "$runner_label" <<'PY'
import json, sys
path, self_hosted, github, fallback = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
required = {"self-hosted", "macOS", sys.argv[5]}
try:
    data = json.load(open(path))
except Exception:
    print(f"No online self-hosted Mac runner (bad runner list); using GitHub {fallback}.", file=sys.stderr)
    print(github)
    raise SystemExit(0)
for runner in data.get("runners") or []:
    if runner.get("status") != "online":
        continue
    labels = set()
    for label in runner.get("labels") or []:
        name = label.get("name") if isinstance(label, dict) else label
        if name:
            labels.add(name)
    if required <= labels:
        print("Online self-hosted Mac runner found.", file=sys.stderr)
        print(self_hosted)
        raise SystemExit(0)
print(f"No online self-hosted Mac runner; using GitHub {fallback}.", file=sys.stderr)
print(github)
PY
