#!/bin/bash
# Add a self-hosted Mac runner for a repo (personal accounts have no shared runners, so one per repo).
#   kit/tools/mac_runner.sh register owner/repo label   # runner in ~/actions-runners/<repo>, as a launchd service
# Workflows run on the Mac by default. A PR with the `ci-on-github` label runs on GitHub-hosted instead. Never prints tokens.
set -euo pipefail
gh=${KIT_GH:-gh}
usage() { sed -n '3p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

register() {        # repo label
    local repo=$1 label=$2 dir tarball token
    dir="$HOME/actions-runners/${repo#*/}"
    [ ! -e "$dir/.runner" ] || { echo "mac_runner.sh: $dir is already configured" >&2; return 1; }
    mkdir -p "$dir"
    tarball=$(ls -t "$HOME"/actions-runner*/actions-runner-osx-arm64-*.tar.gz "$HOME"/actions-runners/*/actions-runner-osx-arm64-*.tar.gz 2>/dev/null | head -1 || true)
    if [ -z "$tarball" ]; then
        "$gh" release download --repo actions/runner --pattern 'actions-runner-osx-arm64-*.tar.gz' --dir "$dir"
        tarball=$(ls "$dir"/actions-runner-osx-arm64-*.tar.gz | head -1)
    fi
    tar -xzf "$tarball" -C "$dir"
    token=$("$gh" api -X POST "repos/$repo/actions/runners/registration-token" --jq .token)
    (cd "$dir" && ./config.sh --unattended --url "https://github.com/$repo" --token "$token" \
        --name "$label-$(hostname -s)" --labels "$label" --replace >/dev/null)
    (cd "$dir" && ./svc.sh install >/dev/null && ./svc.sh start >/dev/null)
    # svc.sh start can load the agent without running it; kickstart makes it listen now.
    launchctl kickstart "gui/$(id -u)/$(basename "$(cat "$dir/.service")" .plist)"
    echo "Runner for $repo (label $label) installed in $dir and started."
}

case "${1:-}" in
    register) [ $# -eq 3 ] || usage; register "$2" "$3" ;;
    *) usage ;;
esac
