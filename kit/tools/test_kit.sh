#!/bin/bash
# kit.sh update/status/contribute and install.sh: the kit/ios add-on goes only to projects that have it (or install --ios).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)   # the kit/ folder under test
fails=0
checks=0
tmp=$(mktemp -d "${TMPDIR:-/tmp}/kit-sh-test.XXXXXX")
tmp=$(cd "$tmp" && pwd -P)
trap 'rm -rf "$tmp"' EXIT
ok() { checks=$((checks + 1)); }
fail() { checks=$((checks + 1)); echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then ok; else fail "$1"; fi; }
g() { git -c user.email=t@example.com -c user.name=t -c init.defaultBranch=main "$@"; }

# Fake dev-kit with tags v9.0.0 and v9.0.1 (v9.0.1 changes a tool and an ios file).
dk=$tmp/dev-kit
mkdir -p "$dk"; cp -R "$root" "$dk/kit"; find "$dk/kit" -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$dk/kit/ios"; echo one > "$dk/kit/ios/addon.txt"; echo v9.0.0 > "$dk/kit/VERSION"
printf '# Changelog\n\n## v9.0.1\n- two\n\n## v9.0.0\n- one\n' > "$dk/CHANGELOG.md"
g -C "$dk" init -q; g -C "$dk" add -A; g -C "$dk" commit -qm one; g -C "$dk" tag v9.0.0
echo two > "$dk/kit/ios/addon.txt"; echo "# two" >> "$dk/kit/tools/lane.sh"; echo v9.0.1 > "$dk/kit/VERSION"
g -C "$dk" commit -qam two; g -C "$dk" tag v9.0.1

# project <name> <with_ios>: a repo with kit/ at v9.0.0.
project() {
    local p=$tmp/$1; mkdir -p "$p"; g -C "$p" init -q
    mkdir -p "$tmp/x-$1"; g -C "$dk" archive v9.0.0 kit | tar -x -C "$tmp/x-$1"
    cp -R "$tmp/x-$1/kit" "$p/kit"; [ "$2" = 1 ] || rm -rf "$p/kit/ios"
    g -C "$p" add -A; g -C "$p" commit -qm init
}
kit() { (cd "$tmp/$1" && env KIT_NO_ENV=1 KIT_REMOTE="$dk" kit/tools/kit.sh "${@:2}"); }

project plain 0
out=$(kit plain status)
check "no ios: status reports no local edits" '[[ $out == *"no local edits"* ]]'
kit plain update v9.0.1 > /dev/null
check "no ios: update applies" 'grep -q "# two" "$tmp/plain/kit/tools/lane.sh" && [ "$(cat "$tmp/plain/kit/VERSION")" = v9.0.1 ]'
check "no ios: update does not add kit/ios" '[ ! -e "$tmp/plain/kit/ios" ]'

project mobile 1
kit mobile update v9.0.1 > /dev/null
check "with ios: update refreshes kit/ios" '[ "$(cat "$tmp/mobile/kit/ios/addon.txt")" = two ]'

# contribute from a project without ios keeps dev-kit's kit/ios.
project edit 0
touch "$tmp/edit/CLAUDE.local.md"
echo "# local" >> "$tmp/edit/kit/tools/lane.sh"
g -C "$dk" config receive.denyCurrentBranch ignore
kit edit contribute fix-x > /dev/null
check "contribute: branch has the edit" 'g -C "$dk" show fix-x:kit/tools/lane.sh | grep -q "# local"'
check "contribute: branch keeps kit/ios" 'g -C "$dk" cat-file -e fix-x:kit/ios/addon.txt'

# install.sh (only inside dev-kit itself).
if [ -f "$root/../install.sh" ]; then
    src=$tmp/src; g clone -q "$dk" "$src"; g -C "$src" checkout -q v9.0.0
    cp "$root/../install.sh" "$src/"; mkdir -p "$src/templates/core" "$src/templates/ios"
    echo 'KIT_PROJECT="myproject"' > "$src/templates/core/kit.env.example"; echo '## iOS (kit/ios/)' > "$src/templates/ios/CLAUDE.ios.md"
    touch "$src/templates/core/CLAUDE.md"
    mkdir -p "$tmp/i1" "$tmp/i2"; g -C "$tmp/i1" init -q; g -C "$tmp/i2" init -q
    "$src/install.sh" "$tmp/i1" > /dev/null
    check "install: no kit/ios without --ios" '[ -d "$tmp/i1/kit/tools" ] && [ ! -e "$tmp/i1/kit/ios" ]'
    "$src/install.sh" "$tmp/i2" --ios > /dev/null
    check "install --ios: has kit/ios" '[ -f "$tmp/i2/kit/ios/addon.txt" ]'
fi

if [ $fails -eq 0 ]; then echo "PASS $checks kit.sh checks"; else echo "FAIL $fails of $checks kit.sh checks"; exit 1; fi
