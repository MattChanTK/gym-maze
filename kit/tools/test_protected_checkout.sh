#!/bin/bash
# QA, test, and upload scripts refuse the protected checkout.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
fails=0
checks=0
tmp=$(mktemp -d "${TMPDIR:-/tmp}/kit-protected-test.XXXXXX")
tmp=$(cd "$tmp" && pwd -P)
trap 'rm -rf "$tmp"' EXIT

ok() { checks=$((checks + 1)); }
fail() { checks=$((checks + 1)); echo "FAIL $1"; fails=$((fails + 1)); }

# A fake repo that sources the real lib.sh after we rewrite only the cd.
# Easier: copy lib.sh into a fake tree so its cd lands in the fake tree,
# then point KIT_PROTECTED_CHECKOUT at it.
mkdir -p "$tmp/protected/tools" "$tmp/other/tools"
# Minimal lib.sh: the real one cds to dirname/.. of itself.
# Copy the real refuse function by sourcing a stub that cds to the given root.
cat > "$tmp/protected/tools/lib.sh" << 'LIB'
#!/bin/bash
cd "$(dirname "$0")/.."
# Pull refuse_protected_checkout from the real lib after the cd, by defining it here
# the same way (keep this test honest by extracting the function from the repo).
LIB
# Append the function from the real lib.sh
sed -n '/^# refuse_protected_checkout/,/^}$/p' "$root/tools/lib.sh" >> "$tmp/protected/tools/lib.sh"
cp "$tmp/protected/tools/lib.sh" "$tmp/other/tools/lib.sh"

cat > "$tmp/run.sh" << 'RUN'
#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
refuse_protected_checkout
echo allowed
RUN
chmod +x "$tmp/run.sh"
cp "$tmp/run.sh" "$tmp/protected/tools/run.sh"
cp "$tmp/run.sh" "$tmp/other/tools/run.sh"
chmod +x "$tmp/protected/tools/run.sh" "$tmp/other/tools/run.sh"

export KIT_PROTECTED_CHECKOUT="$tmp/protected"
if "$tmp/protected/tools/run.sh" >/dev/null 2>"$tmp/err.txt"; then
    fail "protected checkout should be refused"
else
    if grep -q "protected checkout" "$tmp/err.txt" && grep -q "worktree.sh create" "$tmp/err.txt"; then
        ok
    else
        fail "refusal message was unclear: $(cat "$tmp/err.txt")"
    fi
    if grep -qiE 'token|password|@gmail|AKIA|udid' "$tmp/err.txt"; then
        fail "refusal message leaked a secret-like string"
    else
        ok
    fi
fi

if out=$("$tmp/other/tools/run.sh" 2>"$tmp/err2.txt"); then
    [ "$out" = allowed ] && ok || fail "other checkout printed: $out"
else
    fail "a different checkout should be allowed: $(cat "$tmp/err2.txt")"
fi

# Missing protected path is allowed (CI, other machines).
export KIT_PROTECTED_CHECKOUT="$tmp/does-not-exist"
if out=$("$tmp/protected/tools/run.sh" 2>"$tmp/err3.txt"); then
    [ "$out" = allowed ] && ok || fail "missing protected path printed: $out"
else
    fail "missing protected path should be allowed: $(cat "$tmp/err3.txt")"
fi

[ $fails -eq 0 ] && echo "PASS $checks protected-checkout checks" || { echo "FAIL $fails of $checks protected-checkout checks"; exit 1; }
