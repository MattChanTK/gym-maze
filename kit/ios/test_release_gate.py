#!/usr/bin/env python3
"""Behavior of kit/ios/release_gate.py: phone counts, same-SHA builds, build numbers."""
import os
os.environ["KIT_LAYOUT_MIN_PHONES_ALPHA"] = "4"
os.environ["KIT_LAYOUT_MIN_PHONES_BETA"] = "11"
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_gate as rg

count = 0
failures = 0


def test(name, pred):
    global count, failures
    count += 1
    if pred:
        return
    failures += 1
    print("FAIL", name)


ok, need, msg = rg.phone_count_ok("alpha", 4)
test("alpha accepts 4", ok and need == 4)
ok, need, msg = rg.phone_count_ok("alpha", 3)
test("alpha rejects 3", (not ok) and need == 4 and "3" in msg)
ok, _, msg = rg.phone_count_ok("beta", 11)
test("beta accepts 11", ok)
ok, need, msg = rg.phone_count_ok("beta", 10)
test("beta rejects 10", (not ok) and need == 11)
ok, need, msg = rg.phone_count_ok("gamma", 1)
test("a set with no minimum needs 1 phone", ok and need == 1)
ok, _, msg = rg.phone_count_ok("alpha", "four")
test("non-numeric count is rejected", not ok)
ok, _, _ = rg.phone_count_ok("alpha", 5)
test("alpha accepts more than 4", ok)

sha = "abc123"
ok, msg = rg.same_sha_ok(sha, sha, sha)
test("same SHA passes", ok and sha in msg)
ok, msg = rg.same_sha_ok(sha, "", sha)
test("missing Debug fails", not ok and "Debug" in msg)
ok, msg = rg.same_sha_ok(sha, sha, "")
test("missing Release fails", not ok and "Release" in msg)
ok, msg = rg.same_sha_ok(sha, "other", sha)
test("Debug on a different SHA fails", not ok)
ok, msg = rg.same_sha_ok("", sha, sha)
test("missing HEAD fails", not ok)

ok, msg = rg.build_number_ok(48, 47)
test("higher build passes", ok)
ok, msg = rg.build_number_ok(47, 47)
test("equal build fails", not ok and "47" in msg)
ok, msg = rg.build_number_ok(46, 47)
test("lower build fails", not ok)
ok, msg = rg.build_number_ok("x", 47)
test("non-numeric build fails", not ok)

here = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(here, "release_gate.py")


def run(*args):
    return subprocess.run([sys.executable, script, *args], capture_output=True, text=True)


r = run("phones", "alpha", "4")
test("CLI phones PASS exits 0", r.returncode == 0 and r.stdout.startswith("PASS"))
r = run("phones", "alpha", "2")
test("CLI phones FAIL exits 1", r.returncode == 1 and "PASS" not in r.stdout)
r = run("same-sha", "h", "h", "h")
test("CLI same-sha PASS exits 0", r.returncode == 0)
r = run("build-number", "48", "47")
test("CLI build-number PASS exits 0", r.returncode == 0)
r = run("build-number", "47", "47")
test("CLI build-number FAIL exits 1", r.returncode == 1)

# The command must not print secrets or personal values (nothing to hide here,
# but the output should stay a short status line).
r = run("phones", "beta", "11")
test("CLI output is one short line", r.stdout.count("\n") == 1 and "@" not in r.stdout)

print(f"{'PASS' if failures == 0 else 'FAIL'} {count} release_gate tests" + (f" ({failures} failed)" if failures else ""))
raise SystemExit(1 if failures else 0)
