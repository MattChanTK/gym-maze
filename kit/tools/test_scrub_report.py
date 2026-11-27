#!/usr/bin/env python3
"""Behavior of tools/scrub_report.py: hides secrets, keeps ordinary text, CLI works."""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scrub_report as sr

count = 0
failures = 0


def test(name, pred):
    global count, failures
    count += 1
    if pred:
        return
    failures += 1
    print("FAIL", name)


# Fake values only. Never use real keys, UDIDs, team IDs, or emails.
FAKES = {
    "email": "uploaded by qa.tester+dev@example.com today",
    "sim UDID": "Booted: iPhone 17 (4F3C2B1A-1234-4ABC-9DEF-0123456789AB)",
    "device UDID": "device 00008110-001A2B3C4D5E801E connected",
    "team": "DEVELOPMENT_TEAM=AB12CD34EF",
    "signing team": "Apple Development: Test User (Z9Y8X7W6V5)",
    "openrouter": "sk-or-v1-" + "a" * 24,
    "github": "ghp_" + "B" * 20,
    "token": "API_TOKEN=" + "q" * 32,
}
HIDDEN = {
    "email": "qa.tester+dev@example.com",
    "sim UDID": "4F3C2B1A-1234-4ABC-9DEF-0123456789AB",
    "device UDID": "00008110-001A2B3C4D5E801E",
    "team": "AB12CD34EF",
    "signing team": "Z9Y8X7W6V5",
    "openrouter": "sk-or-v1-" + "a" * 24,
    "github": "ghp_" + "B" * 20,
    "token": "q" * 32,
}

for name, text in FAKES.items():
    out = sr.scrub(text, extra=[])
    test("hides " + name, HIDDEN[name] not in out)

test("keeps an ordinary pass line",
     sr.scrub("PASS 612 tests in 9 s, commit 60e62a4", extra=[]) == "PASS 612 tests in 9 s, commit 60e62a4")
test("hides an extra exact string",
     "MYDEVICE42" not in sr.scrub("run on MYDEVICE42", extra=["MYDEVICE42"]))

here = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(here, "scrub_report.py")
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "log.txt")
    open(path, "w").write("contact qa.tester+dev@example.com\n")
    r = subprocess.run([sys.executable, script, path], capture_output=True, text=True)
    test("CLI stdout is scrubbed", r.returncode == 0 and "qa.tester+dev@example.com" not in r.stdout and "[email]" in r.stdout)
    test("CLI left the file alone", "qa.tester+dev@example.com" in open(path).read())
    r = subprocess.run([sys.executable, script, "--in-place", path], capture_output=True, text=True)
    test("CLI --in-place rewrites the file", r.returncode == 0 and "qa.tester+dev@example.com" not in open(path).read() and "[email]" in open(path).read())
    # stdout of --in-place should not contain the secret either
    test("--in-place stdout has no secret", "qa.tester+dev@example.com" not in r.stdout and "qa.tester+dev@example.com" not in r.stderr)

print(f"{'PASS' if failures == 0 else 'FAIL'} {count} scrub_report tests" + (f" ({failures} failed)" if failures else ""))
raise SystemExit(1 if failures else 0)
