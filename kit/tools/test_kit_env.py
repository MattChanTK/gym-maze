"""Tests for kit/tools/kit_env.py: kit.env is read by bash, only KIT_* names are taken, the environment wins, KIT_NO_ENV turns it off."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kit_env import load_kit_env  # noqa: E402

failures, count = 0, 0


def check(name, got, want):
    global failures, count
    count += 1
    if got != want:
        failures += 1
        print("FAIL %s: got %r want %r" % (name, got, want))


for name in ("KIT_NO_ENV", "KIT_FROM_FILE", "KIT_ALREADY_SET", "NOT_A_KIT_NAME"):
    os.environ.pop(name, None)
with tempfile.TemporaryDirectory() as project:
    with open(os.path.join(project, "kit.env"), "w") as f:
        f.write('# comment\nKIT_FROM_FILE="$HOME/x y"\nKIT_ALREADY_SET="file"\nNOT_A_KIT_NAME="no"\n')
    os.environ["KIT_NO_ENV"] = "1"
    load_kit_env(project)
    check("KIT_NO_ENV skips", os.environ.get("KIT_FROM_FILE"), None)
    del os.environ["KIT_NO_ENV"]
    os.environ["KIT_ALREADY_SET"] = "env"
    load_kit_env(project)
    check("expands like bash", os.environ.get("KIT_FROM_FILE"), os.path.expanduser("~") + "/x y")
    check("environment wins", os.environ.get("KIT_ALREADY_SET"), "env")
    check("only KIT_ names", os.environ.get("NOT_A_KIT_NAME"), None)
with tempfile.TemporaryDirectory() as empty:
    load_kit_env(empty)
    check("no kit.env is fine", True, True)

print("FAIL %d of %d kit_env checks" % (failures, count) if failures else "PASS %d kit_env checks" % count)
sys.exit(1 if failures else 0)
