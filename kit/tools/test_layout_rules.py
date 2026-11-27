"""Tests for kit/tools/layout_rules.py with made-up frames."""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout_rules as lr  # noqa: E402

count = failures = 0


def test(name, fn):
    global count, failures
    count += 1
    try:
        ok = fn()
    except Exception as e:  # noqa: BLE001
        ok = False
        name += f": {e!r}"
    if not ok:
        failures += 1
        print("FAIL", name)


def f(i, x, y, w, h):
    return {"id": i, "x": x, "y": y, "width": w, "height": h}


# Screen 400x800; preview fills it. A pill flush right, a chip flush left, a centered card.
DUMP = {"token": "T", "frames": [
    f("preview", 0, 0, 400, 800), f("pill.help", 356, 100, 44, 44), f("pill.more", 356, 150, 44, 30),
    f("chip.one", 0, 300, 120, 40), f("card", 10, 600, 380, 100), f("picture", 10, 706, 100, 60)]}
RULES = {"container": "preview", "states": {"home": {
    "required": ["pill.*", "card"], "hidden": ["sheet"],
    "flush": [{"ids": ["pill.*"], "edge": "right"}, {"ids": ["chip.*"], "edge": "left"}],
    "no_overlap": {}, "equal_inset": [{"id": "card"}],
    "gap": [{"above": "card", "below": "picture", "gap": 6}],
    "min_tap": {"ids": ["pill.*"], "size": 44}}}}


def moved(i, **kw):
    d = json.loads(json.dumps(DUMP))
    for fr in d["frames"]:
        if fr["id"] == i:
            fr.update(kw)
    return d


test("a good layout passes, with a tap warning for the short pill", lambda: (
    lambda r: r[0] == [] and len(r[1]) == 1 and "pill.more" in r[1][0])(lr.check(DUMP, RULES, "home")))
test("a pill off the right edge fails", lambda: any("pill.help" in p and "right edge" in p
     for p in lr.check(moved("pill.help", x=340), RULES, "home")[0]))
test("a missing required prefix fails", lambda: "pill.* missing" in lr.check(
    {"frames": [fr for fr in DUMP["frames"] if not fr["id"].startswith("pill.")]}, RULES, "home")[0])
test("a hidden id on screen fails", lambda: "sheet should be hidden" in lr.check(
    {"frames": DUMP["frames"] + [f("sheet", 0, 0, 10, 10)]}, RULES, "home")[0])
test("overlap fails", lambda: any("overlaps" in p for p in lr.check(moved("chip.one", x=0, y=590), RULES, "home")[0]))
test("unequal inset fails", lambda: any("inset" in p for p in lr.check(moved("card", x=30), RULES, "home")[0]))
test("wrong gap fails", lambda: any("under card" in p for p in lr.check(moved("picture", y=730), RULES, "home")[0]))
test("unknown state fails", lambda: lr.check(DUMP, RULES, "nope")[0] == ["no rules for state 'nope'"])

# Turned 90 degrees clockwise (landscapeLeft): screen 800x400, the upright right edge is the screen's bottom.
LAND = {"frames": [f("preview", 0, 0, 800, 400), f("pill.help", 100, 356, 44, 44)]}
SIDE = {"container": "preview", "states": {"s": {"flush": [{"ids": ["pill.*"], "edge": "right"}]}}}
test("landscapeLeft: flush with the screen's bottom is the right edge", lambda: lr.check(LAND, SIDE, "s", "landscapeLeft")[0] == [])
test("portrait reading of the same frames fails", lambda: lr.check(LAND, SIDE, "s", "portrait")[0] != [])
RIGHT = {"frames": [f("preview", 0, 0, 800, 400), f("pill.help", 100, 0, 44, 44)]}
test("landscapeRight: the right edge is the screen's top", lambda: lr.check(RIGHT, SIDE, "s", "landscapeRight")[0] == [])
UPSIDE = {"frames": [f("preview", 0, 0, 400, 800), f("pill.help", 0, 100, 44, 44)]}
test("upside down: the right edge is the screen's left", lambda: lr.check(UPSIDE, SIDE, "s", "portraitUpsideDown")[0] == [])

test("a stale token is not fresh", lambda: not lr.fresh(DUMP, "OTHER") and lr.fresh(DUMP, "T"))


def cli():
    with tempfile.TemporaryDirectory() as d:
        dp, rp = os.path.join(d, "dump.json"), os.path.join(d, "rules.json")
        json.dump(DUMP, open(dp, "w")); json.dump(RULES, open(rp, "w"))
        here = os.path.dirname(os.path.abspath(__file__))
        ok = subprocess.run([sys.executable, os.path.join(here, "layout_rules.py"), "--rules", rp, "--dump", dp,
                             "--state", "home", "--token", "T", "--where", "X"], capture_output=True, text=True)
        stale = subprocess.run([sys.executable, os.path.join(here, "layout_rules.py"), "--rules", rp, "--dump", dp,
                                "--state", "home", "--token", "Z"], capture_output=True, text=True)
        return ok.returncode == 0 and "PASS X" in ok.stdout and stale.returncode == 1 and "stale" in stale.stdout


test("command line: PASS line, and a stale dump fails", cli)
print(f"FAILED {failures} of {count} layout_rules tests" if failures else f"PASS {count} layout_rules tests")
sys.exit(1 if failures else 0)
