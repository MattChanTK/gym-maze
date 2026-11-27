#!/usr/bin/env python3
"""Generic layout check: a screen's element frames (a "dump") against a rules file. Works for any UI that can write
its element frames: an iOS DEBUG build (kit/ios/check_layouts.sh), a web page via Playwright, anything.

Dump (JSON):  {"token": "T", "frames": [{"id": "pill.help", "x": 0, "y": 0, "width": 44, "height": 44}, ...]}
Rules (JSON): {"container": "preview",              # the frame edges are measured against (default: the screen)
               "states": {"<state>": {
                   "required": ["a", "pill.*"],      # must be on screen ("x.*" = every id starting "x.", at least one)
                   "hidden":   ["b"],                # must not be on screen
                   "flush":    [{"ids": ["pill.*"], "edge": "right", "tol": 4}],   # edge: left/right/top/bottom
                   "no_overlap": {"except": ["preview"], "tol": 1},
                   "equal_inset": [{"id": "card", "axis": "x", "tol": 2}],        # same leftover both sides
                   "gap": [{"above": "card", "below": "picture", "gap": 6, "tol": 4}],
                   "match": [{"id": "sheet", "tol": 1}],                          # same frame as the container
                   "min_tap": {"ids": ["pill.*"], "size": 44}                      # warning only
               }}}
Edges are upright edges: with --orientation landscapeLeft the UI is turned 90 degrees clockwise, so its right edge
is the screen's bottom (landscapeRight: the top; portraitUpsideDown: the left).

    python3 kit/tools/layout_rules.py --rules layout_rules.json --dump dump.json --state home \
        [--orientation portrait] [--token T] [--where "iPhone SE / large / portrait / home"]
Prints PASS <where>, or FAIL <where>: <problem> per problem (exit 1). WARN lines never fail.
"""
import argparse
import fnmatch
import json
import sys

ORIENTATIONS = ("portrait", "landscapeLeft", "landscapeRight", "portraitUpsideDown")


def box(frame):
    """(x0, y0, x1, y1), or None when the frame isn't numbers."""
    try:
        x, y, w, h = (float(frame[k]) for k in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError):
        return None
    return (x, y, x + w, y + h)


def frames_by_id(dump):
    raw = dump.get("frames") if isinstance(dump, dict) else None
    found = {}
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and box(item) is not None:
            found[item["id"]] = box(item)
    return found


def upright(item, container, orientation):
    """The item's box in the upright UI's coordinates, inside the container: (x0, y0, x1, y1, width, height)."""
    cx0, cy0, cx1, cy1 = container
    x0, y0, x1, y1 = item
    if orientation == "landscapeLeft":
        return (y0 - cy0, cx1 - x1, y1 - cy0, cx1 - x0, cy1 - cy0, cx1 - cx0)
    if orientation == "landscapeRight":
        return (cy1 - y1, x0 - cx0, cy1 - y0, x1 - cx0, cy1 - cy0, cx1 - cx0)
    if orientation == "portraitUpsideDown":
        return (cx1 - x1, cy1 - y1, cx1 - x0, cy1 - y0, cx1 - cx0, cy1 - cy0)
    return (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0, cx1 - cx0, cy1 - cy0)


def edge_gap(u, edge):
    x0, y0, x1, y1, w, h = u
    return {"left": x0, "right": w - x1, "top": y0, "bottom": h - y1}[edge]


def select(patterns, found):
    """Ids matching the patterns, in order. A pattern ending in .* matches every id with that prefix."""
    out = []
    for p in patterns:
        out += [i for i in sorted(found) if fnmatch.fnmatchcase(i, p) and i not in out]
    return out


def screen_box(found):
    xs = [b[0] for b in found.values()] + [b[2] for b in found.values()]
    ys = [b[1] for b in found.values()] + [b[3] for b in found.values()]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)


def check(dump, rules, state, orientation="portrait"):
    """(problems, warnings) for one dump."""
    found = frames_by_id(dump)
    if state not in rules.get("states", {}):
        return [f"no rules for state '{state}'"], []
    r = rules["states"][state]
    cid = rules.get("container")
    if cid and cid not in found:
        return [f"container '{cid}' missing"], []
    container = found[cid] if cid else screen_box(found)
    U = {i: upright(b, container, orientation) for i, b in found.items()}
    problems, warnings = [], []
    for p in r.get("required", []):
        if not select([p], found):
            problems.append(f"{p} missing")
    for i in select(r.get("hidden", []), found):
        problems.append(f"{i} should be hidden")
    for rule in r.get("flush", []):
        tol = rule.get("tol", 4)
        for i in select(rule["ids"], found):
            gap = edge_gap(U[i], rule["edge"])
            if abs(gap) > tol:
                problems.append(f"{i} is {gap:.0f}pt from the {rule['edge']} edge, expected flush (±{tol:g})")
    no = r.get("no_overlap")
    if no is not None:
        skip = set(select(no.get("except", []), found)) | ({cid} if cid else set())
        ids = [i for i in sorted(found) if i not in skip]
        tol = no.get("tol", 1)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                p, q = found[ids[a]], found[ids[b]]
                dx, dy = min(p[2], q[2]) - max(p[0], q[0]), min(p[3], q[3]) - max(p[1], q[1])
                if dx > tol and dy > tol:
                    problems.append(f"{ids[a]} overlaps {ids[b]} by {dx:.0f}x{dy:.0f}pt")
    for rule in r.get("equal_inset", []):
        i, tol = rule["id"], rule.get("tol", 2)
        if i not in U:
            problems.append(f"{i} missing"); continue
        a, b = ("left", "right") if rule.get("axis", "x") == "x" else ("top", "bottom")
        ga, gb = edge_gap(U[i], a), edge_gap(U[i], b)
        if abs(ga - gb) > tol:
            problems.append(f"{i} inset {a} {ga:.0f}pt vs {b} {gb:.0f}pt, expected equal (±{tol:g})")
    for rule in r.get("gap", []):
        a, b = rule["above"], rule["below"]
        if a in U and b in U:
            gap, want, tol = U[b][1] - U[a][3], rule.get("gap", 0), rule.get("tol", 4)
            if abs(gap - want) > tol:
                problems.append(f"{b} is {gap:.0f}pt under {a}, expected {want:g} (±{tol:g})")
    for rule in r.get("match", []):
        i, tol = rule["id"], rule.get("tol", 1)
        if i not in found:
            problems.append(f"{i} missing")
        elif max(abs(m - n) for m, n in zip(found[i], container)) > tol:
            problems.append(f"{i} does not match the container's frame (±{tol:g})")
    tap = r.get("min_tap")
    if tap:
        for i in select(tap.get("ids", []), found):
            x0, y0, x1, y1 = found[i]
            if min(x1 - x0, y1 - y0) < tap.get("size", 44):
                warnings.append(f"{i} is {x1 - x0:.0f}x{y1 - y0:.0f}pt, under the {tap.get('size', 44)}pt tap target")
    return problems, warnings


def fresh(dump, token):
    """True when the dump came from this launch (its token matches) and has frames."""
    return isinstance(dump, dict) and (token is None or dump.get("token") == token) and bool(frames_by_id(dump))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rules", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--orientation", default="portrait", choices=ORIENTATIONS)
    ap.add_argument("--token")
    ap.add_argument("--where", default="")
    a = ap.parse_args(argv)
    where = a.where or a.state
    try:
        dump = json.load(open(a.dump, encoding="utf-8"))
    except (OSError, ValueError):
        print(f"FAIL {where}: no layout dump"); return 1
    if not fresh(dump, a.token):
        print(f"FAIL {where}: stale or empty layout dump"); return 1
    problems, warnings = check(dump, json.load(open(a.rules, encoding="utf-8")), a.state, a.orientation)
    for w in warnings:
        print(f"WARN {where}: {w}")
    for p in problems:
        print(f"FAIL {where}: {p}")
    if not problems:
        print(f"PASS {where}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
