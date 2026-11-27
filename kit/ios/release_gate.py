#!/usr/bin/env python3
"""Release-gate checks for kit/ios/release_check.sh (steps of a KIT_QA_RELEASE_<NAME> gate).

    python3 kit/ios/release_gate.py phones <set> <count>       (minimum from KIT_LAYOUT_MIN_PHONES_<SET>, else 1)
    python3 kit/ios/release_gate.py same-sha <head> <debug> <release>
    python3 kit/ios/release_gate.py build-number <candidate> <last-uploaded>

Each command prints PASS or a one-line reason and exits 0 or 1.
The functions are also imported by tools/test_release_gate.py.
No network, no secrets.
"""
import os
import sys


def required_phones(track):
    try:
        return int(os.environ.get("KIT_LAYOUT_MIN_PHONES_" + track.upper().replace("-", "_"), "1"))
    except ValueError:
        return 1


def phone_count_ok(track, count):
    """(ok, required, message). count must be an int >= the track's minimum."""
    need = required_phones(track)
    try:
        n = int(count)
    except (TypeError, ValueError):
        return False, need, "release_gate: phone count must be a whole number"
    if n < need:
        return False, need, f"release_gate: {track} needs {need} phones, got {n}"
    return True, need, f"phones {n} >= {need} for {track}"


def same_sha_ok(head, debug, release):
    """Debug and Release must both have passed on the same SHA as HEAD."""
    if not head:
        return False, "release_gate: missing HEAD SHA"
    if not debug:
        return False, "release_gate: Debug has not passed on this SHA"
    if not release:
        return False, "release_gate: Release has not passed on this SHA"
    if debug != head or release != head:
        return False, "release_gate: Debug and Release must both pass on the same SHA as HEAD"
    return True, f"Debug and Release passed on {head}"


def build_number_ok(candidate, last_uploaded):
    """The build about to ship must be greater than the last TestFlight upload."""
    try:
        cand = int(candidate)
        last = int(last_uploaded)
    except (TypeError, ValueError):
        return False, "release_gate: build numbers must be whole numbers"
    if cand <= last:
        return False, f"release_gate: build {cand} must be greater than last uploaded {last}"
    return True, f"build {cand} > last uploaded {last}"


def _cmd(argv):
    if len(argv) >= 1 and argv[0] == "phones" and len(argv) == 3:
        ok, _, msg = phone_count_ok(argv[1], argv[2])
    elif len(argv) >= 1 and argv[0] == "same-sha" and len(argv) == 4:
        ok, msg = same_sha_ok(argv[1], argv[2], argv[3])
    elif len(argv) >= 1 and argv[0] == "build-number" and len(argv) == 3:
        ok, msg = build_number_ok(argv[1], argv[2])
    else:
        sys.exit(__doc__)
    print("PASS " + msg if ok else msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cmd(sys.argv[1:]))
