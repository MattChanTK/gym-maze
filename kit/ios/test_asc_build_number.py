"""Tests for kit/ios/asc_build_number.py with made-up answers (no network, no real key). Run by kit/ios/test_py.sh;
prints failures and one summary line."""
import os
import sys
import tempfile
import urllib.parse

sys.path.insert(0, os.path.dirname(__file__))
import asc_build_number as abn  # noqa: E402

failures, count = 0, 0


def test(name, check):
    global failures, count
    count += 1
    try:
        if check() is not True:
            failures += 1
            print(f"FAIL {name}")
    except Exception as error:
        failures += 1
        print(f"FAIL {name}: {error!r}")


def builds(*versions):
    return {"data": [{"id": f"b{i}", "attributes": {"version": v}} for i, v in enumerate(versions)]}


test("the highest number wins, compared as numbers not text", lambda: abn.highest(builds("9", "44", "43")) == 44)
test("the project's number is a floor", lambda: abn.highest(builds("12"), floor=43) == 43)
test("no builds gives the floor", lambda: abn.highest({}, floor=43) == 43)
test("odd versions are skipped", lambda: abn.highest(builds("1.2", None, "", "40")) == 40)
test("the query asks for this app's newest uploads", lambda: (lambda q: q["filter[app]"] == ["app1"]
     and q["sort"] == ["-uploadedDate"] and q["limit"] == [str(abn.PAGE)])(
     urllib.parse.parse_qs(abn.builds_query("app1").split("?", 1)[1])))


def reads_the_floor():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "project.pbxproj")
        open(path, "w").write("CURRENT_PROJECT_VERSION = 8;\nCURRENT_PROJECT_VERSION = 8;\n")
        return abn.project_floor(path) == 8


test("the floor comes from the project", reads_the_floor)


def missing_key_stops():
    try:
        abn.credentials({"ASC_KEY_ID": "k"})
    except SystemExit as stopped:
        return "Missing" in str(stopped)
    return False


test("a missing key stops before any call", missing_key_stops)
test("BUNDLE stands in for APPLE_BUNDLE_ID, ASC_KEY_PATH for the key folder", lambda: abn.credentials(
     {"ASC_KEY_ID": "k", "ASC_ISSUER_ID": "i", "BUNDLE": "b", "ASC_KEY_PATH": "/tmp/x.p8"}) == ("k", "i", "/tmp/x.p8", "b"))


def asks_apple(answers):
    calls, real = [], (abn.asc.token, abn.asc.get)
    abn.asc.token = lambda *a: "t"
    abn.asc.get = lambda url, auth: (calls.append(url), answers.pop(0))[1]
    try:
        return abn.latest_uploaded({"ASC_KEY_ID": "k", "ASC_ISSUER_ID": "i", "APPLE_BUNDLE_ID": "b"}, floor=43), calls
    finally:
        abn.asc.token, abn.asc.get = real


test("finds the app, then its highest build", lambda: asks_apple([{"data": [{"id": "app1"}]}, builds("45", "44")])[0] == 45)

print(f"FAILED {failures} of {count} tool checks" if failures else f"PASS {count} tool checks")
raise SystemExit(1 if failures else 0)
