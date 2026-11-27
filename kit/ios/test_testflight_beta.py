"""Tests for kit/ios/testflight_beta.py with made-up answers (no network, no real key). Run by kit/ios/test.sh;
prints failures and one summary line."""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("beta", os.path.join(os.path.dirname(__file__), "testflight_beta.py"))
beta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(beta)

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


groups = {"data": [{"id": "g1", "attributes": {"name": "Friends", "isInternalGroup": False}},
                   {"id": "g2", "attributes": {"name": "App Store Connect Users", "isInternalGroup": True}},
                   {"id": "g3", "attributes": {"name": "Public link"}}]}
test("only the external groups get the build", lambda: beta.external_groups(groups) == [("g1", "Friends"), ("g3", "Public link")])
test("no groups is an empty list", lambda: beta.external_groups({}) == [])
test("adding a build to a group", lambda: beta.add_to_group_request("g1", "b9") ==
     ("POST", "/v1/betaGroups/g1/relationships/builds", {"data": [{"type": "builds", "id": "b9"}]}))
test("submitting a build for beta review", lambda: beta.review_request("b9") == ("POST", "/v1/betaAppReviewSubmissions",
     {"data": {"type": "betaAppReviewSubmissions", "relationships": {"build": {"data": {"type": "builds", "id": "b9"}}}}}))

print(f"FAILED {failures} of {count} tool checks" if failures else f"PASS {count} tool checks")
raise SystemExit(1 if failures else 0)
