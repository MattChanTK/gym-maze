"""Tests for kit/ios/pull_testflight_feedback.py with made-up data (no network, no real key). Run by kit/ios/test.sh;
prints failures and one summary line."""
import base64
import importlib.util
import json
import os
import subprocess
import tempfile

spec = importlib.util.spec_from_file_location("pull", os.path.join(os.path.dirname(__file__), "pull_testflight_feedback.py"))
pull = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pull)

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


test("DER signature → r‖s, 64 bytes, leading zeros handled", lambda: (
    pull.der_to_raw(bytes([0x30, 0x46, 0x02, 0x21, 0x00]) + b"\x80" + b"\x01" * 31 + bytes([0x02, 0x21, 0x00]) + b"\x81" + b"\x02" * 31)
    == b"\x80" + b"\x01" * 31 + b"\x81" + b"\x02" * 31
    and pull.der_to_raw(bytes([0x30, 0x06, 0x02, 0x01, 0x05, 0x02, 0x01, 0x07])) == b"\x00" * 31 + b"\x05" + b"\x00" * 31 + b"\x07"))


def signs_a_token():
    with tempfile.TemporaryDirectory() as folder:   # a throwaway key, made here
        key = os.path.join(folder, "k.p8")
        subprocess.run(["openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", key], check=True, capture_output=True)
        jwt = pull.token("KEYID", "ISSUER", key, now=1000)
        header, claims, signature = jwt.split(".")
        pad = lambda s: s + "=" * (-len(s) % 4)
        return (json.loads(base64.urlsafe_b64decode(pad(header))) == {"alg": "ES256", "kid": "KEYID", "typ": "JWT"}
                and json.loads(base64.urlsafe_b64decode(pad(claims)))["exp"] == 2200
                and len(base64.urlsafe_b64decode(pad(signature))) == 64)

test("token: ES256 header, 20 minutes, a 64-byte signature", signs_a_token)

item = {"id": "abc", "attributes": {"createdDate": "2026-11-05T02:09:37.145Z", "comment": "Chin", "timeZone": "Asia/Hong_Kong",
                                    "screenshots": [{"url": "https://x/1.jpg", "width": 1}, "https://x/2.jpg", {"no": "url"}]},
        "relationships": {"build": {"data": {"type": "builds", "id": "b1"}}}}
test("a submission becomes a feedback.json the sorter reads, with the build number", lambda: (
    (info := pull.as_feedback_json(item, {"b1": "6"}))["timestamp"] == "2026-11-05T02:09:37.145Z" and info["cfBundleVersion"] == "6"
    and info["timezone"] == "Asia/Hong_Kong" and info["id"] == "abc"
    and pull.sort_feedback.testflight_place(info).endswith("1009-testflight-chin")))
test("odd or missing fields don't stop it", lambda: (
    pull.as_feedback_json({"id": "x"}, {})["cfBundleVersion"] == "?" and pull.as_feedback_json({"id": "x"}, {})["comment"] == ""
    and pull.as_feedback_json({"id": "x", "attributes": {"comment": None}}, {})["comment"] == ""
    and pull.sort_feedback.comment_slug(None) == "no-comment"
    and pull.screenshot_urls({"screenshots": "nope"}) == []))
test("screenshots: urls from objects or strings", lambda: pull.screenshot_urls(item["attributes"]) == ["https://x/1.jpg", "https://x/2.jpg"])

spec2 = importlib.util.spec_from_file_location("notes", os.path.join(os.path.dirname(__file__), "testflight_notes.py"))
notes = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(notes)
test("What to Test: the first text is added to the build; a later one replaces it", lambda: (
    (add := notes.localization_request("b1", "Hi"))[0:2] == ("POST", "/v1/betaBuildLocalizations")
    and add[2]["data"]["attributes"] == {"whatsNew": "Hi", "locale": "en-US"} and add[2]["data"]["relationships"]["build"]["data"]["id"] == "b1"
    and (edit := notes.localization_request("b1", "Yo", "l9"))[0:2] == ("PATCH", "/v1/betaBuildLocalizations/l9")
    and edit[2]["data"]["id"] == "l9" and "locale" not in edit[2]["data"]["attributes"]))
test("What to Test: the new testers' text follows this build's", lambda: notes.full_text(" New \n", "Welcome\n") == "New\n\nWelcome")

def too_long_text_stops():
    try:
        notes.full_text("x" * notes.MAX_TEXT_CHARACTERS, "Welcome")
    except SystemExit as stopped:
        return "characters" in str(stopped)
    return False

test("What to Test: text over Apple's limit stops before sending", too_long_text_stops)

print(f"FAILED {failures} of {count} tests" if failures else f"PASS {count} tests")
raise SystemExit(1 if failures else 0)
