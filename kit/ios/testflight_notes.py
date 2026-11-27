#!/usr/bin/env python3
"""Sets a TestFlight build's "What to Test" text (what testers read) through App Store Connect.

    python3 kit/ios/testflight_notes.py <text file>                  # the latest upload on App Store Connect
    python3 kit/ios/testflight_notes.py <build number> <text file>   # another build

The text file says what's new in this build; the getting-started text for new testers (docs/testflight_new_testers.txt)
is added after it every time. Finds the build (waits up to 10 minutes for Apple to list it), then adds or replaces its en-US text. Uses the same
key as kit/ios/pull_testflight_feedback.py. It's what testers see: ask the user first.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import asc_build_number  # noqa: E402
import pull_testflight_feedback as asc  # noqa: E402

LOCALE = "en-US"
WAIT_SECONDS, POLL_SECONDS = 600, 20
MAX_TEXT_CHARACTERS = 4000   # App Store Connect's limit for What to Test
NEW_TESTERS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "testflight_new_testers.txt")


def full_text(whats_new, new_testers):
    """What testers read: this build's text, then the getting-started text. Too long for Apple stops here."""
    text = f"{whats_new.strip()}\n\n{new_testers.strip()}"
    if len(text) > MAX_TEXT_CHARACTERS:
        sys.exit(f"What to Test is {len(text)} characters; App Store Connect allows {MAX_TEXT_CHARACTERS}")
    return text


def localization_request(build_id, text, existing_id=None):
    """(method, path, body) that sets the text: replaces the build's existing text, or adds the first."""
    attributes = {"whatsNew": text}
    if existing_id:
        return "PATCH", f"/v1/betaBuildLocalizations/{existing_id}", {
            "data": {"type": "betaBuildLocalizations", "id": existing_id, "attributes": attributes}}
    return "POST", "/v1/betaBuildLocalizations", {"data": {
        "type": "betaBuildLocalizations", "attributes": {**attributes, "locale": LOCALE},
        "relationships": {"build": {"data": {"type": "builds", "id": build_id}}}}}


def send(method, path, body, auth):
    request = urllib.request.Request(asc.API + path, data=json.dumps(body).encode(), method=method,
                                     headers={"Authorization": f"Bearer {auth}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
            return json.loads(data) if data else {}   # adding to a group answers 204, no body
    except urllib.error.HTTPError as error:
        sys.exit(f"App Store Connect said HTTP {error.code}: {error.read().decode(errors='replace')[:400]}")


def main(argv):
    if len(argv) not in (1, 2):
        sys.exit(__doc__)
    number = argv[0] if len(argv) == 2 else str(asc_build_number.latest_uploaded())
    text = full_text(open(argv[-1]).read(), open(NEW_TESTERS_FILE).read())
    env = {**asc.load_env(), **{k: v for k, v in os.environ.items() if k.isupper()}}
    key_id, issuer, bundle = env.get("ASC_KEY_ID"), env.get("ASC_ISSUER_ID"), env.get("APPLE_BUNDLE_ID")
    if not (key_id and issuer and bundle):
        sys.exit("Missing ASC_KEY_ID, ASC_ISSUER_ID or APPLE_BUNDLE_ID in .env.local")
    key_path = os.path.expanduser(f"~/.appstoreconnect/private_keys/AuthKey_{key_id}.p8")
    auth = asc.token(key_id, issuer, key_path)
    apps = asc.get("/v1/apps?" + urllib.parse.urlencode({"filter[bundleId]": bundle}), auth).get("data") or []
    if not apps:
        sys.exit("No app with that bundle ID on App Store Connect")
    query = urllib.parse.urlencode({"filter[app]": apps[0]["id"], "filter[version]": number, "limit": 1})
    deadline = time.time() + WAIT_SECONDS
    while True:
        builds = asc.get(f"/v1/builds?{query}", auth).get("data") or []
        if builds:
            break
        if time.time() > deadline:
            sys.exit(f"Build {number} isn't listed yet; try again in a few minutes")
        time.sleep(POLL_SECONDS)
    build = builds[0]
    existing = asc.get(f"/v1/builds/{build['id']}/betaBuildLocalizations", auth).get("data") or []
    match = next((l["id"] for l in existing if (l.get("attributes") or {}).get("locale") == LOCALE), None)
    send(*localization_request(build["id"], text, match), asc.token(key_id, issuer, key_path))
    state = (build.get("attributes") or {}).get("processingState", "?")
    print(f"Set What to Test for build {number} ({state})")


if __name__ == "__main__":
    main(sys.argv[1:])
