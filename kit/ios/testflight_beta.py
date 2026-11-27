#!/usr/bin/env python3
"""Sends a TestFlight build to the external testers (the public-link and invited beta testers) through App Store Connect.

    python3 kit/ios/testflight_beta.py                  # the latest upload on App Store Connect
    python3 kit/ios/testflight_beta.py <build number>   # another build

Waits for Apple to finish processing the build (up to 30 minutes), adds it to every external beta group, and submits
it for Apple's beta review (a new version needs one; Apple then releases it to the groups by itself). Internal testers
get builds without this. Uses the same key as kit/ios/pull_testflight_feedback.py. It's what testers get: ask the user first.
"""
import os
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(__file__))
import asc_build_number  # noqa: E402
import pull_testflight_feedback as asc  # noqa: E402
import testflight_notes as notes  # noqa: E402

WAIT_SECONDS, POLL_SECONDS = 1800, 30


def external_groups(answer):
    """(id, name) of each group that isn't App Store Connect's internal one."""
    return [(g["id"], (g.get("attributes") or {}).get("name", "?")) for g in answer.get("data") or []
            if not (g.get("attributes") or {}).get("isInternalGroup")]


def add_to_group_request(group_id, build_id):
    return "POST", f"/v1/betaGroups/{group_id}/relationships/builds", {"data": [{"type": "builds", "id": build_id}]}


def review_request(build_id):
    return "POST", "/v1/betaAppReviewSubmissions", {"data": {
        "type": "betaAppReviewSubmissions", "relationships": {"build": {"data": {"type": "builds", "id": build_id}}}}}


def main(argv):
    if len(argv) > 1:
        sys.exit(__doc__)
    number = argv[0] if argv else str(asc_build_number.latest_uploaded())
    env = {**asc.load_env(), **{k: v for k, v in os.environ.items() if k.isupper()}}
    key_id, issuer, bundle = env.get("ASC_KEY_ID"), env.get("ASC_ISSUER_ID"), env.get("APPLE_BUNDLE_ID")
    if not (key_id and issuer and bundle):
        sys.exit("Missing ASC_KEY_ID, ASC_ISSUER_ID or APPLE_BUNDLE_ID in .env.local")
    key_path = os.path.expanduser(f"~/.appstoreconnect/private_keys/AuthKey_{key_id}.p8")
    auth = lambda: asc.token(key_id, issuer, key_path)   # a fresh token each call: the wait outlasts one
    apps = asc.get("/v1/apps?" + urllib.parse.urlencode({"filter[bundleId]": bundle}), auth()).get("data") or []
    if not apps:
        sys.exit("No app with that bundle ID on App Store Connect")
    app_id = apps[0]["id"]
    query = urllib.parse.urlencode({"filter[app]": app_id, "filter[version]": number, "limit": 1})
    deadline = time.time() + WAIT_SECONDS
    while True:
        builds = asc.get(f"/v1/builds?{query}", auth()).get("data") or []
        state = builds and (builds[0].get("attributes") or {}).get("processingState")
        if state == "VALID":
            break
        if state in ("FAILED", "INVALID"):
            sys.exit(f"Build {number} is {state} on App Store Connect")
        if time.time() > deadline:
            sys.exit(f"Build {number} is still {state or 'not listed'}; try again later")
        time.sleep(POLL_SECONDS)
    build_id = builds[0]["id"]
    groups = external_groups(asc.get(f"/v1/apps/{app_id}/betaGroups?limit=50", auth()))
    if not groups:
        sys.exit("No external beta group on App Store Connect")
    for group_id, _ in groups:
        notes.send(*add_to_group_request(group_id, build_id), auth())
    notes.send(*review_request(build_id), auth())
    print(f"Build {number} added to {', '.join(name for _, name in groups)} and submitted for beta review")


if __name__ == "__main__":
    main(sys.argv[1:])
