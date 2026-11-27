#!/usr/bin/env python3
"""The TestFlight build number, from App Store Connect (read-only). The repo doesn't keep it any more.

    python3 "$KIT_DIR/ios/asc_build_number.py"           # the next one: highest uploaded + 1 (what kit/ios/testflight.sh passes)
    python3 "$KIT_DIR/ios/asc_build_number.py" --latest  # the highest uploaded (what testflight_notes.py and testflight_beta.py default to)

Prints just the number. Never lower than the project's CURRENT_PROJECT_VERSION (43 when this started), so a short or
empty answer can't reuse an old number. Key: ASC_KEY_ID, ASC_ISSUER_ID and APPLE_BUNDLE_ID (or BUNDLE) from
.env.local or the environment; the .p8 at ASC_KEY_PATH, else ~/.appstoreconnect/private_keys/AuthKey_<id>.p8.
A build Apple hasn't listed yet (the first minutes after an upload) isn't counted: wait before the next upload.
"""
import os
import re
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(__file__))
import pull_testflight_feedback as asc  # noqa: E402

PBXPROJ = os.path.join(os.path.dirname(__file__), "..", "..", os.environ.get("KIT_XCODE_PROJECT", "App.xcodeproj"), "project.pbxproj")
PAGE = 200   # the most recent uploads asked for; the highest number is always among them


def project_floor(pbxproj=PBXPROJ):
    """CURRENT_PROJECT_VERSION in the project: no longer raised, only a floor."""
    match = re.search(r"CURRENT_PROJECT_VERSION = (\d+);", open(pbxproj).read())
    return int(match.group(1)) if match else 0


def highest(answer, floor=0):
    """The highest whole build number in a /v1/builds answer, or the floor if it's higher."""
    numbers = [int(v) for v in ((b.get("attributes") or {}).get("version") for b in answer.get("data") or [])
               if isinstance(v, str) and v.isdigit()]
    return max(numbers + [floor])


def builds_query(app_id):
    # Build numbers are text to Apple, so sorting by them puts "9" after "43": take the newest uploads instead.
    return "/v1/builds?" + urllib.parse.urlencode(
        {"filter[app]": app_id, "sort": "-uploadedDate", "limit": PAGE, "fields[builds]": "version"})


def credentials(env):
    key_id, issuer = env.get("ASC_KEY_ID"), env.get("ASC_ISSUER_ID")
    bundle = env.get("APPLE_BUNDLE_ID") or env.get("BUNDLE")
    if not (key_id and issuer and bundle):
        sys.exit("Missing ASC_KEY_ID, ASC_ISSUER_ID or APPLE_BUNDLE_ID in .env.local")
    key_path = env.get("ASC_KEY_PATH") or os.path.expanduser(f"~/.appstoreconnect/private_keys/AuthKey_{key_id}.p8")
    return key_id, issuer, key_path, bundle


def latest_uploaded(env=None, floor=None):
    """Asks App Store Connect for the highest build number uploaded for the app."""
    env = env if env is not None else {**asc.load_env(), **{k: v for k, v in os.environ.items() if k.isupper()}}
    key_id, issuer, key_path, bundle = credentials(env)
    auth = asc.token(key_id, issuer, key_path)
    apps = asc.get("/v1/apps?" + urllib.parse.urlencode({"filter[bundleId]": bundle}), auth).get("data") or []
    if not apps:
        sys.exit("No app with that bundle ID on App Store Connect")
    return highest(asc.get(builds_query(apps[0]["id"]), auth), project_floor() if floor is None else floor)


def main(argv):
    if argv not in ([], ["--latest"]):
        sys.exit(__doc__)
    latest = latest_uploaded()
    print(latest if argv else latest + 1)


if __name__ == "__main__":
    main(sys.argv[1:])
