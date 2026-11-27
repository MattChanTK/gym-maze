#!/usr/bin/env python3
"""Pulls TestFlight feedback (testers' screenshots and comments) from App Store Connect into feedbacks/ (git-ignored),
then files it by date with kit/ios/sort_feedback.py, the same as a downloaded zip.

    python3 kit/ios/pull_testflight_feedback.py

Uses the App Store Connect API key from the git-ignored .env.local (ASC_KEY_ID, ASC_ISSUER_ID; the key itself at
~/.appstoreconnect/private_keys/AuthKey_<ASC_KEY_ID>.p8) and APPLE_BUNDLE_ID. Read-only: it never changes anything on
App Store Connect. Feedback already pulled (feedbacks/.pulled) is skipped. Nothing secret is printed.
"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import sort_feedback  # noqa: E402

API = "https://api.appstoreconnect.apple.com"
FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "feedbacks")
PULLED = ".pulled"   # in FOLDER: the submission IDs already pulled, one per line


def load_env(path=".env.local"):
    values = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def b64url(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def der_to_raw(der, size=32):
    """openssl's ECDSA signature (DER: SEQUENCE of two INTEGERs) → the r‖s bytes a JWT (ES256) wants."""
    if len(der) < 8 or der[0] != 0x30:
        raise ValueError("not a DER signature")
    i = 2 if der[1] < 0x80 else 2 + (der[1] & 0x7F)
    parts = []
    for _ in range(2):
        if der[i] != 0x02:
            raise ValueError("not a DER integer")
        length = der[i + 1]
        value = der[i + 2:i + 2 + length].lstrip(b"\x00")
        parts.append(value.rjust(size, b"\x00"))
        i += 2 + length
    return parts[0] + parts[1]


def token(key_id, issuer_id, key_path, now=None):
    """A 20-minute App Store Connect token, signed with the .p8 key by macOS's openssl."""
    now = int(now or time.time())
    header = b64url(json.dumps({"alg": "ES256", "kid": key_id, "typ": "JWT"}).encode())
    claims = b64url(json.dumps({"iss": issuer_id, "iat": now, "exp": now + 1200, "aud": "appstoreconnect-v1"}).encode())
    signing_input = f"{header}.{claims}".encode()
    der = subprocess.run(["openssl", "dgst", "-sha256", "-sign", key_path], input=signing_input,
                         capture_output=True, check=True).stdout
    return f"{header}.{claims}.{b64url(der_to_raw(der))}"


def get(url, auth):
    request = urllib.request.Request(url if url.startswith("http") else API + url, headers={"Authorization": f"Bearer {auth}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(f"App Store Connect said HTTP {error.code}: {error.read().decode(errors='replace')[:300]}")


def as_feedback_json(item, builds):
    """One submission as TestFlight's own feedback.json looks (what sort_feedback reads): its attributes, with
    "timestamp" (from createdDate), "timezone" (from timeZone) and "cfBundleVersion" (the build) filled in."""
    attributes = dict(item.get("attributes") or {})
    info = {"id": item.get("id", ""), **attributes}
    info.setdefault("timestamp", attributes.get("createdDate") or "")
    info.setdefault("timezone", attributes.get("timeZone") or attributes.get("timezone") or "")
    build_id = ((item.get("relationships") or {}).get("build") or {}).get("data") or {}
    info.setdefault("cfBundleVersion", builds.get(build_id.get("id"), "?") if isinstance(build_id, dict) else "?")
    info["comment"] = info.get("comment") or ""   # a screenshot sent without a comment comes back as null
    return info


def screenshot_urls(info):
    """The screenshots' addresses, wherever the submission keeps them (a list of {url: …} or of strings)."""
    shots = info.get("screenshots") or []
    urls = []
    for shot in shots if isinstance(shots, list) else []:
        url = shot.get("url") if isinstance(shot, dict) else shot if isinstance(shot, str) else None
        if url:
            urls.append(url)
    return urls


def main():
    env = {**load_env(), **{k: v for k, v in os.environ.items() if k.isupper()}}
    key_id, issuer, bundle = env.get("ASC_KEY_ID"), env.get("ASC_ISSUER_ID"), env.get("APPLE_BUNDLE_ID")
    if not (key_id and issuer and bundle):
        sys.exit("Missing ASC_KEY_ID, ASC_ISSUER_ID or APPLE_BUNDLE_ID in .env.local")
    key_path = os.path.expanduser(f"~/.appstoreconnect/private_keys/AuthKey_{key_id}.p8")
    if not os.path.exists(key_path):
        sys.exit(f"No App Store Connect key at ~/.appstoreconnect/private_keys/AuthKey_<ASC_KEY_ID>.p8")
    auth = token(key_id, issuer, key_path)

    apps = get("/v1/apps?" + urllib.parse.urlencode({"filter[bundleId]": bundle}), auth).get("data") or []
    if not apps:
        sys.exit("No app with that bundle ID on App Store Connect")
    url = f"/v1/apps/{apps[0]['id']}/betaFeedbackScreenshotSubmissions?" + urllib.parse.urlencode({"limit": 200, "include": "build"})

    pulled_path = os.path.join(FOLDER, PULLED)
    pulled = set(open(pulled_path).read().split()) if os.path.exists(pulled_path) else set()
    new = 0
    while url:
        page = get(url, auth)
        builds = {b.get("id"): (b.get("attributes") or {}).get("version", "?") for b in page.get("included") or [] if b.get("type") == "builds"}
        for item in page.get("data") or []:
            if item.get("id") in pulled:
                continue
            info = as_feedback_json(item, builds)
            staging = os.path.join(FOLDER, f"testflight_feedback_{item['id']}")
            os.makedirs(staging, exist_ok=True)
            with open(os.path.join(staging, "feedback.json"), "w") as out:
                json.dump(info, out, indent=2)
            for n, shot in enumerate(screenshot_urls(info), 1):
                try:
                    urllib.request.urlretrieve(shot, os.path.join(staging, f"screenshot_{n:02d}.jpg"))
                except (urllib.error.URLError, OSError):
                    pass   # an expired link: the comment and details are still kept
            pulled.add(item["id"])
            new += 1
        url = (page.get("links") or {}).get("next")
    with open(pulled_path, "w") as out:
        out.write("\n".join(sorted(pulled)) + "\n")
    rows = sort_feedback.sort_feedback(FOLDER) if new else []
    print("\n".join(rows) if rows else "No new TestFlight feedback.")


if __name__ == "__main__":
    main()
