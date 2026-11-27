"""Files new testing feedback in feedbacks/ (git-ignored) by date, and lists it in feedbacks/README.md.

Copy TestFlight feedback (the downloaded zip or its unzipped folder) and photos into feedbacks/, then run:
    python3 kit/ios/sort_feedback.py
- TestFlight feedback goes to feedbacks/<date>/<time>-testflight-<first words of the comment>/ (the phone's own date
  and time, from feedback.json); the zip is unpacked and removed.
- Photos and screenshots go to feedbacks/<date>/ (the date the photo was taken).
- Each new item gets a row in the README's table with "?" (not triaged yet): see the routine in docs/FEEDBACK.md.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

PHOTO_TYPES = {".heic", ".jpg", ".jpeg", ".png", ".mov", ".mp4"}
TABLE_HEADER = "| When | What | Comment | Issues |"


def comment_slug(comment, words=5):
    """The comment's first few words, for a folder name: "It modified the scene too much" → "it-modified-the-scene-too"."""
    return "-".join(re.findall(r"[a-z0-9]+", (comment or "").lower())[:words]) or "no-comment"


def feedback_time(info):
    """When the tester sent it, in the phone's time zone (TestFlight writes UTC like "2026-11-04T03:38:07.682Z[UTC]")."""
    stamp = info["timestamp"].split("[")[0].replace("Z", "+00:00")
    # Python 3.9's fromisoformat takes only 3 or 6 digits of fraction; Apple sometimes sends 2 ("06:18:46.78").
    stamp = re.sub(r"\.(\d{1,6})", lambda m: "." + m.group(1).ljust(6, "0"), stamp, count=1)
    sent = datetime.fromisoformat(stamp)
    try:
        return sent.astimezone(ZoneInfo(info.get("timezone") or "UTC"))
    except Exception:
        return sent.astimezone(timezone.utc)


def testflight_place(info):
    """The folder (relative to feedbacks/) for one TestFlight feedback."""
    when = feedback_time(info)
    return os.path.join(when.strftime("%Y-%m-%d"), f"{when.strftime('%H%M')}-testflight-{comment_slug(info.get('comment', ''))}")


def photo_time(path):
    """When the photo was taken (Spotlight's content date, from the photo itself), else when the file was made."""
    try:
        found = subprocess.run(["mdls", "-raw", "-name", "kMDItemContentCreationDate", path],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return datetime.strptime(found, "%Y-%m-%d %H:%M:%S %z").astimezone()
    except Exception:
        stat = os.stat(path)
        return datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_mtime))


def unpack_zip(zip_path, into):
    """Unpacks only plain file names (no folders or ../), so a strange zip can't write outside `into`."""
    os.makedirs(into, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = os.path.basename(member.filename)
            if member.is_dir() or not name or name.startswith("."):
                continue
            with archive.open(member) as source, open(os.path.join(into, name), "wb") as target:
                shutil.copyfileobj(source, target)


def add_rows(readme, rows):
    """Appends rows to the README's table (made if missing)."""
    text = open(readme).read() if os.path.exists(readme) else (
        "# Testing feedback (on this Mac only, not in git)\n\n"
        "Sorted by `python3 kit/ios/sort_feedback.py`. Issue codes are in docs/FEEDBACK.md (FB-…).\n\n"
        f"{TABLE_HEADER}\n|---|---|---|---|\n")
    with open(readme, "w") as out:
        out.write(text.rstrip("\n") + "\n" + "".join(row + "\n" for row in rows))


def sort_feedback(folder):
    """Files everything new at the top of `folder`; returns the README rows it added."""
    rows = []
    for name in os.listdir(folder):   # a zip that was already unzipped next to it: the folder has the same files
        stem, kind = os.path.splitext(name)
        if "testflight_feedback" in name and kind.lower() == ".zip" and os.path.isdir(os.path.join(folder, stem)):
            os.remove(os.path.join(folder, name))
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        stem, kind = os.path.splitext(name)
        if "testflight_feedback" in name:   # also "?testflight_feedback (2).zip" from some downloads
            staging = path
            if kind.lower() == ".zip":
                staging = os.path.join(folder, stem)
                unpack_zip(path, staging)
                os.remove(path)
            elif not os.path.isdir(path):
                continue
            info = json.load(open(os.path.join(staging, "feedback.json")))
            place = testflight_place(info)
            if os.path.exists(os.path.join(folder, place)):   # already filed (e.g. downloaded, then pulled from the API)
                shutil.rmtree(staging)
                continue
            os.makedirs(os.path.dirname(os.path.join(folder, place)), exist_ok=True)
            shutil.move(staging, os.path.join(folder, place))
            when = feedback_time(info)
            rows.append(f"| {when.strftime('%Y-%m-%d %H:%M')} | [TestFlight, build {info.get('cfBundleVersion', '?')}]"
                        f"({place.replace(' ', '%20')}/) | {(info.get('comment') or '').strip()} | ? |")
        elif kind.lower() in PHOTO_TYPES and os.path.isfile(path):
            made = photo_time(path)
            day = made.strftime("%Y-%m-%d")
            os.makedirs(os.path.join(folder, day), exist_ok=True)
            shutil.move(path, os.path.join(folder, day, name))
            rows.append(f"| {made.strftime('%Y-%m-%d %H:%M')} | [Photo]({day}/{name}) |  | ? |")
    if rows:
        add_rows(os.path.join(folder, "README.md"), rows)
    return rows


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..", "feedbacks")
    added = sort_feedback(target)
    print("\n".join(added) if added else "Nothing new to sort.")
