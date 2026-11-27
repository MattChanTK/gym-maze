"""Tests for kit/ios/sort_feedback.py on made-up feedback in a temporary folder (no real feedback, no network).
Run by kit/ios/test.sh; prints failures and one summary line."""
import importlib.util
import json
import os
import tempfile
import zipfile

spec = importlib.util.spec_from_file_location("sort_feedback", os.path.join(os.path.dirname(__file__), "sort_feedback.py"))
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)

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


sample = {"timestamp": "2026-11-04T03:38:07.682Z[UTC]", "timezone": "Asia/Hong_Kong", "cfBundleVersion": "5",
          "comment": "It changed the scene, too much!"}

test("folder name: the phone's date and time and the comment's first words", lambda:
     sf.testflight_place(sample) == os.path.join("2026-11-04", "1138-testflight-it-changed-the-scene-too"))
test("no comment and an unknown time zone still give a name", lambda:
     sf.testflight_place({**sample, "comment": "", "timezone": "Nowhere/Here"}).endswith("0338-testflight-no-comment"))

test("a two-digit fraction of a second (as Apple sometimes sends) still reads", lambda:
     sf.testflight_place({**sample, "timestamp": "2026-11-08T06:18:46.78+00:00"}).startswith(os.path.join("2026-11-08", "1418-")))


def filed_again_is_skipped(folder):
    again = os.path.join(folder, "testflight_feedback_abc")   # the same report, pulled from the API later
    os.makedirs(again)
    json.dump(sample, open(os.path.join(again, "feedback.json"), "w"))
    return sf.sort_feedback(folder) == [] and not os.path.exists(again)


def sorts_zips_folders_and_photos():
    with tempfile.TemporaryDirectory() as folder:
        with zipfile.ZipFile(os.path.join(folder, "testflight_feedback.zip"), "w") as archive:
            archive.writestr("feedback.json", json.dumps(sample))
            archive.writestr("../escape.txt", "x")          # stays inside: only the plain name is used
        unzipped = os.path.join(folder, "testflight_feedback (1)")
        os.makedirs(unzipped)
        json.dump({**sample, "comment": "Zoom is made up"}, open(os.path.join(unzipped, "feedback.json"), "w"))
        open(unzipped + ".zip", "w").close()                 # its zip, already unzipped: just removed
        open(os.path.join(folder, "IMG_1.HEIC"), "w").close()
        with zipfile.ZipFile(os.path.join(folder, "?testflight_feedback (2).zip"), "w") as archive:   # an odd name from a download
            archive.writestr("feedback.json", json.dumps({**sample, "comment": "Odd name"}))
        rows = sf.sort_feedback(folder)
        day = os.path.join(folder, "2026-11-04")
        readme = open(os.path.join(folder, "README.md")).read()
        return (len(rows) == 4 and os.path.isfile(os.path.join(day, "1138-testflight-odd-name", "feedback.json")) and os.path.isfile(os.path.join(day, "1138-testflight-it-changed-the-scene-too", "feedback.json"))
                and os.path.isfile(os.path.join(day, "1138-testflight-zoom-is-made-up", "feedback.json"))
                and not any(n.endswith(".zip") or n == "IMG_1.HEIC" for n in os.listdir(folder))
                and not os.path.exists(os.path.join(folder, "escape.txt")) and readme.count("| ? |") == 4
                and sf.sort_feedback(folder) == [] and filed_again_is_skipped(folder))

test("zips, folders and photos are filed by date and listed once", sorts_zips_folders_and_photos)

print(f"FAILED {failures} of {count} tests" if failures else f"PASS {count} tests")
raise SystemExit(1 if failures else 0)
