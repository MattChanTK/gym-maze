"""Tests for tools/roll_changelog.py on made-up changelogs (no real files touched).
Run by tools/test_py.sh; prints failures and one summary line."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location("roll_changelog", os.path.join(os.path.dirname(__file__), "roll_changelog.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

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


def make_changelog(entry_count, entry_bytes):
    """Newest first: entry 0 is the newest, dated one day later each step back toward 2026-11-01."""
    preamble = "# Changelog\n\nIntro with a [link](BACKLOG.md).\n\n"
    entries = []
    for number in range(entry_count):
        day = entry_count - number
        body = "x" * entry_bytes
        entries.append(f"## 2026-11-{day:02d}: Entry {number} ([README](../README.md))\n\n{body}\n")
    return preamble + "\n".join(entries)


small = make_changelog(5, 100)
test("a changelog within the limit moves nothing", lambda: rc.roll(small, []) == (small, {}))

big = make_changelog(28, rc.MAX_BYTES // 28 + 200)  # a little over the limit
new_text, archives = rc.roll(big, [])
_, kept_entries = rc.split_changelog(new_text)
_, original_entries = rc.split_changelog(big)
moved_entries = [e for text in archives.values() for e in rc.split_changelog(text)[1]]

test("over the limit, the kept entries fit the target", lambda: rc.size(kept_entries) <= rc.TARGET_BYTES + len(original_entries[0]))
test("the newest entry stays in the changelog", lambda: kept_entries[0].startswith(original_entries[0].split("\n")[0]))
test("the oldest entry is among those moved", lambda: rc.entry_date(original_entries[-1]) in [rc.entry_date(e) for e in moved_entries])
test("no entry is lost or duplicated", lambda: len(kept_entries) + len(moved_entries) == len(original_entries))
test("every archive file stays under the file limit", lambda: all(len(text.encode()) <= rc.ARCHIVE_FILE_BYTES + 400 for text in archives.values()))
test("an archive is named by its date range, oldest first",
     lambda: all(name.endswith(".md") and "_to_" in name and name.split("_to_")[0] <= name.split("_to_")[1][:10] for name in archives))
test("relative links in moved entries point one folder up", lambda: all("](../../README.md)" in text for text in archives.values()))
test("the changelog's own links are kept", lambda: "](BACKLOG.md)" in new_text)
test("the archive pointer is in the preamble once", lambda: new_text.count(rc.POINTER_START) == 1)

again_text, again_archives = rc.roll(new_text, archives.keys())
test("rolling again right after moves nothing", lambda: again_archives == {} and again_text == new_text)

huge = make_changelog(40, 5000)  # 200 KB: needs several archive files
_, huge_archives = rc.roll(huge, [])
test("a big backlog of entries is spread over several files", lambda: len(huge_archives) >= 3)

one_big_entry = make_changelog(1, rc.MAX_BYTES + 5000)
test("a lone oversized entry is kept, never cut", lambda: rc.roll(one_big_entry, [])[1] == {})

same_day = ["## 2026-11-05: a\n\nbody\n", "## 2026-11-05: b\n\nbody\n"]
test("a name clash gets a number",
     lambda: rc.archive_name(same_day, {"2026-11-05_to_2026-11-05.md"}) == "2026-11-05_to_2026-11-05-2.md")

try:
    rc.split_changelog("# Changelog\n\nno entries\n")
    test("a changelog with no entries is an error", lambda: False)
except SystemExit:
    test("a changelog with no entries is an error", lambda: True)

print(f"{'PASS' if failures == 0 else 'FAIL'} {count} roll_changelog tests" + (f" ({failures} failed)" if failures else ""))
raise SystemExit(1 if failures else 0)
