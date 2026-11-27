#!/usr/bin/env python3
"""Keeps docs/CHANGELOG.md small. When it is over MAX_BYTES, the oldest entries move to files in docs/changelog/,
named by the dates they cover, until it is down to TARGET_BYTES. Entries are never cut; each archive file stays
under ARCHIVE_FILE_BYTES unless one entry alone is bigger.

  python3 tools/roll_changelog.py            roll if needed, print what moved
  python3 tools/roll_changelog.py --check    exit 1 if CHANGELOG.md is over the limit (used by tools/check.sh)
"""
import os
import re
import subprocess
import sys

MAX_BYTES = 80_000
TARGET_BYTES = 40_000
ARCHIVE_FILE_BYTES = 45_000

CHANGELOG_PATH = "docs/CHANGELOG.md"
ARCHIVE_FOLDER = "docs/changelog"
ENTRY_HEADING = re.compile(r"## (20\d\d-\d\d-\d\d)")
ARCHIVE_POINTER = (
    "This file holds the newest entries only. Older entries are in `docs/changelog/`, one file per run of dates "
    "(`<first date>_to_<last date>.md`, sorted by name). Read the top of this file for recent history and grep the "
    "older files instead of reading them whole. `kit/tools/roll_changelog.py` moves the oldest entries there when this "
    "file passes 80 KB.\n\n"
)
POINTER_START = "This file holds the newest entries only."


def split_changelog(text):
    """(preamble, entries): each entry is the text from its `## date` heading to the next, newest first."""
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if ENTRY_HEADING.match(line)]
    if not starts:
        raise SystemExit(f"{CHANGELOG_PATH} has no '## YYYY-MM-DD' entries")
    preamble = "\n".join(lines[: starts[0]])
    entries = ["\n".join(lines[a:b]).rstrip("\n") + "\n" for a, b in zip(starts, starts[1:] + [len(lines)])]
    return preamble, entries


def entry_date(entry):
    return ENTRY_HEADING.match(entry).group(1)


def size(entries):
    return sum(len(entry.encode()) for entry in entries)


def shift_relative_links(text, prefix):
    def fix(match):
        target = match.group(2)
        if re.match(r"(https?:|mailto:|#|/)", target):
            return match.group(0)
        return f"{match.group(1)}{prefix}{target})"
    return re.sub(r"(\]\()([^)]+)\)", fix, text)


def chunk_oldest_first(entries_newest_first, file_bytes):
    """Groups entries into runs each under file_bytes (one entry alone may exceed it); each run stays newest first."""
    chunks, current, current_bytes = [], [], 0
    for entry in reversed(entries_newest_first):
        entry_bytes = len(entry.encode())
        if current and current_bytes + entry_bytes > file_bytes:
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(entry)
        current_bytes += entry_bytes
    if current:
        chunks.append(current)
    return [list(reversed(chunk)) for chunk in chunks]


def entries_to_move(entries):
    """How many of the oldest entries to move so the rest fits TARGET_BYTES; 0 when already within MAX_BYTES."""
    if size(entries) <= MAX_BYTES:
        return 0
    kept, kept_bytes = 0, 0
    for entry in entries:
        entry_bytes = len(entry.encode())
        if kept_bytes + entry_bytes > TARGET_BYTES and kept > 0:
            break
        kept += 1
        kept_bytes += entry_bytes
    return len(entries) - kept


def archive_name(chunk, existing_names):
    base = f"{entry_date(chunk[-1])}_to_{entry_date(chunk[0])}"
    name, number = f"{base}.md", 2
    while name in existing_names:
        name, number = f"{base}-{number}.md", number + 1
    return name


def roll(changelog_text, existing_archive_names):
    """Returns (new changelog text, {archive file name: text}); unchanged and empty when within the limit."""
    preamble, entries = split_changelog(changelog_text)
    move_count = entries_to_move(entries)
    if move_count == 0:
        return changelog_text, {}
    kept, moved = entries[:-move_count], entries[-move_count:]
    archives, names = {}, set(existing_archive_names)
    for chunk in chunk_oldest_first(moved, ARCHIVE_FILE_BYTES):
        name = archive_name(chunk, names)
        names.add(name)
        header = (f"# Changelog {entry_date(chunk[-1])} to {entry_date(chunk[0])}\n\n"
                  "Newest first. The newest entries are in [CHANGELOG.md](../CHANGELOG.md).\n\n")
        archives[name] = header + shift_relative_links("\n".join(chunk), "../")
    if POINTER_START in preamble:
        intro = preamble.split(POINTER_START)[0].rstrip("\n")
    else:
        intro = preamble.rstrip("\n")
    return intro + "\n\n" + ARCHIVE_POINTER + "\n".join(kept), archives


def main():
    text = open(CHANGELOG_PATH).read()
    _, entries = split_changelog(text)
    if "--check" in sys.argv:
        if size(entries) > MAX_BYTES:
            print(f"{CHANGELOG_PATH} is {size(entries) // 1000} KB, over {MAX_BYTES // 1000} KB: run python3 kit/tools/roll_changelog.py")
            sys.exit(1)
        return
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
    new_text, archives = roll(text, os.listdir(ARCHIVE_FOLDER))
    for name, content in archives.items():
        with open(f"{ARCHIVE_FOLDER}/{name}", "w") as archive_file:
            archive_file.write(content)
    open(CHANGELOG_PATH, "w").write(new_text)
    print(f"Moved {len(archives)} file(s): {', '.join(archives)}" if archives else "Within the limit; nothing moved")


if __name__ == "__main__":
    os.chdir(subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())
    main()
