#!/usr/bin/env python3
"""Checks what's about to be committed or pushed for secrets and personal values, without printing them.

    python3 tools/scan_secrets.py            # the staged diff (before a commit)
    python3 tools/scan_secrets.py --push     # the commits not on origin yet (before a push)
    python3 tools/scan_secrets.py --all      # every tracked file (CI). Doesn't need CLAUDE.local.md.

Looks for: the values in the git-ignored CLAUDE.local.md (device IDs, team IDs…) and .env.local (keys, secrets; not the
servers' public addresses), your git email, private key blocks and common API key shapes. Only added lines count, except
--all, which reads the tracked tree (and secret file names: *.p8, *.pem, AuthKey*, qa.env). Prints which name was found in which file
(never the value) and exits 1, or prints nothing and exits 0. The staged and push scans also exit 1 if CLAUDE.local.md is missing, since its values
couldn't be checked. --all still scans patterns when that file isn't there (CI has none).
"""
import fnmatch
import os
import re
import subprocess
import sys

PATTERNS = {
    "a private key block": r"BEGIN [A-Z ]*PRIVATE KEY",
    "an OpenRouter key": r"sk-or-v1-[0-9a-f]{20,}",
    "an Anthropic key": r"sk-ant-[A-Za-z0-9_-]{20,}",
    "an OpenAI key": r"sk-(proj-)?[A-Za-z0-9]{32,}",
    "a GitHub token": r"\b(ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{20,}",
    "an AWS access key": r"\bAKIA[0-9A-Z]{16}\b",
}
MIN_LENGTH = 6   # shorter values ("1", "true") would match everywhere
# Settings in .env.local that are public on purpose (server addresses the app ships, a public contact address).
# List them in kit.env: KIT_PUBLIC_SETTINGS="SERVER_URL TEST_SERVER_URL CONTACT_EMAIL".
PUBLIC_SETTINGS = set(os.environ.get("KIT_PUBLIC_SETTINGS", "").replace(",", " ").split())


def local_values(*paths):
    """NAME=value pairs from the git-ignored files (a value after # is a comment); short ones are skipped."""
    values = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for line in open(path):
            match = re.match(r"\s*([A-Z][A-Z0-9_]*)=([^#\s]+)", line)
            if match and match.group(1) not in PUBLIC_SETTINGS and len(match.group(2).strip("'\"")) >= MIN_LENGTH:
                values[match.group(1)] = match.group(2).strip("'\"")
    return values


def findings(diff, values):
    """(what, file) for each secret seen on an added line of a unified diff."""
    found, current = [], "?"
    for line in diff.splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+") and not line.startswith("+++"):
            for name, value in values.items():
                if value in line:
                    found.append((name, current))
            for what, pattern in PATTERNS.items():
                if re.search(pattern, line):
                    found.append((what, current))
    return sorted(set(found))


def secret_file_name(path):
    """A tracked file whose name is a signing key or the QA env, or None."""
    name = os.path.basename(path)
    if name == "qa.env" or fnmatch.fnmatch(name, "AuthKey*") or fnmatch.fnmatch(name, "*.p8") or fnmatch.fnmatch(name, "*.pem"):
        return "a secret file name"
    return None


def tree_findings(files, values=None):
    """(what, file) for secret patterns and secret file names. `files` is (path, text) for tracked files."""
    values = values or {}
    found = []
    for path, text in files:
        named = secret_file_name(path)
        if named:
            found.append((named, path))
        added = "+++ b/" + path + "\n" + "\n".join("+" + line for line in text.splitlines())
        found.extend(findings(added, values))
    return sorted(set(found))


def tracked_texts(root):
    """Tracked text files as (path, text). Binary files (a NUL in the first 8KB) are skipped."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True).stdout.split(b"\0")
    files = []
    for raw in listed:
        if not raw:
            continue
        path = raw.decode("utf-8", "surrogateescape")
        try:
            with open(os.path.join(root, path), "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        files.append((path, data.decode("utf-8", "replace")))
    return files


def local_secret_values(root):
    """Values from CLAUDE.local.md and .env.local when those files are here, plus the git email. Empty when they aren't."""
    device_settings_path = os.path.join(root, "CLAUDE.local.md")
    if not os.path.exists(device_settings_path):
        return {}
    values = local_values(device_settings_path, os.path.join(root, ".env.local"))
    email = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True, cwd=root).stdout.strip()
    if email:
        values["your git email"] = email
    return values


def main(argv):
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    if "--all" in argv:
        found = tree_findings(tracked_texts(root), local_secret_values(root))
        for what, where in found:
            print(f"Found {what} in {where}")
        return 1 if found else 0
    device_settings_path = os.path.join(root, "CLAUDE.local.md")
    if not os.path.exists(device_settings_path):
        print("Can't scan: CLAUDE.local.md is missing, so its values wouldn't be checked")
        return 1
    values = local_secret_values(root)
    command = ["git", "log", "-p", "origin/" + os.environ.get("KIT_MAIN_BRANCH", "main") + "..HEAD"] if "--push" in argv else ["git", "diff", "--cached"]
    diff = subprocess.run(command, capture_output=True, text=True, cwd=root).stdout
    found = findings(diff, values)
    for what, where in found:
        print(f"Found {what} in {where}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
