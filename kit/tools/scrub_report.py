#!/usr/bin/env python3
"""Scrub secrets and personal values from QA logs and reports.

    python3 tools/scrub_report.py [file ...]          # write scrubbed text to stdout
    python3 tools/scrub_report.py --in-place file ...  # rewrite those files
    python3 tools/scrub_report.py                      # stdin to stdout

Replaces API keys, tokens, JWTs, emails, simulator and device UDIDs, and
Apple team IDs. Never prints the original values. Used by tools/qa_dashboard.py.
"""
import argparse
import os
import re
import sys

SCRUB_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----|$)"), "[private key]"),
    (re.compile(r"\bsk-(?:or-v1-|ant-|proj-)?[A-Za-z0-9_-]{20,}"), "[key]"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}"), "[key]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"), "[jwt]"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{16,}"), r"\1 [key]"),
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:api[_-]?key|secret|token|password|passwd|authorization)[A-Z0-9_]*)(\s*[=:]\s*)[\"']?[^\s\"',;&]{16,}"),
     r"\1\2[redacted]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    (re.compile(r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b"), "[udid]"),
    (re.compile(r"\b0000[0-9A-Fa-f]{4}-[0-9A-Fa-f]{16}\b"), "[udid]"),
    (re.compile(r"(?i)\b((?:development_?team|team_?id|teamidentifier|team)\s*[=:]\s*[\"']?)[A-Z0-9]{10}\b"), r"\1[team]"),
    (re.compile(r"\((?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10}\)"), "([team])"),
]


def scrub(text, extra=None):
    """Return text with secrets replaced. extra is an optional list of exact strings to hide."""
    if text is None:
        return ""
    for value in extra or []:
        if value:
            text = text.replace(value, "[redacted]")
    for pattern, repl in SCRUB_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def _read(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _write(path, text, in_place):
    if in_place:
        tmp = path + ".scrub-tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    else:
        sys.stdout.write(text)


def main(argv):
    parser = argparse.ArgumentParser(description="Scrub secrets from QA logs and reports.")
    parser.add_argument("files", nargs="*", help="files to scrub (stdin if omitted)")
    parser.add_argument("--in-place", action="store_true", help="rewrite the given files")
    args = parser.parse_args(argv)
    if args.in_place and not args.files:
        sys.exit("scrub_report.py: --in-place needs at least one file")
    if args.in_place and "-" in args.files:
        sys.exit("scrub_report.py: --in-place cannot be used with stdin")
    paths = args.files or ["-"]
    for path in paths:
        _write(path, scrub(_read(path)), args.in_place)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
