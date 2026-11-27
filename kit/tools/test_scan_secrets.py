"""Tests for tools/scan_secrets.py with made-up values (no real secrets). Run by tools/test.sh."""
import importlib.util
import os
os.environ["KIT_PUBLIC_SETTINGS"] = "KIT_SERVER_URL TEST_KIT_SERVER_URL"
import shutil
import subprocess
import sys
import tempfile

tools_folder = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("scan", os.path.join(tools_folder, "scan_secrets.py"))
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

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


# The fake key and key block are put together here, so this file itself never holds a key-shaped line.
fake_block = "-----BEGIN EC " + "PRIVATE KEY-----"
fake_key = "sk-or-" + "v1-" + "0123456789abcdef" * 2
diff = f"""diff --git a/a.swift b/a.swift
+++ b/a.swift
+let device = "FAKE-DEVICE-1234"
-removed FAKE-DEVICE-1234 is fine
 context FAKE-DEVICE-1234 is fine
+++ b/notes.md
+{fake_block}
+placeholder "sk-or-…" is fine
+key {fake_key}
"""
test("added lines only: a local value, a key block, an OpenRouter key; a placeholder is fine", lambda: (
    scan.findings(diff, {"DEVICE": "FAKE-DEVICE-1234"})
    == [("DEVICE", "a.swift"), ("a private key block", "notes.md"), ("an OpenRouter key", "notes.md")]))
test("nothing found: an empty list", lambda: scan.findings("+++ b/x\n+hello\n", {"TEAM": "ABCDEF1234"}) == [])


def reads_local_files():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "local.md")
        open(path, "w").write("DEVICE=FAKE-DEVICE-1234   # my phone\nSHORT=ab\n  TEAM='ABCDEF1234'\nnot a setting\n")
        return scan.local_values(path, os.path.join(folder, "missing")) == {"DEVICE": "FAKE-DEVICE-1234", "TEAM": "ABCDEF1234"}

test("reads NAME=value lines, skips comments, short values and missing files", reads_local_files)


def skips_only_the_public_addresses():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "env")
        with open(path, "w") as f:
            f.write("KIT_SERVER_URL=https://live.example\nTEST_KIT_SERVER_URL=https://test.example\nTEST_ADMIN_SECRET=made-up-secret\n")
        return scan.local_values(path) == {"TEST_ADMIN_SECRET": "made-up-secret"}


test("the servers' public addresses aren't secrets; a test secret still is", skips_only_the_public_addresses)


def run_git(folder, *arguments):
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false", *arguments],
                   cwd=folder, check=True, capture_output=True)


def run_scan(folder):
    return subprocess.run([sys.executable, os.path.join(tools_folder, "scan_secrets.py")], cwd=folder, capture_output=True, text=True)


def make_repository_with_worktree(folder):
    """A made-up main checkout holding the real lib.sh, and a worktree of it. Returns (main checkout, worktree)."""
    main_checkout, worktree = os.path.join(folder, "main"), os.path.join(folder, "worktree")
    os.makedirs(os.path.join(main_checkout, "tools"))
    shutil.copy(os.path.join(tools_folder, "lib.sh"), os.path.join(main_checkout, "tools", "lib.sh"))
    cache = os.path.join(tools_folder, "test_cache.sh")
    if os.path.isfile(cache):
        shutil.copy(cache, os.path.join(main_checkout, "tools", "test_cache.sh"))
    with open(os.path.join(main_checkout, "tools", "probe.sh"), "w") as f:
        f.write('source "$(dirname "$0")/lib.sh"\n')
    with open(os.path.join(main_checkout, ".gitignore"), "w") as f:
        f.write("CLAUDE.local.md\n.env.local\n")
    run_git(main_checkout, "init", "-q")
    run_git(main_checkout, "add", ".")
    run_git(main_checkout, "commit", "-q", "-m", "start")
    run_git(main_checkout, "worktree", "add", "-q", "-b", "session", worktree)
    return main_checkout, worktree


def read_file(path):
    with open(path) as f:
        return f.read()


def write_setup_files(folder):
    for name, text in (("CLAUDE.local.md", "DEVICE=FAKE-DEVICE-1234\n"), (".env.local", "FAKE_SECRET=made-up-secret\n")):
        with open(os.path.join(folder, name), "w") as f:
            f.write(text)


def scan_fails_without_device_settings():
    with tempfile.TemporaryDirectory() as folder:
        main_checkout, _ = make_repository_with_worktree(folder)
        result = run_scan(main_checkout)
        return result.returncode == 1 and "CLAUDE.local.md is missing" in result.stdout


test("the scan fails when CLAUDE.local.md is missing", scan_fails_without_device_settings)


def tools_copy_setup_files_into_a_worktree():
    with tempfile.TemporaryDirectory() as folder:
        main_checkout, worktree = make_repository_with_worktree(folder)
        write_setup_files(main_checkout)
        subprocess.run(["bash", "tools/probe.sh"], cwd=worktree, check=True)
        return (read_file(os.path.join(worktree, "CLAUDE.local.md")) == "DEVICE=FAKE-DEVICE-1234\n"
                and read_file(os.path.join(worktree, ".env.local")) == "FAKE_SECRET=made-up-secret\n"
                and run_scan(worktree).returncode == 0)


test("any tool copies the setup files into a new worktree, and the scan then passes", tools_copy_setup_files_into_a_worktree)


def tools_keep_a_worktrees_own_setup_files():
    with tempfile.TemporaryDirectory() as folder:
        main_checkout, worktree = make_repository_with_worktree(folder)
        write_setup_files(main_checkout)
        with open(os.path.join(worktree, "CLAUDE.local.md"), "w") as f:
            f.write("DEVICE=WORKTREE-OWN-VALUE\n")
        subprocess.run(["bash", "tools/probe.sh"], cwd=worktree, check=True)
        return read_file(os.path.join(worktree, "CLAUDE.local.md")) == "DEVICE=WORKTREE-OWN-VALUE\n"


test("a setup file already in the worktree isn't overwritten", tools_keep_a_worktrees_own_setup_files)


def tree_scan_catches_a_tracked_key_without_device_settings():
    with tempfile.TemporaryDirectory() as folder:
        run_git(folder, "init", "-q")
        with open(os.path.join(folder, "notes.txt"), "w") as f:
            f.write("hello\n")
        run_git(folder, "add", "notes.txt")
        run_git(folder, "commit", "-q", "-m", "start")
        clean = subprocess.run([sys.executable, os.path.join(tools_folder, "scan_secrets.py"), "--all"],
                               cwd=folder, capture_output=True, text=True)
        with open(os.path.join(folder, "AuthKey_TEST.p8"), "w") as f:
            f.write("not really\n")
        with open(os.path.join(folder, "leak.txt"), "w") as f:
            f.write(fake_key + "\n")
        run_git(folder, "add", "-f", "AuthKey_TEST.p8", "leak.txt")
        run_git(folder, "commit", "-q", "-m", "leak")
        leaked = subprocess.run([sys.executable, os.path.join(tools_folder, "scan_secrets.py"), "--all"],
                                cwd=folder, capture_output=True, text=True)
        return (clean.returncode == 0 and leaked.returncode == 1
                and "a secret file name" in leaked.stdout and "AuthKey_TEST.p8" in leaked.stdout
                and "an OpenRouter key" in leaked.stdout and "leak.txt" in leaked.stdout
                and fake_key not in leaked.stdout)


test("a full-tree scan needs no CLAUDE.local.md, and catches a tracked key and a key file name",
     tree_scan_catches_a_tracked_key_without_device_settings)

print(f"FAILED {failures} of {count} tests" if failures else f"PASS {count} tests")
raise SystemExit(1 if failures else 0)
