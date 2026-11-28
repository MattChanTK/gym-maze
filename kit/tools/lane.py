#!/usr/bin/env python3
"""Hold a heavy (1) or light (N) run-queue slot, then run the command.

The slot is fcntl.flock on an open file, so the kernel drops it when this
process exits, crashes, or is kill -9'd. No pid files.

Env:
  KIT_LANE_DIR          lock directory (default: ${TMPDIR}/kit-lanes)
  KIT_LANE_LIGHT_SLOTS  light slot count (default 3)
  KIT_LANE_RANK         inherited rank: 1=light, 2=heavy

Best-effort status for tools/qa_dashboard.py: <lock dir>/meta/<pid>.json while
waiting or holding, removed at exit. Errors never touch the lock; a kill -9
leaves a stale file that the dashboard drops (dead pid).
"""
import atexit
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

RANKS = {"light": 1, "heavy": 2}
WAIT_NOTICE_S = 5
LIGHT_POLL_S = 0.2


def die(msg, code=2):
    print(f"lane.sh: {msg}", file=sys.stderr)
    raise SystemExit(code)


def held_rank():
    raw = os.environ.get("KIT_LANE_RANK", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def lock_dir():
    path = os.environ.get("KIT_LANE_DIR")
    if not path:
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "kit-lanes")
    os.makedirs(path, exist_ok=True)
    return path


def light_slots():
    raw = os.environ.get("KIT_LANE_LIGHT_SLOTS", "3")
    try:
        n = int(raw)
    except ValueError:
        die("KIT_LANE_LIGHT_SLOTS must be a positive integer")
    if n < 1:
        die("KIT_LANE_LIGHT_SLOTS must be a positive integer")
    return n


META = {"pid": os.getpid(), "queued_at": time.time(), "started_at": None, "slot": None}


def started_by(env):
    """Who started this run, for the QA dashboard: KIT_LANE_OWNER if set, a GitHub Actions job (repo, PR or
    branch, workflow / job, run link), a Claude Code session (short id), or a terminal."""
    owner = env.get("KIT_LANE_OWNER")
    if owner:
        return {"by": owner, "by_url": None}
    if env.get("GITHUB_ACTIONS") == "true":
        repo = env.get("GITHUB_REPOSITORY", "")
        ref = env.get("GITHUB_REF", "")
        pr = re.match(r"refs/pull/(\d+)/", ref)
        where = "PR #" + pr.group(1) if pr else (env.get("GITHUB_REF_NAME") or ref or "?")
        url = None
        if repo and env.get("GITHUB_RUN_ID"):
            url = "%s/%s/actions/runs/%s" % (env.get("GITHUB_SERVER_URL", "https://github.com"), repo, env["GITHUB_RUN_ID"])
        return {"by": "GitHub %s %s · %s / %s" % (repo.split("/")[-1], where, env.get("GITHUB_WORKFLOW", "?"),
                                                  env.get("GITHUB_JOB", "?")), "by_url": url}
    if env.get("CLAUDECODE"):
        sid = (env.get("CLAUDE_CODE_SESSION_ID") or "")[:8]
        return {"by": "Claude session" + (" " + sid if sid else ""), "by_url": None}
    return {"by": "terminal", "by_url": None}


def write_meta(**fields):
    META.update(fields)
    try:
        d = os.path.join(lock_dir(), "meta")
        os.makedirs(d, exist_ok=True)
        path, tmp = os.path.join(d, "%d.json" % os.getpid()), os.path.join(d, ".%d.tmp" % os.getpid())
        if "branch" not in META:
            META["branch"] = None
            atexit.register(clear_meta, path)
            try:
                git = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=2)
                META["branch"] = git.stdout.strip() if git.returncode == 0 else None
            except (OSError, subprocess.SubprocessError):
                pass
        with open(tmp, "w") as fh:
            json.dump(META, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def clear_meta(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def announce_while_waiting(name, stop):
    print(f"waiting for {name} lane", file=sys.stderr, flush=True)
    while not stop.wait(WAIT_NOTICE_S):
        print(f"waiting for {name} lane", file=sys.stderr, flush=True)


def try_lock(fh):
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def run_command(cmd):
    # Keep the lock fd in this process (CLOEXEC, not inherited). A child like
    # sleep or xcodebuild then cannot pin the slot after this process is killed.
    proc = subprocess.Popen(cmd)
    def forward(signum, _frame):
        if proc.poll() is None:
            proc.send_signal(signum)
    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    raise SystemExit(proc.wait())


def acquire_heavy():
    fh = open(os.path.join(lock_dir(), "heavy"), "a+")
    if try_lock(fh):
        write_meta(state="running", slot="heavy", started_at=time.time())
        return [fh]
    write_meta(state="waiting")
    stop = threading.Event()
    threading.Thread(target=announce_while_waiting, args=("heavy", stop), daemon=True).start()
    fcntl.flock(fh, fcntl.LOCK_EX)
    stop.set()
    write_meta(state="running", slot="heavy", started_at=time.time())
    return [fh]


def acquire_light():
    n = light_slots()
    d = lock_dir()
    files = [open(os.path.join(d, "light-%d" % i), "a+") for i in range(n)]

    def any_slot():
        for fh in files:
            if try_lock(fh):
                return fh
        return None

    got = any_slot()
    if got is None:
        write_meta(state="waiting")
        stop = threading.Event()
        threading.Thread(target=announce_while_waiting, args=("light", stop), daemon=True).start()
        while True:
            time.sleep(LIGHT_POLL_S)
            got = any_slot()
            if got is not None:
                stop.set()
                break
    write_meta(state="running", slot=os.path.basename(got.name), started_at=time.time())
    for fh in files:
        if fh is not got:
            fh.close()
    return [got]


def parse_args(argv):
    usage = "usage: tools/lane.sh heavy|light [--] command [args…]"
    if not argv or argv[0] in ("-h", "--help"):
        die(usage)
    kind = argv[0]
    if kind not in RANKS:
        die(usage)
    rest = argv[1:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        die(usage)
    return kind, rest


def main(argv):
    kind, cmd = parse_args(argv)
    rank = RANKS[kind]
    have = held_rank()
    try:
        if have < rank:
            META.update(lane=kind, command=" ".join(cmd), cwd=os.getcwd(), **started_by(os.environ))
            held = acquire_heavy() if kind == "heavy" else acquire_light()
            globals()["_LANE_FILES"] = held
            os.environ["KIT_LANE_RANK"] = str(rank)
            run_command(cmd)
        else:
            os.execvp(cmd[0], cmd)
    except OSError as exc:
        die("could not exec %s: %s" % (cmd[0], exc), 127)


if __name__ == "__main__":
    main(sys.argv[1:])
