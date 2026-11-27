#!/usr/bin/env python3
"""Read-only live status page for Mac QA: lanes, logs, Mac health, open PRs.

    python3 tools/qa_dashboard.py                    # serve on the LAN, port 8765
    python3 tools/qa_dashboard.py --install-launchagent [--dry-run]
    python3 tools/qa_dashboard.py --uninstall-launchagent

Same Wi-Fi only. Every request needs the private token from
~/.config/kit-qa-dashboard/token (made on first run, chmod 600) as ?token=,
the cookie set on the first valid visit, or an X-QA-Token header. The bookmark
URL is ~/.config/kit-qa-dashboard/url.txt (chmod 600). It is rewritten at
startup and about every 60 s with the current LAN address (ipconfig getifaddr
en0, otherwise the first non-loopback IPv4). The token is kept and is never
printed or logged. No controls: the page only reads.

Lane state comes from tools/lane.py's <lock dir>/meta/<pid>.json files; dead
pids are dropped. Python 3.9+, standard library only.
"""
import argparse
import hmac
import html
import http.cookies
import http.server
import json
import os
import plistlib
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))


def _toplevel():
    try:
        out = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.path.dirname(os.path.dirname(HERE))


ROOT = _toplevel()   # the project repo that vendors kit/
TITLE = os.environ.get("KIT_DASHBOARD_TITLE") or ("%s QA" % os.environ.get("KIT_PROJECT", "")).strip()
DEFAULT_PORT = 8765
COOKIE = "qa_token"
LABEL = "com.kit.qa-dashboard"
LOG_LIST_MAX = 40
TAIL_LINES = 200
TAIL_BYTES = 64 * 1024
HEALTH_EVERY_S = 10
PRS_EVERY_S = 30
URL_CHECK_EVERY_S = 60
CMD_TIMEOUT_S = 15
_INET_IPV4 = re.compile(r"(?:^|\s)inet\s+(\d{1,3}(?:\.\d{1,3}){3})\b")
_PLAIN_IPV4 = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")


# ---- config and token -------------------------------------------------------

def config_dir():
    return os.environ.get("KIT_QA_DASHBOARD_DIR") or os.path.expanduser("~/.config/kit-qa-dashboard")


def write_private(path, text):
    """Replace path atomically. Mode 600. A reader sees the old file or the new one."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", dir=directory)
    try:
        data = memoryview(text.encode())
        while data:
            wrote = os.write(fd, data)
            if wrote <= 0:
                raise OSError("short write")
            data = data[wrote:]
        os.fchmod(fd, 0o600)
        os.close(fd)
        fd = -1
        os.replace(tmp, path)
        tmp = None
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    os.chmod(path, 0o600)


def load_token(cdir):
    """The saved token, or a new one written with mode 600."""
    os.makedirs(cdir, mode=0o700, exist_ok=True)
    os.chmod(cdir, 0o700)
    path = os.path.join(cdir, "token")
    if os.path.exists(path):
        token = open(path).read().strip()
        if len(token) >= 24:
            os.chmod(path, 0o600)
            return token
    token = secrets.token_urlsafe(24)
    write_private(path, token + "\n")
    return token


def usable_ipv4(text):
    """A dotted IPv4 a phone can open: not loopback, not 0.0.0.0."""
    if not isinstance(text, str):
        return False
    parts = text.split(".")
    if len(parts) != 4:
        return False
    nums = []
    for part in parts:
        if not part.isdigit() or (len(part) > 1 and part.startswith("0")):
            return False
        num = int(part)
        if num > 255:
            return False
        nums.append(num)
    return nums[0] != 127 and nums != [0, 0, 0, 0]


def choose_lan_ip(en0, others):
    """Prefer en0. Otherwise the first non-loopback IPv4. Otherwise 127.0.0.1."""
    en0 = (en0 or "").strip()
    if usable_ipv4(en0):
        return en0
    for raw in others or ():
        ip = (raw or "").strip()
        if usable_ipv4(ip):
            return ip
    return "127.0.0.1"


def ipv4s_after_inet(text):
    """IPv4s from `ip addr` or ifconfig (`inet 1.2.3.4`), not netmasks or broadcasts."""
    return _INET_IPV4.findall(text or "")


def plain_ipv4s(text):
    return _PLAIN_IPV4.findall(text or "")


def _run(argv):
    return subprocess.run(argv, capture_output=True, text=True, timeout=5)


def _capture(run, argv):
    try:
        result = run(argv)
    except (OSError, subprocess.SubprocessError):
        return None
    return getattr(result, "stdout", "") or ""


def en0_via_ipconfig(run):
    """`ipconfig getifaddr en0`, then /usr/sbin/ipconfig when it is off PATH."""
    for binary in ("ipconfig", "/usr/sbin/ipconfig"):
        text = _capture(run, [binary, "getifaddr", "en0"])
        if text is None:
            continue
        return text.strip()
    return ""


def candidate_ipv4s(run):
    """Interface addresses, first command that has a usable IPv4."""
    for argv in (["ip", "-4", "-o", "addr"], ["ifconfig"], ["/usr/sbin/ifconfig"], ["hostname", "-I"]):
        text = _capture(run, argv)
        if text is None:
            continue
        found = plain_ipv4s(text) if argv[0] == "hostname" else ipv4s_after_inet(text)
        if any(usable_ipv4(ip) for ip in found):
            return found
    return []


def lan_ip(run=None):
    """This Mac's Wi-Fi address. en0, not the default route: a VPN makes the route a dead 10.x address."""
    run = run or _run
    en0 = en0_via_ipconfig(run)
    if usable_ipv4(en0):
        return en0
    return choose_lan_ip("", candidate_ipv4s(run))


def bookmark_text(ip, port, token):
    return "http://%s:%d/?token=%s\n" % (ip, int(port), token)


def write_url_file(cdir, host, port, token, ip=None):
    """Write url.txt when the bookmark changed. Same token, atomic, mode 600.
    Bound to one address, that address is the host. Bound to every interface, the host is the LAN address."""
    os.makedirs(cdir, mode=0o700, exist_ok=True)
    os.chmod(cdir, 0o700)
    if ip is None:
        ip = lan_ip() if host in ("0.0.0.0", "") else host
    path = os.path.join(cdir, "url.txt")
    wanted = bookmark_text(ip, port, token)
    try:
        with open(path) as fh:
            current = fh.read()
    except OSError:
        current = None
    if current == wanted:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return path
    write_private(path, wanted)
    return path


def url_refresh_loop(cdir, host, port, token, stop, every=URL_CHECK_EVERY_S, ip_fn=None):
    """Rewrite url.txt when the LAN address changes. `stop` is a threading.Event.
    `token` is the one already saved; this does not mint a new one. Failures stay quiet (no token in a log)."""
    while not stop.wait(every):
        if host not in ("0.0.0.0", ""):
            continue
        try:
            found = ip_fn() if ip_fn else None
            write_url_file(cdir, host, port, token, ip=found)
        except Exception:
            pass


def lane_dir():
    path = os.environ.get("KIT_LANE_DIR")
    if path:
        return path
    tmp = os.environ.get("TMPDIR")
    if not tmp and sys.platform == "darwin":   # launchd agents may not get TMPDIR
        try:
            tmp = subprocess.run(["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            tmp = None
    return os.path.join(tmp or "/tmp", "kit-lanes")


def log_dir():
    return os.environ.get("KIT_QA_LOG_DIR") or os.path.expanduser("~/qa-logs")


# ---- scrubbing --------------------------------------------------------------
# Patterns and scrub() live in tools/scrub_report.py so reports and the
# dashboard hide the same secrets.

import scrub_report


def local_secret_values():
    """Values from the git-ignored CLAUDE.local.md / .env.local (DEVICE, TEAM, keys), via scan_secrets."""
    roots = [ROOT]
    try:
        common = subprocess.run(["git", "-C", ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        if common:
            roots.append(os.path.dirname(common))
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        sys.path.insert(0, HERE)
        import scan_secrets
        values = {}
        for r in roots:
            values.update(scan_secrets.local_values(os.path.join(r, "CLAUDE.local.md"), os.path.join(r, ".env.local")))
        return sorted(set(values.values()), key=len, reverse=True)
    except Exception:
        return []


LOCAL_VALUES = []


def scrub(text, extra=None):
    return scrub_report.scrub(text, extra if extra is not None else LOCAL_VALUES)


# ---- collectors -------------------------------------------------------------


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False
    return True


def light_slot_count(ldir):
    raw = os.environ.get("KIT_LANE_LIGHT_SLOTS")
    if raw and raw.isdigit() and int(raw) > 0:
        return int(raw)
    try:
        n = len([f for f in os.listdir(ldir) if re.fullmatch(r"light-\d+", f)])
    except OSError:
        n = 0
    return max(n, 3)


def lanes(ldir, now=None):
    now = now or time.time()
    out = {"heavy": {"slots": 1, "holders": [], "waiters": []},
           "light": {"slots": light_slot_count(ldir), "holders": [], "waiters": []}}
    mdir = os.path.join(ldir, "meta")
    try:
        names = sorted(f for f in os.listdir(mdir) if f.endswith(".json") and not f.startswith("."))
    except OSError:
        names = []
    for name in names:
        try:
            with open(os.path.join(mdir, name)) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict) or rec.get("lane") not in out or not pid_alive(rec.get("pid")):
            continue
        try:
            started, queued = float(rec.get("started_at") or 0), float(rec.get("queued_at") or now)
        except (TypeError, ValueError):
            continue
        cwd = rec.get("cwd") or ""
        row = {"pid": rec.get("pid"), "command": scrub(str(rec.get("command") or "")), "cwd": cwd,
               "clone": os.path.basename(cwd.rstrip("/")) or cwd, "branch": rec.get("branch"), "slot": rec.get("slot")}
        if rec.get("state") == "running" and started:
            row["elapsed_s"] = round(now - started, 1)
            out[rec["lane"]]["holders"].append(row)
        else:
            row["waiting_s"] = round(now - queued, 1)
            out[rec["lane"]]["waiters"].append(row)
    for lane in out.values():
        lane["holders"].sort(key=lambda r: -r["elapsed_s"])
        lane["waiters"].sort(key=lambda r: -r["waiting_s"])
    return out


def list_logs(ldir):
    try:
        entries = [e for e in os.scandir(ldir) if e.is_file() and not e.name.startswith(".")]
    except OSError:
        return []
    entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
    return [{"name": e.name, "size": e.stat().st_size, "mtime": e.stat().st_mtime} for e in entries[:LOG_LIST_MAX]]


def tail_log(ldir, name, lines=TAIL_LINES):
    """Last lines of one top-level file in ldir, scrubbed; None if the name isn't one."""
    if not name or name != os.path.basename(name) or name.startswith("."):
        return None
    path = os.path.join(ldir, name)
    if not os.path.isfile(path) or os.path.islink(path):
        return None
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - TAIL_BYTES))
        data = fh.read()
    text = data.decode("utf-8", errors="replace")
    if size > TAIL_BYTES:
        text = text.split("\n", 1)[-1]
    return scrub("\n".join(text.splitlines()[-lines:]))


def run(cmd, timeout=CMD_TIMEOUT_S, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)


def swap_usage():
    if sys.platform == "darwin":
        out = run(["sysctl", "-n", "vm.swapusage"]).stdout
        vals = dict(re.findall(r"(total|used|free)\s*=\s*([\d.]+)M", out))
        return {k: float(v) for k, v in vals.items()} or None
    try:
        info = dict(re.findall(r"(Swap\w+):\s+(\d+) kB", open("/proc/meminfo").read()))
        total, free = int(info["SwapTotal"]) / 1024, int(info["SwapFree"]) / 1024
        return {"total": round(total, 1), "used": round(total - free, 1), "free": round(free, 1)}
    except (OSError, KeyError, ValueError):
        return None


def booted_simulators():
    """Booted iOS simulators, or None on a machine without Xcode (non-iOS projects)."""
    if shutil.which("xcrun") is None:
        return None
    out = run(["xcrun", "simctl", "list", "devices", "booted", "-j"])
    if out.returncode != 0:
        raise RuntimeError("simctl exited %d" % out.returncode)
    sims = []
    for runtime, devices in json.loads(out.stdout).get("devices", {}).items():
        for d in devices:
            if d.get("state") == "Booted":
                sims.append({"name": d.get("name"), "runtime": runtime.split(".")[-1].replace("-", " ", 1).replace("-", ".")})
    return sims


def health():
    h = {"updated_at": time.time(), "cpus": os.cpu_count(), "load": None, "swap": None, "simulators": None, "errors": []}
    try:
        h["load"] = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        h["errors"].append("load average unavailable")
    try:
        h["swap"] = swap_usage()
    except (OSError, subprocess.SubprocessError) as e:
        h["errors"].append("swap: %s" % type(e).__name__)
    try:
        h["simulators"] = booted_simulators()
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as e:
        h["errors"].append("simulators: %s" % (e if isinstance(e, RuntimeError) else type(e).__name__))
    return h


def checks_summary(rollup):
    s = {"pass": 0, "fail": 0, "pending": 0}
    for c in rollup or []:
        state = (c.get("conclusion") or c.get("state") or c.get("status") or "").upper()
        if state in ("SUCCESS", "NEUTRAL", "SKIPPED"):
            s["pass"] += 1
        elif state in ("FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE"):
            s["fail"] += 1
        else:
            s["pending"] += 1
    return s


def open_prs():
    fields = "number,title,headRefName,author,isDraft,updatedAt,url,reviewDecision,statusCheckRollup"
    out = run(["gh", "pr", "list", "--state", "open", "--limit", "30", "--json", fields], timeout=30, cwd=ROOT)
    if out.returncode != 0:
        raise RuntimeError("gh exited %d" % out.returncode)
    prs = []
    for p in json.loads(out.stdout):
        prs.append({"number": p.get("number"), "title": scrub(p.get("title") or ""), "branch": p.get("headRefName"),
                    "author": (p.get("author") or {}).get("login"), "draft": p.get("isDraft"),
                    "updated_at": p.get("updatedAt"), "url": p.get("url"), "review": p.get("reviewDecision"),
                    "checks": checks_summary(p.get("statusCheckRollup"))})
    return prs


class Cache:
    """Slow facts (health, PRs) refreshed in background threads so /api/status stays fast."""

    def __init__(self):
        self.lock = threading.Lock()
        self.health = {"updated_at": None, "load": None, "swap": None, "simulators": None, "errors": ["starting"]}
        self.prs = {"items": [], "updated_at": None, "error": "starting"}

    def refresh_health(self):
        h = health()
        with self.lock:
            self.health = h

    def refresh_prs(self):
        try:
            p = {"items": open_prs(), "updated_at": time.time(), "error": None}
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as e:
            with self.lock:
                p = dict(self.prs, error="gh pr list failed: %s" % (e if isinstance(e, RuntimeError) else type(e).__name__))
        with self.lock:
            self.prs = p

    def start(self):
        for fn, every in ((self.refresh_health, HEALTH_EVERY_S), (self.refresh_prs, PRS_EVERY_S)):
            threading.Thread(target=self._loop, args=(fn, every), daemon=True).start()

    @staticmethod
    def _loop(fn, every):
        while True:
            try:
                fn()
            except Exception:
                pass
            time.sleep(every)

    def snapshot(self):
        with self.lock:
            return dict(self.health), dict(self.prs)


def status(ldir, qa_logs, cache, selected=None):
    files = list_logs(qa_logs)
    names = {f["name"] for f in files}
    if selected not in names:
        selected = files[0]["name"] if files else None
    h, p = cache.snapshot()
    return {"generated_at": time.time(), "host": socket.gethostname(), "lane_dir": ldir, "lanes": lanes(ldir),
            "logs": {"dir": qa_logs, "files": files, "selected": selected,
                     "tail": tail_log(qa_logs, selected) if selected else None},
            "health": h, "prs": p}


# ---- HTTP -------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "kit-qa-dashboard"
    sys_version = ""

    def log_message(self, fmt, *args):   # never log the query string (it may hold the token)
        path = urllib.parse.urlsplit(getattr(self, "path", "") or "").path
        code = args[1] if fmt.startswith('"%s"') and len(args) > 1 else (args[0] if args else "")
        sys.stderr.write("%s %s %s\n" % (getattr(self, "command", None) or "-", path, code))

    def token_ok(self, query):
        token = self.server.token
        candidates = query.get("token", []) + [self.headers.get("X-QA-Token", "")]
        jar = http.cookies.SimpleCookie()
        try:
            jar.load(self.headers.get("Cookie", ""))
        except http.cookies.CookieError:
            pass
        if COOKIE in jar:
            candidates.append(jar[COOKIE].value)
        return any(c and hmac.compare_digest(c.encode(), token.encode()) for c in candidates)

    def send(self, code, body, ctype="text/plain; charset=utf-8", headers=()):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(url.query)
        if not self.token_ok(query):
            return self.send(401, "401: open the bookmark URL from ~/.config/kit-qa-dashboard/url.txt\n")
        if "token" in query:   # first visit: keep the token in an HttpOnly cookie and drop it from the address bar
            cookie = "%s=%s; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000" % (COOKIE, self.server.token)
            return self.send(302, "", headers=(("Location", url.path or "/"), ("Set-Cookie", cookie)))
        if url.path == "/":
            return self.send(200, PAGE.replace("__TITLE__", html.escape(TITLE or "QA")), "text/html; charset=utf-8")
        if url.path == "/api/status":
            body = status(self.server.lane_dir, self.server.log_dir, self.server.cache, (query.get("log") or [None])[0])
            return self.send(200, json.dumps(body), "application/json")
        return self.send(404, "not found\n")

    do_HEAD = do_GET

    def do_POST(self):
        self.send(405, "read-only\n")

    do_PUT = do_DELETE = do_PATCH = do_POST


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(host, port, token, ldir, qa_logs, cache):
    srv = Server((host, port), Handler)
    srv.token, srv.lane_dir, srv.log_dir, srv.cache = token, ldir, qa_logs, cache
    return srv


# ---- LaunchAgent ------------------------------------------------------------

def plist_path():
    return os.path.expanduser("~/Library/LaunchAgents/%s.plist" % LABEL)


def launchagent_plist(port, ldir, qa_logs):
    """The agent definition. Holds paths only, never the token."""
    path_env = ":".join([os.path.expanduser("~/.local/bin"), "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin",
                         "/usr/sbin", "/sbin"])
    logfile = os.path.expanduser("~/Library/Logs/kit-qa-dashboard.log")
    return {"Label": LABEL,
            "ProgramArguments": ["/usr/bin/python3", os.path.abspath(__file__), "--port", str(port)],
            "WorkingDirectory": ROOT, "RunAtLoad": True, "KeepAlive": True, "ProcessType": "Background",
            "EnvironmentVariables": {"PATH": path_env, "KIT_LANE_DIR": ldir, "KIT_QA_LOG_DIR": qa_logs},
            "StandardOutPath": logfile, "StandardErrorPath": logfile}


def install_launchagent(port, dry_run):
    data = plistlib.dumps(launchagent_plist(port, lane_dir(), log_dir()))
    if dry_run:
        sys.stdout.write(data.decode())
        return 0
    os.makedirs(os.path.dirname(plist_path()), exist_ok=True)
    with open(plist_path(), "wb") as fh:
        fh.write(data)
    domain = "gui/%d" % os.getuid()
    subprocess.run(["launchctl", "bootout", domain + "/" + LABEL], capture_output=True)
    r = subprocess.run(["launchctl", "bootstrap", domain, plist_path()])
    print("installed %s; bookmark URL in %s" % (plist_path(), os.path.join(config_dir(), "url.txt")))
    return r.returncode


def uninstall_launchagent():
    subprocess.run(["launchctl", "bootout", "gui/%d/%s" % (os.getuid(), LABEL)], capture_output=True)
    if os.path.exists(plist_path()):
        os.unlink(plist_path())
    print("removed %s" % plist_path())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only Mac QA status page (same Wi-Fi, token).")
    ap.add_argument("--host", default="0.0.0.0", help="bind address (default all interfaces, for the phone)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--install-launchagent", action="store_true")
    ap.add_argument("--uninstall-launchagent", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="with --install-launchagent: print the plist only")
    args = ap.parse_args(argv)
    if args.install_launchagent:
        return install_launchagent(args.port, args.dry_run)
    if args.uninstall_launchagent:
        return uninstall_launchagent()
    global LOCAL_VALUES
    LOCAL_VALUES = local_secret_values()
    cdir = config_dir()
    token = load_token(cdir)
    cache = Cache()
    cache.start()
    srv = make_server(args.host, args.port, token, lane_dir(), log_dir(), cache)
    port = srv.server_address[1]
    url_file = write_url_file(cdir, args.host, port, token)
    stop_refresh = threading.Event()
    threading.Thread(target=url_refresh_loop, args=(cdir, args.host, port, token, stop_refresh),
                     daemon=True, name="qa-url-refresh").start()
    print("qa dashboard listening on %s:%d; bookmark URL in %s" % (args.host, port, url_file), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_refresh.set()
    return 0


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#0f1115;--card:#181b22;--fg:#e8e8ea;--dim:#8b8f98;--ok:#3ecf8e;--warn:#f5a524;--bad:#f0565b;--acc:#6aa9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 -apple-system,system-ui,sans-serif;
padding:max(12px,env(safe-area-inset-top)) max(12px,env(safe-area-inset-right)) 24px max(12px,env(safe-area-inset-left))}
h1{font-size:18px;margin:0 0 8px;display:flex;justify-content:space-between;align-items:center}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:0 0 8px}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.card{background:var(--card);border-radius:12px;padding:12px;overflow:hidden}
.row{padding:6px 0;border-top:1px solid #262a33}.row:first-of-type{border-top:0}
.cmd{font:12px ui-monospace,Menlo,monospace;word-break:break-all}.dim{color:var(--dim);font-size:12px}
.pill{display:inline-block;border-radius:999px;padding:1px 8px;font-size:12px;margin-right:4px;background:#262a33}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}a{color:var(--acc);text-decoration:none}
pre{margin:8px 0 0;max-height:55vh;overflow:auto;background:#0b0d10;border-radius:8px;padding:8px;font:11px/1.35 ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-all}
select{width:100%;font-size:15px;background:#0b0d10;color:var(--fg);border:1px solid #262a33;border-radius:8px;padding:6px}
#stale{font-size:12px}
</style></head><body>
<h1><span>__TITLE__</span><span id="stale" class="dim">loading…</span></h1>
<div class="grid">
<div class="card"><h2>Heavy lane</h2><div id="heavy"></div></div>
<div class="card"><h2>Light lane</h2><div id="light"></div></div>
<div class="card"><h2>Mac health</h2><div id="health"></div></div>
<div class="card"><h2>Open PRs</h2><div id="prs"></div></div>
<div class="card" style="grid-column:1/-1"><h2>Logs</h2><select id="logsel"></select><pre id="tail"></pre></div>
</div>
<script>
"use strict";
let chosen = null, pinned = false;
const $ = id => document.getElementById(id);
function el(tag, cls, text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;}
function dur(s){s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),r=s%60;return h?`${h}h ${m}m`:m?`${m}m ${r}s`:`${r}s`;}
function ago(t){return t?dur(Date.now()/1000-t)+" ago":"never";}
function lane(box, l){
  box.replaceChildren();
  const used=l.holders.length;
  box.append(el("div","",`${used}/${l.slots} busy · ${l.waiters.length} waiting`));
  for(const h of l.holders){const r=el("div","row");r.append(el("div","cmd",h.command));
    r.append(el("div","dim",`▶ ${dur(h.elapsed_s)} · ${h.clone}${h.branch?" @ "+h.branch:""} · pid ${h.pid}${h.slot&&h.slot!=="heavy"?" · "+h.slot:""}`));box.append(r);}
  for(const w of l.waiters){const r=el("div","row");r.append(el("div","cmd",w.command));
    r.append(el("div","dim warn",`⏳ waiting ${dur(w.waiting_s)} · ${w.clone}${w.branch?" @ "+w.branch:""} · pid ${w.pid}`));box.append(r);}
}
function health(box,h){
  box.replaceChildren();
  if(h.load){const per=h.load[0]/(h.cpus||1);box.append(el("div",per>1.5?"bad":per>1?"warn":"ok",`Load ${h.load.join(" / ")} (${h.cpus} cores)`));}
  if(h.swap){const u=h.swap.used||0;box.append(el("div",u>4096?"bad":u>1024?"warn":"",`Swap ${Math.round(u)} MB used of ${Math.round(h.swap.total||0)} MB`));}
  const sims=h.simulators;
  box.append(el("div","",sims==null?"Simulators: unknown":`Booted simulators: ${sims.length}`));
  for(const s of sims||[])box.append(el("div","dim",`• ${s.name} (${s.runtime})`));
  for(const e of h.errors||[])box.append(el("div","dim warn",e));
  box.append(el("div","dim",`updated ${ago(h.updated_at)}`));
}
function prs(box,p){
  box.replaceChildren();
  if(!p.items.length)box.append(el("div","dim",p.error?"":"No open PRs"));
  for(const x of p.items){const r=el("div","row");const a=el("a","",`#${x.number} ${x.title}`);a.href=x.url;a.target="_blank";a.rel="noreferrer";r.append(a);
    const c=x.checks,meta=el("div","dim");
    meta.append(el("span","pill "+(c.fail?"bad":c.pending?"warn":"ok"),c.fail?`${c.fail} failing`:c.pending?`${c.pending} pending`:`${c.pass} passed`));
    if(x.draft)meta.append(el("span","pill","draft"));if(x.review)meta.append(el("span","pill",x.review.toLowerCase().replace("_"," ")));
    meta.append(document.createTextNode(`${x.branch} · ${x.author||""}`));r.append(meta);box.append(r);}
  if(p.error)box.append(el("div","dim warn",p.error));
  box.append(el("div","dim",`updated ${ago(p.updated_at)}`));
}
function logs(l){
  const sel=$("logsel"),names=l.files.map(f=>f.name);
  if(sel.dataset.names!==names.join("\\n")){sel.replaceChildren();
    for(const f of l.files){const o=el("option","",`${f.name} · ${ago(f.mtime)}`);o.value=f.name;sel.append(o);}sel.dataset.names=names.join("\\n");}
  sel.value=l.selected||"";
  const pre=$("tail"),atBottom=pre.scrollHeight-pre.scrollTop-pre.clientHeight<40;
  pre.textContent=l.tail==null?(l.files.length?"":"No logs in "+l.dir):l.tail;
  if(atBottom)pre.scrollTop=pre.scrollHeight;
}
$("logsel").addEventListener("change",e=>{chosen=e.target.value;pinned=true;tick();});
async function tick(){
  try{
    const r=await fetch("/api/status"+(pinned&&chosen?"?log="+encodeURIComponent(chosen):""),{credentials:"same-origin",cache:"no-store"});
    if(r.status===401){$("stale").textContent="token needed: open the bookmark URL";$("stale").className="bad";return;}
    const s=await r.json();
    lane($("heavy"),s.lanes.heavy);lane($("light"),s.lanes.light);health($("health"),s.health);prs($("prs"),s.prs);logs(s.logs);
    $("stale").textContent=`${s.host} · ${new Date(s.generated_at*1000).toLocaleTimeString()}`;$("stale").className="dim";
  }catch(e){$("stale").textContent="offline (retrying)";$("stale").className="bad";}
}
tick();setInterval(tick,2000);
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
