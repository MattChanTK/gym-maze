"""Tests for tools/qa_dashboard.py and lane.py's status files (no network, gh, or simulator).
Run by tools/test_py.sh; prints failures and one summary line."""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qa_dashboard as qd

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


tmp = tempfile.mkdtemp(prefix="qa-dash-test-")
LANES = os.path.join(tmp, "lanes")
LOGS = os.path.join(tmp, "logs")
CONF = os.path.join(tmp, "conf")
os.makedirs(os.path.join(LANES, "meta"))
os.makedirs(LOGS)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


opener = urllib.request.build_opener(NoRedirect)


def get(path, headers=None):
    req = urllib.request.Request(BASE + path, headers=headers or {})
    try:
        with opener.open(req, timeout=5) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def write_meta(pid, lane, state, **extra):
    rec = {"pid": pid, "lane": lane, "state": state, "command": "tools/test.sh", "cwd": "/Users/x/qa-clones/c1",
           "branch": "b1", "queued_at": time.time() - 30, "started_at": time.time() - 10 if state == "running" else None}
    rec.update(extra)
    with open(os.path.join(LANES, "meta", "%d.json" % pid), "w") as fh:
        json.dump(rec, fh)


# ---- token files ----
os.environ["KIT_QA_DASHBOARD_DIR"] = CONF
TOKEN = qd.load_token(CONF)
mode = lambda p: stat.S_IMODE(os.stat(p).st_mode)
test("token is long and random", lambda: len(TOKEN) >= 32 and TOKEN != qd.load_token(tempfile.mkdtemp(dir=tmp)))
test("token file is 600 and reused", lambda: mode(os.path.join(CONF, "token")) == 0o600 and qd.load_token(CONF) == TOKEN)
url_file = qd.write_url_file(CONF, "0.0.0.0", 8765, TOKEN)
test("url.txt is 600 and holds the bookmark",
     lambda: mode(url_file) == 0o600 and open(url_file).read().strip().endswith(":8765/?token=" + TOKEN))


class _Cmd:
    def __init__(self, out="", code=0):
        self.stdout, self.returncode, self.stderr = out, code, ""


def test_choose_lan_ip():
    vpn = "10.14.0.2"
    wifi = "192.168.1.9"
    return (qd.choose_lan_ip(wifi, [vpn, "127.0.0.1"]) == wifi
            and qd.choose_lan_ip("", ["127.0.0.1", vpn, "192.168.1.5"]) == vpn
            and qd.choose_lan_ip("127.0.0.1", ["192.168.1.5"]) == "192.168.1.5"
            and qd.choose_lan_ip("not-an-ip", ["0.0.0.0", "127.0.0.1"]) == "127.0.0.1"
            and qd.usable_ipv4("192.168.0.1") and not qd.usable_ipv4("127.0.0.1")
            and not qd.usable_ipv4("10.14.0.2\n"))


def test_en0_beats_route():
    calls = []

    def run(argv):
        calls.append(list(argv))
        if argv == ["ipconfig", "getifaddr", "en0"]:
            return _Cmd("192.168.1.9\n")
        if argv[:2] == ["ip", "-4"]:
            return _Cmd("2: utun0 inet 10.14.0.2/32\n")
        raise AssertionError("unexpected")

    return qd.lan_ip(run=run) == "192.168.1.9" and calls == [["ipconfig", "getifaddr", "en0"]]


def test_fallback_first_non_loopback():
    sample = "lo0: flags= inet 127.0.0.1\nutun0: flags= inet 10.14.0.2 netmask 0xffffffff\n"

    def run(argv):
        if argv[0] in ("ipconfig", "/usr/sbin/ipconfig"):
            raise FileNotFoundError(argv[0])
        if argv == ["ip", "-4", "-o", "addr"]:
            raise FileNotFoundError("ip")
        if argv == ["ifconfig"]:
            return _Cmd(sample)
        raise AssertionError(argv)

    parsed = qd.ipv4s_after_inet(sample + "broadcast 10.14.0.255\n")
    return (parsed == ["127.0.0.1", "10.14.0.2"] and "255.255.255.0" not in parsed
            and qd.lan_ip(run=run) == "10.14.0.2")


def test_rewrite_keeps_token():
    d = os.path.join(tmp, "rewrite")
    os.makedirs(d)
    kept = "kept-token-value"
    path = os.path.join(d, "url.txt")
    qd.write_private(path, "http://10.14.0.2:8765/?token=%s\n" % kept)
    before = os.stat(path).st_ino
    qd.write_url_file(d, "0.0.0.0", 8765, kept, ip="192.168.1.40")
    body = open(path).read()
    changed = os.stat(path).st_ino
    names = os.listdir(d)
    qd.write_url_file(d, "0.0.0.0", 8765, kept, ip="192.168.1.40")
    same = os.stat(path).st_ino
    pinned = qd.write_url_file(d, "127.0.0.1", 9, kept)
    return (body == "http://192.168.1.40:8765/?token=%s\n" % kept
            and mode(path) == 0o600 and changed != before and same == changed
            and all(not name.startswith(".url.txt.") for name in names)
            and open(pinned).read() == "http://127.0.0.1:9/?token=%s\n" % kept)


def test_refresh_loop_follows_ip():
    d = os.path.join(tmp, "refresh-loop")
    os.makedirs(d)
    kept = "kept-token-value"
    qd.write_private(os.path.join(d, "url.txt"), "http://10.14.0.2:8765/?token=%s\n" % kept)
    stop = threading.Event()
    thread = threading.Thread(target=qd.url_refresh_loop,
                              args=(d, "0.0.0.0", 8765, kept, stop),
                              kwargs={"every": 0.05, "ip_fn": lambda: "192.168.1.41"},
                              daemon=True)
    thread.start()
    deadline = time.time() + 2
    body = ""
    while time.time() < deadline:
        body = open(os.path.join(d, "url.txt")).read()
        if body.startswith("http://192.168.1.41:"):
            break
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=2)
    return (body == "http://192.168.1.41:8765/?token=%s\n" % kept
            and "10.14.0.2" not in body and thread.is_alive() is False)


test("en0 is chosen over the VPN route", test_en0_beats_route)
test("no en0: first non-loopback IPv4", test_fallback_first_non_loopback)
test("choose_lan_ip skips loopback and 0.0.0.0", test_choose_lan_ip)
test("rewrite swaps the host, keeps the token, mode 600", test_rewrite_keeps_token)
test("refresh loop rewrites a stale host", test_refresh_loop_follows_ip)

# ---- server ----
cache = qd.Cache()   # collectors not started: no gh or simctl in tests
cache.health = {"updated_at": 1.0, "cpus": 8, "load": [1.0, 1.0, 1.0], "swap": None, "simulators": [], "errors": []}
qd.Handler.log_message = lambda *args: None   # quiet in-process server; the subprocess check reads real logs
srv = qd.make_server("127.0.0.1", 0, TOKEN, LANES, LOGS, cache)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = "http://127.0.0.1:%d" % srv.server_address[1]

test("401 on / without token", lambda: get("/")[0] == 401)
test("401 on /api/status without token", lambda: get("/api/status")[0] == 401)
test("401 with a wrong token", lambda: get("/api/status", {"X-QA-Token": "nope"})[0] == 401
     and get("/?token=nope")[0] == 401)
test("200 with X-QA-Token header", lambda: get("/api/status", {"X-QA-Token": TOKEN})[0] == 200)


def first_visit_sets_cookie():
    code, headers, _ = get("/?token=" + TOKEN)
    cookie = headers.get("Set-Cookie", "")
    return code == 302 and headers.get("Location") == "/" and "HttpOnly" in cookie and TOKEN in cookie


test("?token sets an HttpOnly cookie and drops the token from the URL", first_visit_sets_cookie)
test("cookie alone works for page and API",
     lambda: get("/", {"Cookie": "qa_token=" + TOKEN})[0] == 200 and get("/api/status", {"Cookie": "qa_token=" + TOKEN})[0] == 200)
test("page has no-referrer and no-store",
     lambda: (lambda h: h.get("Referrer-Policy") == "no-referrer" and h.get("Cache-Control") == "no-store")(
         get("/", {"X-QA-Token": TOKEN})[1]))

def post_refused():
    try:
        urllib.request.urlopen(urllib.request.Request(BASE + "/", data=b"x", headers={"X-QA-Token": TOKEN}), timeout=5)
    except urllib.error.HTTPError as e:
        return e.code == 405
    return False


# ---- status schema and lanes ----
me = os.getpid()
gone = dead_pid()
write_meta(me, "heavy", "running", slot="heavy")
write_meta(gone, "heavy", "running", slot="heavy")
waiter = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
write_meta(waiter.pid, "light", "waiting")
with open(os.path.join(LANES, "meta", "999999999.json"), "w") as fh:
    fh.write("{not json")
with open(os.path.join(LOGS, "old.log"), "w") as fh:
    fh.write("old\n")
os.utime(os.path.join(LOGS, "old.log"), (time.time() - 100, time.time() - 100))
with open(os.path.join(LOGS, "new.log"), "w") as fh:
    fh.write("line1\nkey sk-or-v1-" + "a" * 40 + " for me@example.com\n")


def status():
    code, _, body = get("/api/status", {"X-QA-Token": TOKEN})
    assert code == 200, code
    return json.loads(body)


def schema():
    s = status()
    lanes = s["lanes"]
    return (set(s) >= {"generated_at", "host", "lanes", "logs", "health", "prs"}
            and lanes["heavy"]["slots"] == 1 and lanes["light"]["slots"] >= 1
            and set(lanes["heavy"]["holders"][0]) >= {"pid", "command", "clone", "branch", "elapsed_s", "slot"}
            and set(lanes["light"]["waiters"][0]) >= {"pid", "command", "waiting_s"}
            and set(s["logs"]) >= {"dir", "files", "selected", "tail"}
            and set(s["health"]) >= {"load", "swap", "simulators"} and set(s["prs"]) >= {"items", "updated_at", "error"})


test("/api/status schema", schema)
test("dead pids and bad files are dropped",
     lambda: [h["pid"] for h in status()["lanes"]["heavy"]["holders"]] == [me]
     and [w["pid"] for w in status()["lanes"]["light"]["waiters"]] == [waiter.pid])
test("holder shows clone, branch, elapsed",
     lambda: (lambda h: h["clone"] == "c1" and h["branch"] == "b1" and 9 <= h["elapsed_s"] <= 60)(
         status()["lanes"]["heavy"]["holders"][0]))
test("newest log is selected and its tail is scrubbed",
     lambda: (lambda l: l["selected"] == "new.log" and "line1" in l["tail"] and "sk-or" not in l["tail"]
              and "example.com" not in l["tail"])(status()["logs"]))
test("a chosen log is tailed",
     lambda: json.loads(get("/api/status?log=old.log", {"X-QA-Token": TOKEN})[2])["logs"]["tail"] == "old")
test("path traversal falls back to the newest log",
     lambda: json.loads(get("/api/status?log=../lanes/meta/%d.json" % me, {"X-QA-Token": TOKEN})[2])["logs"]["selected"] == "new.log")
test("tail_log refuses names outside the log dir", lambda: qd.tail_log(LOGS, "../x") is None and qd.tail_log(LOGS, "/etc/passwd") is None)
test("POST is refused (405)", post_refused)
waiter.kill()
waiter.wait()

# ---- scrubbing ----
SAMPLES = {
    "OpenRouter": "OPENROUTER_API_KEY=sk-or-v1-" + "0123456789abcdef" * 4,
    "Anthropic": "using sk-ant-api03-" + "Z" * 30,
    "GitHub": "ghp_" + "A1b2C3d4" * 5,
    "bearer": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345",
    "email": "uploaded by matthew.chan+qa@gmail.com today",
    "sim UDID": "Booted: iPhone 17 (4F3C2B1A-1234-4ABC-9DEF-0123456789AB)",
    "device UDID": "device 00008110-001A2B3C4D5E801E connected",
    "team": "DEVELOPMENT_TEAM = AB12CD34EF;",
    "signing team": 'Signing Identity: "Apple Development: Matt (Z9Y8X7W6V5)"',
    "url token": "GET /?token=" + "q" * 32,
    "private key": "-----BEGIN " + "PRIVATE KEY-----\nMIIabc\n-----END " + "PRIVATE KEY-----",   # split so scan_secrets passes
}
SECRETS = {"OpenRouter": "0123456789abcdef", "Anthropic": "ZZZZZZZZZZ", "GitHub": "A1b2C3d4", "bearer": "abcdefghijklmnop",
           "email": "gmail.com", "sim UDID": "4F3C2B1A", "device UDID": "001A2B3C4D5E801E", "team": "AB12CD34EF",
           "signing team": "Z9Y8X7W6V5", "url token": "qqqqqqqqqqqqqqqq", "private key": "MIIabc"}
for what, text in SAMPLES.items():
    test("scrub hides " + what, lambda t=text, s=SECRETS[what]: s not in qd.scrub(t, extra=[]))
test("scrub keeps ordinary lines", lambda: qd.scrub("PASS 612 tests in 9 s, commit 60e62a4", extra=[]) ==
     "PASS 612 tests in 9 s, commit 60e62a4")
test("scrub hides CLAUDE.local.md values", lambda: "MYDEVICE42" not in qd.scrub("run on MYDEVICE42", extra=["MYDEVICE42"]))

# ---- LaunchAgent plist ----
test("plist has the label and no token",
     lambda: (lambda p: p["Label"] == "com.kit.qa-dashboard" and TOKEN not in json.dumps(p)
              and p["ProgramArguments"][1].endswith("qa_dashboard.py"))(qd.launchagent_plist(8765, LANES, LOGS)))

srv.shutdown()

# ---- the real process never prints or logs the token ----


def process_hides_token():
    env = dict(os.environ, KIT_LANE_DIR=LANES, KIT_QA_LOG_DIR=LOGS, KIT_QA_DASHBOARD_DIR=CONF, PATH="/usr/bin:/bin")
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "qa_dashboard.py"), "--host", "127.0.0.1", "--port", "0"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        line = p.stdout.readline()
        port = int(line.split("127.0.0.1:")[1].split(";")[0])
        url = "http://127.0.0.1:%d" % port
        try:
            opener.open(url + "/?token=" + TOKEN, timeout=5)
        except urllib.error.HTTPError:
            pass
        with opener.open(urllib.request.Request(url + "/api/status", headers={"X-QA-Token": TOKEN}), timeout=5) as r:
            ok = r.status == 200
    finally:
        p.terminate()
    out, err = p.communicate(timeout=10)
    return ok and TOKEN not in line + out + err and "url.txt" in line and "GET /api/status" in err


test("server output never shows the token", process_hides_token)

# ---- lane.py status files ----
LANE = os.path.join(HERE, "lane.py")


def lane_env(d):
    env = dict(os.environ, KIT_LANE_DIR=d)
    env.pop("KIT_LANE_RANK", None)
    return env


def wait_file(pred, seconds=5):
    end = time.time() + seconds
    while time.time() < end:
        got = pred()
        if got:
            return got
        time.sleep(0.05)
    return None


def metas(d):
    m = os.path.join(d, "meta")
    out = {}
    for f in os.listdir(m) if os.path.isdir(m) else []:
        if f.endswith(".json") and not f.startswith("."):
            try:
                out[int(f[:-5])] = json.load(open(os.path.join(m, f)))
            except ValueError:
                pass
    return out


def lane_meta_lifecycle():
    d = tempfile.mkdtemp(dir=tmp)
    gate = os.path.join(d, "gate")
    hold = subprocess.Popen([sys.executable, LANE, "heavy", "--", sys.executable, "-c",
                             "import os,time\nwhile not os.path.exists(%r): time.sleep(0.05)" % gate],
                            env=lane_env(d), cwd=HERE, stderr=subprocess.DEVNULL)
    held = wait_file(lambda: metas(d).get(hold.pid))
    wait = subprocess.Popen([sys.executable, LANE, "heavy", "--", "true"], env=lane_env(d), stderr=subprocess.DEVNULL)
    waiting = wait_file(lambda: (metas(d).get(wait.pid) or {}).get("state") == "waiting")
    view = qd.lanes(d)
    open(gate, "w").close()
    hold.wait(timeout=10)
    wait.wait(timeout=10)
    gone_after = wait_file(lambda: metas(d) == {} or None)
    return bool(held and held["state"] == "running" and held["lane"] == "heavy" and held["slot"] == "heavy"
                and held["started_at"] and held["queued_at"] and "branch" in held and held["cwd"] == HERE
                and sys.executable in held["command"] and waiting
                and [h["pid"] for h in view["heavy"]["holders"]] == [hold.pid]
                and [w["pid"] for w in view["heavy"]["waiters"]] == [wait.pid] and gone_after
                and hold.returncode == 0 and wait.returncode == 0)


def lane_light_slot():
    d = tempfile.mkdtemp(dir=tmp)
    gate = os.path.join(d, "gate")
    p = subprocess.Popen([sys.executable, LANE, "light", "--", sys.executable, "-c",
                          "import os,time\nwhile not os.path.exists(%r): time.sleep(0.05)" % gate],
                         env=lane_env(d), stderr=subprocess.DEVNULL)
    m = wait_file(lambda: metas(d).get(p.pid))
    open(gate, "w").close()
    p.wait(timeout=10)
    return bool(m and m["lane"] == "light" and m["slot"] == "light-0" and p.returncode == 0 and not metas(d))


def lane_meta_error_is_harmless():
    d = tempfile.mkdtemp(dir=tmp)
    open(os.path.join(d, "meta"), "w").close()   # a file where the meta dir should be
    r = subprocess.run([sys.executable, LANE, "heavy", "--", sys.executable, "-c", "print('ran')"],
                       env=lane_env(d), capture_output=True, text=True, timeout=10)
    return r.returncode == 0 and r.stdout.strip() == "ran"


def lane_keeps_exit_code():
    d = tempfile.mkdtemp(dir=tmp)
    r = subprocess.run([sys.executable, LANE, "light", "--", sys.executable, "-c", "raise SystemExit(7)"],
                       env=lane_env(d), capture_output=True, timeout=10)
    return r.returncode == 7 and not metas(d)


test("lane.py writes status on wait and hold, removes it on release", lane_meta_lifecycle)
test("lane.py records the light slot", lane_light_slot)
test("lane.py still runs when status can't be written", lane_meta_error_is_harmless)
test("lane.py keeps the exit code and cleans up", lane_keeps_exit_code)


def dashboard_url_no_token():
    """ip and check print no token=. check follows the bookmark redirect with a cookie jar."""
    d = os.path.join(tmp, "dash-url")
    os.makedirs(d, 0o700)
    secret = "unit-test-token-value"
    token_path = os.path.join(d, "token")
    with open(token_path, "w") as fh:
        fh.write(secret + "\n")
    os.chmod(token_path, 0o600)
    with open(os.path.join(d, "url.txt"), "w") as fh:
        fh.write("http://10.14.0.2:8765/?token=%s\n" % secret)
    os.chmod(os.path.join(d, "url.txt"), 0o600)
    script = os.path.join(HERE, "dashboard_url.sh")
    env = dict(os.environ)
    env["KIT_QA_DASHBOARD_DIR"] = d
    env.pop("KIT_QA_DASHBOARD_PORT", None)

    def clean(proc):
        blob = (proc.stdout or "") + (proc.stderr or "")
        return "token=" not in blob and secret not in blob

    ip = subprocess.run(["bash", script, "ip"], capture_output=True, text=True, env=env, timeout=15)
    # Equals the live address, so a stale host left in url.txt is not what got printed.
    if ip.returncode != 0 or not clean(ip) or ip.stdout.strip() != qd.lan_ip():
        return False
    closed = dict(env)
    closed["KIT_QA_DASHBOARD_PORT"] = "9"
    refused = subprocess.run(["bash", script, "check"], capture_output=True, text=True, env=closed, timeout=20)
    if refused.returncode != 0 or not clean(refused) or refused.stdout.strip() not in ("000", "200"):
        return False
    srv_env = dict(env, KIT_LANE_DIR=LANES, KIT_QA_LOG_DIR=LOGS)
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "qa_dashboard.py"), "--host", "0.0.0.0", "--port", "0"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=srv_env)
    try:
        line = proc.stdout.readline()
        if "0.0.0.0:" not in line or secret in line:
            return False
        port = line.split("0.0.0.0:")[1].split(";")[0]
        live = dict(env)
        live["KIT_QA_DASHBOARD_PORT"] = port
        chk = subprocess.run(["bash", script, "check"], capture_output=True, text=True, env=live, timeout=20)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    return chk.returncode == 0 and chk.stdout.strip() == "200" and clean(chk)


test("dashboard_url.sh ip and check never print token=", dashboard_url_no_token)

shutil.rmtree(tmp, ignore_errors=True)
if failures:
    print(f"FAIL {failures} of {count} qa dashboard checks")
    sys.exit(1)
print(f"PASS {count} qa dashboard checks")
