#!/bin/bash
# Behavior of tools/lane.sh: wait, auto-release on kill -9, light slot limit,
# re-entrancy, and a lint that every test or build script takes a lane.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
lane="$root/tools/lane.sh"
fails=0
checks=0

tmp=$(mktemp -d "${TMPDIR:-/tmp}/kit-lane-test.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

# Isolate from a parent queue (tools/test_py.sh may already hold a light slot).
unset KIT_LANE_RANK
unset KIT_LANE_FD
export KIT_LANE_DIR="$tmp/lanes"
export KIT_LANE_LIGHT_SLOTS=3

ok() { checks=$((checks + 1)); }
fail() { checks=$((checks + 1)); echo "FAIL $1"; fails=$((fails + 1)); }

wait_for() {
    # wait_for <file> <needle> <seconds>
    local f=$1 needle=$2 end now
    end=$(python3 -c "import time; print(time.time() + float('$3'))")
    while true; do
        if [ -f "$f" ] && grep -q "$needle" "$f" 2>/dev/null; then
            return 0
        fi
        now=$(python3 -c "import time; print(time.time())")
        python3 -c "import sys; sys.exit(0 if float('$now') >= float('$end') else 1)" && return 1
        sleep 0.05
    done
}

# 1) Two heavy runs: the second waits, then runs.
hold_log=$tmp/hold.log
wait_log=$tmp/wait.log
"$lane" heavy -- bash -c 'echo hold-start; sleep 2; echo hold-end' >"$hold_log" 2>&1 &
hp=$!
if ! wait_for "$hold_log" "hold-start" 3; then
    fail "first heavy run did not start: $(cat "$hold_log" 2>/dev/null)"
    kill -9 $hp 2>/dev/null || true
else
    "$lane" heavy -- bash -c 'echo wait-ran' >"$wait_log" 2>&1 &
    wp=$!
    if ! wait_for "$wait_log" "waiting for heavy lane" 3; then
        fail "second heavy run did not print waiting message: $(cat "$wait_log" 2>/dev/null)"
    else
        ok
    fi
    if grep -q wait-ran "$wait_log" 2>/dev/null; then
        fail "second heavy run finished before the first released"
    else
        ok
    fi
    wait $hp
    wait $wp
    if grep -q wait-ran "$wait_log" && grep -q hold-end "$hold_log"; then
        ok
    else
        fail "second heavy run did not run after the first finished: $(cat "$wait_log")"
    fi
fi

# 2) kill -9 the holder: the waiter proceeds within 1 s.
: >"$hold_log"
: >"$wait_log"
"$lane" heavy -- bash -c 'echo ready; sleep 60' >"$hold_log" 2>&1 &
hp=$!
if ! wait_for "$hold_log" "ready" 3; then
    fail "kill-9 holder did not start: $(cat "$hold_log" 2>/dev/null)"
    kill -9 $hp 2>/dev/null || true
else
    "$lane" heavy -- bash -c 'echo released' >"$wait_log" 2>&1 &
    wp=$!
    if ! wait_for "$wait_log" "waiting for heavy lane" 3; then
        fail "kill-9 waiter did not start waiting: $(cat "$wait_log" 2>/dev/null)"
        kill -9 $hp $wp 2>/dev/null || true
    else
        t0=$(python3 -c "import time; print(time.time())")
        kill -9 $hp
        wait $hp 2>/dev/null || true
        wait $wp
        elapsed=$(python3 -c "import time; print(time.time() - float('$t0'))")
        if grep -q released "$wait_log"; then
            ok
        else
            fail "waiter did not run after kill -9: $(cat "$wait_log")"
        fi
        if python3 -c "import sys; sys.exit(0 if float('$elapsed') <= 1.0 else 1)"; then
            ok
        else
            fail "waiter took ${elapsed}s after kill -9 (want <= 1s)"
        fi
    fi
fi

# 3) 4 light runs with N=3: 3 run at once, 1 waits.
for i in 1 2 3 4; do
    "$lane" light -- bash -c 'echo start; sleep 2; echo end' >"$tmp/light-$i.log" 2>&1 &
    eval "lp$i=$!"
done
# Wait until at least 3 have started.
deadline=$(python3 -c "import time; print(time.time() + 3)")
started=0
while true; do
    started=0
    for i in 1 2 3 4; do
        grep -q start "$tmp/light-$i.log" 2>/dev/null && started=$((started + 1))
    done
    [ "$started" -ge 3 ] && break
    now=$(python3 -c "import time; print(time.time())")
    python3 -c "import sys; sys.exit(0 if float('$now') >= float('$deadline') else 1)" && break
    sleep 0.05
done
waiting=0
finished_early=0
for i in 1 2 3 4; do
    grep -q "waiting for light lane" "$tmp/light-$i.log" 2>/dev/null && waiting=$((waiting + 1))
    grep -q end "$tmp/light-$i.log" 2>/dev/null && finished_early=$((finished_early + 1))
done
if [ "$started" -eq 3 ]; then
    ok
else
    fail "expected 3 light runs at once, got $started started"
fi
if [ "$waiting" -eq 1 ]; then
    ok
else
    fail "expected 1 light waiter, got $waiting (logs in $tmp)"
fi
if [ "$finished_early" -eq 0 ]; then
    ok
else
    fail "a light run finished before the first wave should have ended"
fi
for i in 1 2 3 4; do
    eval "wait \$lp$i"
done
ended=0
for i in 1 2 3 4; do
    grep -q end "$tmp/light-$i.log" && ended=$((ended + 1))
done
if [ "$ended" -eq 4 ]; then
    ok
else
    fail "expected all 4 light runs to finish, got $ended"
fi

# 4) Nested holders do not deadlock (qa.sh -> test.sh).
nested_log=$tmp/nested.log
"$lane" heavy -- "$lane" heavy -- "$lane" light -- bash -c 'echo nested-ok' >"$nested_log" 2>&1 &
np=$!
if wait_for "$nested_log" "nested-ok" 3; then
    wait $np
    if grep -q "waiting for" "$nested_log"; then
        fail "nested lane re-entry waited: $(cat "$nested_log")"
    else
        ok
    fi
else
    kill -9 $np 2>/dev/null || true
    fail "nested qa.sh -> test.sh deadlocked or timed out: $(cat "$nested_log" 2>/dev/null)"
fi

# A child script that takes a heavy lane itself, invoked from a heavy holder.
cat > "$tmp/parent.sh" <<'EOS'
#!/bin/bash
set -euo pipefail
root=$1
# Pretend to be qa.sh already holding heavy (lane.sh set the env).
"$root/tools/lane.sh" heavy -- "$root/tools/lane.sh" heavy -- echo child-ok
EOS
chmod +x "$tmp/parent.sh"
parent_log=$tmp/parent.log
"$lane" heavy -- bash "$tmp/parent.sh" "$root" >"$parent_log" 2>&1 &
pp=$!
if wait_for "$parent_log" "child-ok" 3; then
    wait $pp
    ok
else
    kill -9 $pp 2>/dev/null || true
    fail "child test.sh under qa.sh did not finish: $(cat "$parent_log" 2>/dev/null)"
fi

# 5) Lint: every test or build script calls the lane helper, and a script
# with no call is reported.
# Allowlist: tools/*.sh that are not test or build entry points. A new
# tools/*.sh must take a lane or be added here.
lane_allowed() {
    case "$1" in
        check.sh|commit.sh|kit.sh|dashboard_url.sh|test_swift_gate.sh|test_qa.sh|test_check_layouts.sh) return 0 ;;
        fetch_test_photos.sh|lane.sh|lib.sh|mac_pass_check.sh) return 0 ;;
        mail_report.sh|openrouter_usage.sh|pick_mac_runner.sh|push.sh) return 0 ;;
        run_js.sh|swift_gate.sh|test_cache.sh) return 0 ;;
        test_check_layouts_order.sh|test_lane.sh|test_mac_pass.sh|test_ops009.sh|test_qa.sh) return 0 ;;
        test_worktree.sh|test_protected_checkout.sh|test_testflight_guard.sh|test_wrappers.sh) return 0 ;;
        worktree.sh) return 0 ;;
        try_idea_picture.sh|try_preview.sh) return 0 ;;
        *) return 1 ;;
    esac
}

missing_lane_scripts() {
    local base=$1 f name
    for f in "$base"/tools/*.sh; do
        [ -f "$f" ] || continue
        name=$(basename "$f")
        if lane_allowed "$name"; then
            continue
        fi
        if ! grep -qE 'lane\.sh|need_lane' "$f"; then
            printf '%s\n' "tools/$name"
        fi
    done
    for f in "$base/server/test.sh" "$base/site/test.sh"; do
        if [ -f "$f" ] && ! grep -qE 'lane\.sh|need_lane' "$f"; then
            printf '%s\n' "${f#$base/}"
        fi
    done
}

offenders=$(missing_lane_scripts "$root" || true)
if [ -z "$offenders" ]; then
    ok
else
    fail "scripts missing a lane.sh call:"$'\n'"$offenders"
fi

mkdir -p "$tmp/fake/tools"
printf '%s\n' '#!/bin/bash' 'echo rogue' > "$tmp/fake/tools/rogue.sh"
# Also hide the required server/site files so only rogue.sh is checked as a tools script.
caught=$(missing_lane_scripts "$tmp/fake" || true)
if printf '%s\n' "$caught" | grep -q 'tools/rogue.sh'; then
    ok
else
    fail "lint did not catch a script with no lane: '$caught'"
fi

[ $fails -eq 0 ] && echo "PASS $checks lane checks" || { echo "FAIL $fails of $checks lane checks"; exit 1; }
