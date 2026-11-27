#!/bin/bash
# Layout check on iPhone simulators: for each phone x text size x shot, launch the DEBUG app in that state, wait for its
# layout dump, screenshot it, and check the dump with kit/tools/layout_rules.py. Screens go to build/layout-shots/.
#   kit/ios/check_layouts.sh --set alpha            the phones in KIT_LAYOUT_PHONES_ALPHA (";"-separated names)
#   kit/ios/check_layouts.sh "iPhone 17e" …         just these phones
#   --text-size S / --state S / --orientation O     only the matching shots (a cheap recheck of one failure)
#   --jobs N                                        phones at a time (default KIT_LAYOUT_JOBS or 3)
#   --print-plan                                    print the shots, then exit (no build, no simulators; works on Linux)
#   --phone-count                                   print how many phones the selection has, then exit
# Settings (kit.env): KIT_LAYOUT_SHOTS (default layout_shots.txt: "state | orientations | launch args" per line),
# KIT_LAYOUT_RULES (default layout_rules.json), KIT_LAYOUT_TEXT_SIZES (default "large"), KIT_LAYOUT_DUMP
# (default layout-dump.json, in the app's Documents), KIT_LAYOUT_PHONE_TIMEOUT (default 600 s), KIT_SIM_BUNDLE.
# The app's side (DEBUG only): read -kitLayoutToken, -kitLayoutState, -kitLayoutOrientation, show that state,
# then write {"token", "frames":[{id,x,y,width,height}]} to Documents/<KIT_LAYOUT_DUMP>. See templates/ios/LayoutDump.swift.
# A missing dump or a timeout is a real failure, not a flake: bring it to the owner instead of rerunning. Ask first.
set -uo pipefail
source "$(dirname "$0")/../tools/lib.sh"

shots_file=${KIT_LAYOUT_SHOTS:-layout_shots.txt}
rules=${KIT_LAYOUT_RULES:-layout_rules.json}
sizes=${KIT_LAYOUT_TEXT_SIZES:-large}
dump_name=${KIT_LAYOUT_DUMP:-layout-dump.json}
jobs=${KIT_LAYOUT_JOBS:-3}
phone_timeout=${KIT_LAYOUT_PHONE_TIMEOUT:-600}
shot_timeout=${KIT_LAYOUT_SHOT_TIMEOUT:-60}
bundle=${KIT_SIM_BUNDLE:-com.example.${KIT_SCHEME:-app}}
out=build/layout-shots
phones=() only_size= only_state= only_orient= plan=0 count=0 set_name=
while [ $# -gt 0 ]; do
    case "$1" in
        --set) set_name=$2; shift 2 ;;
        --text-size) only_size=$2; shift 2 ;;
        --state|--pass) only_state=$2; shift 2 ;;
        --orientation) only_orient=$2; shift 2 ;;
        --jobs) jobs=$2; shift 2 ;;
        --print-plan) plan=1; shift ;;
        --phone-count) count=1; shift ;;
        --device) phones+=("$2"); shift 2 ;;
        --help|-h) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --*) echo "check_layouts.sh: unknown option $1"; exit 2 ;;
        *) phones+=("$1"); shift ;;
    esac
done
if [ -n "$set_name" ]; then
    var="KIT_LAYOUT_PHONES_$(printf '%s' "$set_name" | tr '[:lower:]-' '[:upper:]_')"
    [ -n "${!var:-}" ] || { echo "check_layouts.sh: no $var in kit.env"; exit 2; }
    IFS=';' read -r -a from_set <<< "${!var}"
    for p in "${from_set[@]}"; do p=$(echo "$p" | sed 's/^ *//; s/ *$//'); [ -n "$p" ] && phones+=("$p"); done
fi
[ ${#phones[@]} -gt 0 ] || { echo "check_layouts.sh: name phones or --set <name>"; exit 2; }
[ $count = 1 ] && { echo "${#phones[@]}"; exit 0; }
[ -f "$shots_file" ] || { echo "check_layouts.sh: no $shots_file"; exit 2; }
[ -f "$rules" ] || { echo "check_layouts.sh: no $rules"; exit 2; }

# shot_list: "size|state|orientation|args" for every selected shot.
shot_list() {
    local size state orients args o
    for size in $sizes; do
        [ -z "$only_size" ] || [ "$size" = "$only_size" ] || continue
        while IFS='|' read -r state orients args; do
            state=$(echo "$state" | xargs); orients=$(echo "${orients:-portrait}" | tr -d ' '); args=$(echo "${args:-}" | sed 's/^ *//; s/ *$//')
            [ -n "$state" ] && [ "${state:0:1}" != "#" ] || continue
            [ -z "$only_state" ] || [ "$state" = "$only_state" ] || continue
            for o in ${orients//,/ }; do
                [ -z "$only_orient" ] || [ "$o" = "$only_orient" ] || continue
                echo "$size|$state|$o|$args"
            done
        done < "$shots_file"
    done
}
shots=$(shot_list)
[ -n "$shots" ] || { echo "check_layouts.sh: no shots match"; exit 2; }
if [ $plan = 1 ]; then
    for p in "${phones[@]}"; do echo "$shots" | sed "s/^/$p|/"; done
    echo "PLAN ${#phones[@]} phones x $(echo "$shots" | wc -l | tr -d ' ') shots, $jobs at a time" >&2
    exit 0
fi
[ "$(uname -s)" = Darwin ] || { echo "check_layouts.sh needs a Mac with Xcode"; exit 1; }
refuse_protected_checkout || exit 1
need_lane heavy "$KIT_DIR/ios/check_layouts.sh" ${set_name:+--set "$set_name"} "${phones[@]}" \
    ${only_size:+--text-size "$only_size"} ${only_state:+--state "$only_state"} ${only_orient:+--orientation "$only_orient"} --jobs "$jobs"
sim() { "$KIT_DIR/ios/simctl.sh" "$@"; }

echo "== build for the simulator =="
"$KIT_DIR/ios/xcodebuild.sh" -project "$KIT_XCODE_PROJECT" -scheme "$KIT_SCHEME" -configuration Debug \
    -destination 'generic/platform=iOS Simulator' -derivedDataPath build PRODUCT_BUNDLE_IDENTIFIER="$bundle" \
    CODE_SIGNING_ALLOWED=NO build > build/layout-build.log 2>&1 \
    || { grep -E "error:" build/layout-build.log | sort -u | head; echo "LAYOUT BUILD FAILED"; exit 1; }
app=$(ls -d build/Build/Products/Debug-iphonesimulator/*.app | head -1)

# udid_for <name>: an existing simulator with that name, else a new one on the newest iOS runtime.
udid_for() {
    local u
    u=$(sim list devices available -j | python3 -c 'import json,sys; n=sys.argv[1]; d=json.load(sys.stdin)["devices"]
print(next((x["udid"] for r in sorted(d, reverse=True) for x in d[r] if x["name"] == n), ""))' "$1")
    if [ -z "$u" ]; then
        local rt; rt=$(sim list runtimes -j | python3 -c 'import json,sys; r=[x for x in json.load(sys.stdin)["runtimes"] if x.get("platform")=="iOS" and x.get("isAvailable")]; print(r[-1]["identifier"] if r else "")')
        u=$(sim create "$1" "$1" "$rt" 2>/dev/null) || return 1
    fi
    echo "$u"
}

# run_phone <name>: every shot on one phone; FAIL/PASS lines on stdout.
run_phone() {
    local name=$1 udid size state o args token dir data dump png waited last_size=
    udid=$(udid_for "$name") || { echo "FAIL $name: no simulator"; return 1; }
    sim boot "$udid" >/dev/null 2>&1; sim bootstatus "$udid" -b >/dev/null 2>&1
    sim install "$udid" "$app" || { echo "FAIL $name: install failed"; return 1; }
    local st=0
    while IFS='|' read -r size state o args; do
        [ "$size" = "$last_size" ] || { sim ui "$udid" content_size "$size" >/dev/null 2>&1; last_size=$size; }
        token=$(python3 -c 'import secrets; print(secrets.token_hex(6))')
        data=$(sim get_app_container "$udid" "$bundle" data 2>/dev/null)
        dump="$data/Documents/$dump_name"; rm -f "$dump"
        # shellcheck disable=SC2086
        sim launch --terminate-running-process "$udid" "$bundle" -kitLayoutToken "$token" -kitLayoutState "$state" \
            -kitLayoutOrientation "$o" $args >/dev/null 2>&1
        waited=0
        until python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("token")==sys.argv[2] else 1)' "$dump" "$token" 2>/dev/null; do
            sleep 1; waited=$((waited + 1))
            [ $waited -lt "$shot_timeout" ] || break
        done
        dir="$out/$name/$size"; mkdir -p "$dir"; png="$dir/$state-$o.png"
        sim io "$udid" screenshot "$png" >/dev/null 2>&1
        cp "$dump" "$dir/$state-$o.json" 2>/dev/null
        python3 "$KIT_DIR/tools/layout_rules.py" --rules "$rules" --dump "$dump" --state "$state" --orientation "$o" \
            --token "$token" --where "$name / $size / $o / $state" || st=1
    done <<< "$shots"
    sim shutdown "$udid" >/dev/null 2>&1
    return $st
}

mkdir -p "$out" build/layout-logs
pids=() names=() watchers=()
running() { local n=0 q; for q in "${pids[@]+"${pids[@]}"}"; do kill -0 "$q" 2>/dev/null && n=$((n + 1)); done; echo $n; }
for p in "${phones[@]}"; do
    while [ "$(running)" -ge "$jobs" ]; do sleep 1; done
    log="build/layout-logs/$(echo "$p" | tr ' /()' '____').log"
    ( run_phone "$p" > "$log" 2>&1 ) &
    pid=$!
    ( sleep "$phone_timeout"; kill "$pid" 2>/dev/null && echo "FAIL $p: timed out after ${phone_timeout}s" >> "$log" ) &
    watchers+=($!); pids+=("$pid"); names+=("$p")
done
st=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || st=1
    grep -E '^(FAIL|WARN) ' "build/layout-logs/$(echo "${names[$i]}" | tr ' /()' '____').log" || true
done
kill "${watchers[@]}" 2>/dev/null
passed=$(cat build/layout-logs/*.log 2>/dev/null | grep -c '^PASS ')
failed=$(cat build/layout-logs/*.log 2>/dev/null | grep -c '^FAIL ')
echo "LAYOUT ${passed} passed, ${failed} failed; screens in $out"
[ $st -eq 0 ] && [ "$failed" -eq 0 ]
