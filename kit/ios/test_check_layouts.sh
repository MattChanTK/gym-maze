#!/bin/bash
# kit/ios/check_layouts.sh shot planning (--print-plan, --phone-count, filters). No Xcode needed.
set -uo pipefail
unset KIT_NO_ENV
kit=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/kit-layout-test.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
cd "$tmp" && git init -q && cp -R "$kit" kit
printf 'home | portrait,landscapeLeft | -demo YES\n# comment\nsheet | portrait |\n' > layout_shots.txt
echo '{}' > layout_rules.json
printf 'KIT_LAYOUT_PHONES_ALPHA="Phone A; Phone B"\nKIT_LAYOUT_TEXT_SIZES="large xl"\n' > kit.env
checks=0 fails=0
is() { checks=$((checks+1)); [ "$2" = "$3" ] || { fails=$((fails+1)); echo "FAIL $1: got '$2' want '$3'"; }; }
is "phone count" "$(kit/ios/check_layouts.sh --set alpha --phone-count)" 2
is "plan rows" "$(kit/ios/check_layouts.sh --set alpha --print-plan 2>/dev/null | wc -l | tr -d ' ')" 12
is "comment skipped" "$(kit/ios/check_layouts.sh --set alpha --print-plan 2>/dev/null | grep -c comment)" 0
is "args kept" "$(kit/ios/check_layouts.sh 'Phone A' --print-plan --state home --orientation landscapeLeft --text-size xl 2>/dev/null)" "Phone A|xl|home|landscapeLeft|-demo YES"
kit/ios/check_layouts.sh --set beta --print-plan >/dev/null 2>&1; is "unknown set fails" $? 2
kit/ios/check_layouts.sh 'Phone A' --print-plan --state nope >/dev/null 2>&1; is "no matching shots fails" $? 2
[ $fails -eq 0 ] && echo "PASS $checks check_layouts checks" || { echo "FAIL $fails of $checks check_layouts checks"; exit 1; }
