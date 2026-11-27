#!/bin/bash
# kit/ios/swift_gate.sh decisions and path matching, with made-up settings.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
gate="$here/swift_gate.sh"
export KIT_NO_ENV=1 KIT_APP_FILE_RE='^(App/|App\.xcodeproj/|tests/)' KIT_UI_FILE_RE='^App/.*View\.swift$'
checks=0 fails=0
is() { checks=$((checks+1)); [ "$2" = "$3" ] || { fails=$((fails+1)); echo "FAIL $1: got '$2' want '$3'"; }; }
is "app file" "$(printf 'App/A.swift\n' | "$gate" app-changed)" true
is "docs only" "$(printf 'docs/x.md\nREADME.md\n' | "$gate" app-changed)" false
is "ui file" "$(printf 'App/HomeView.swift\n' | "$gate" ui-changed)" true
is "logic not ui" "$(printf 'App/Model.swift\n' | "$gate" ui-changed)" false
is "no app files" "$(EVENT=pull_request APP_CHANGED=false "$gate" decide)" none
is "override" "$(EVENT=pull_request APP_CHANGED=true LABELS=skip-swift "$gate" decide)" override
is "no label waits" "$(EVENT=pull_request APP_CHANGED=true LABELS= "$gate" decide 2>/dev/null)" wait
is "label added runs" "$(EVENT=pull_request ACTION=labeled ADDED_LABEL=ready-to-merge APP_CHANGED=true LABELS=ready-to-merge "$gate" decide 2>/dev/null)" run
is "push after label waits" "$(EVENT=pull_request ACTION=synchronize APP_CHANGED=true LABELS=ready-to-merge "$gate" decide 2>/dev/null)" wait
DECISION=wait "$gate" verdict >/dev/null; is "wait blocks" $? 1
DECISION=run SWIFT_RESULT=success "$gate" verdict >/dev/null; is "pass ok" $? 0
[ $fails -eq 0 ] && echo "PASS $checks swift_gate checks" || { echo "FAIL $fails of $checks swift_gate checks"; exit 1; }
