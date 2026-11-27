#!/bin/bash
# The Swift workflow's merge gate (OPS-007). GitHub's Mac job runs only when the
# ready-to-merge label is added (not on later pushes). A new commit stays blocked
# until the label is added again or a Mac pass note exists for the new tree.
#   kit/ios/swift_gate.sh decide   env EVENT, ACTION, ADDED_LABEL, LABELS (comma list),
#     APP_CHANGED (true/false), SKIP_TESTS/SKIP_RELEASE/SKIP_LAYOUTS (dispatch inputs);
#     prints the decision:
#     run (Mac job runs), skip (Mac pass for this tree), wait (no fresh label yet),
#     override (skip-swift label), none (no app files, or a dispatch with nothing to do)
#   kit/ios/swift_gate.sh verdict  env DECISION, SWIFT_RESULT (the Mac job's result); exits 1 to block the merge.
#   kit/ios/swift_gate.sh app-changed  stdin: PR filenames, one per line; prints true if any needs the Swift check, else false.
#   kit/ios/swift_gate.sh ui-changed   stdin: PR filenames, one per line; prints true if any can change
#     what the layout check screenshots (views, overlay, camera/UI helpers, assets, template photos).
# Tested by kit/ios/test_mac_pass.sh.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../tools/lib.sh"
ready=ready-to-merge override=skip-swift
has_label() { case ",${LABELS:-}," in *",$1,"*) return 0 ;; esac; return 1; }
wanted() { [ "${1:-}" != true ]; }   # empty or false means the step is wanted
mac_pass() { "$here/mac_pass_check.sh" HEAD; }

# Paths that need the Swift check (KIT_APP_FILE_RE in kit.env, an ERE). Others print false.
APP_FILE_RE=${KIT_APP_FILE_RE:?set KIT_APP_FILE_RE in kit.env}

# Paths that can change screenshots (KIT_UI_FILE_RE in kit.env).
UI_FILE_RE=${KIT_UI_FILE_RE:-^$}

case "${1:-}" in
decide)
    if [ "${EVENT:-}" = "workflow_dispatch" ]; then
        pass=$(mac_pass)
        if wanted "${SKIP_RELEASE:-}" || wanted "${SKIP_LAYOUTS:-}"; then echo run
        elif wanted "${SKIP_TESTS:-}" && [ "$pass" != skip ]; then echo run
        elif [ "$pass" = skip ]; then echo skip
        else echo none
        fi
    elif [ "${APP_CHANGED:-true}" != "true" ]; then echo none
    elif has_label "$override"; then echo override
    elif has_label "$ready"; then
        pass=$(mac_pass)
        if [ "$pass" = skip ]; then echo skip
        elif [ "${ACTION:-}" = labeled ] && [ "${ADDED_LABEL:-}" = "$ready" ]; then echo run
        else echo wait
        fi
    else echo wait
    fi ;;
verdict)
    case "${DECISION:-}" in
        none) echo "No app files changed: no Swift check needed." ;;
        skip) echo "Mac pass found for this tree: ready to merge." ;;
        override) echo "Swift check skipped by the $override label." ;;
        run) [ "${SWIFT_RESULT:-}" = "success" ] && echo "Swift tests and Debug build passed: ready to merge." \
                 || { echo "Swift tests or Debug build: ${SWIFT_RESULT:-not run}. Merge blocked."; exit 1; } ;;
        wait) echo "Not checked yet. Run kit/ios/mac_pass.sh on the Mac, then add the $ready label (remove and add it again if a new commit landed after it was added). Or $override to merge without a check. Merge blocked."; exit 1 ;;
        *) echo "Unknown decision '${DECISION:-}'. Merge blocked."; exit 1 ;;
    esac ;;
app-changed)
    grep -qE "$APP_FILE_RE" && echo true || echo false ;;
ui-changed)
    grep -qE "$UI_FILE_RE" && echo true || echo false ;;
*) echo "usage: kit/ios/swift_gate.sh decide|verdict|app-changed|ui-changed"; exit 2 ;;
esac
