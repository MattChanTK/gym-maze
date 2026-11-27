#!/bin/bash
# Hold a heavy (1) or light (N, default 3) run-queue slot, then exec the command.
#   tools/lane.sh heavy|light [--] command [args…]
# The slot is a kernel lock on an open file, so it frees when this process
# exits, crashes, or is kill -9'd. No pid files. See docs/TESTING.md.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/lane.py" "$@"
