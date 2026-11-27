#!/bin/bash
# Live link to the QA dashboard. The address comes from qa_dashboard.lan_ip
# (en0, else the first non-loopback IPv4), computed now. url.txt's host is not used.
# The token is never printed.
#
#   tools/dashboard_url.sh            open in the browser (default); no output
#   tools/dashboard_url.sh open
#   tools/dashboard_url.sh copy       pbcopy
#   tools/dashboard_url.sh ip         print the current IP only
#   tools/dashboard_url.sh check      HTTP status after redirects (cookie jar)
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

die() {
    printf '%s\n' "$1" >&2
    exit 1
}

is_ipv4() {
    local ip="$1" octet
    [[ "$ip" =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]] || return 1
    for octet in "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" "${BASH_REMATCH[4]}"; do
        [ "$((10#$octet))" -le 255 ] || return 1
    done
}

lan_ip() {
    local ip
    ip=$(PYTHONPATH="$here" python3 -B -c 'import qa_dashboard; print(qa_dashboard.lan_ip())' 2>/dev/null) || return 1
    ip=${ip//$'\r'/}
    ip=${ip//$'\n'/}
    printf '%s' "$ip"
}

read_token() {
    local file token
    file="${KIT_QA_DASHBOARD_DIR:-$HOME/.config/kit-qa-dashboard}/token"
    [ -f "$file" ] || die "no dashboard token file"
    token=$(tr -d '[:space:]' < "$file") || die "could not read the dashboard token"
    [ -n "$token" ] || die "dashboard token file is empty"
    printf '%s' "$token"
}

dashboard_port() {
    local port="${KIT_QA_DASHBOARD_PORT:-8765}"
    case "$port" in
        ''|*[!0-9]*) die "bad dashboard port" ;;
    esac
    printf '%s' "$port"
}

# Build the link in memory. Callers must not echo it.
dashboard_url() {
    local addr token port
    addr=$(lan_ip) || die "could not read the LAN address"
    is_ipv4 "$addr" || die "could not read the LAN address"
    token=$(read_token)
    port=$(dashboard_port)
    printf '%s' "http://${addr}:${port}/?token=${token}"
}

open_url() {
    command -v open >/dev/null 2>&1 || die "open is not available"
    open "$1" >/dev/null 2>&1 || die "could not open the dashboard"
}

copy_url() {
    command -v pbcopy >/dev/null 2>&1 || die "pbcopy is not available"
    printf '%s' "$1" | pbcopy
}

# Final status after redirects. The cookie jar keeps the HttpOnly cookie the
# bookmark sets on the way from /?token= to /. Stderr is dropped so a failure
# cannot print the link.
check_url() {
    local url="$1" jar cfg code
    command -v curl >/dev/null 2>&1 || die "curl is not available"
    jar=$(mktemp "${TMPDIR:-/tmp}/kit-qa-cookies.XXXXXX")
    cfg=$(mktemp "${TMPDIR:-/tmp}/kit-qa-curl.XXXXXX")
    chmod 600 "$jar" "$cfg" || { rm -f "$jar" "$cfg"; die "could not check the dashboard"; }
    printf 'url = "%s"\n' "$url" > "$cfg" || { rm -f "$jar" "$cfg"; die "could not check the dashboard"; }
    code=$(curl --silent --output /dev/null --location \
        --cookie-jar "$jar" --cookie "$jar" \
        --connect-timeout 3 --max-time 8 --max-redirs 5 \
        --write-out '%{http_code}' \
        --config "$cfg" 2>/dev/null) || true
    rm -f "$jar" "$cfg"
    case "$code" in
        [0-9][0-9][0-9]) printf '%s\n' "$code" ;;
        *) printf '%s\n' "000" ;;
    esac
}

cmd="${1:-open}"
if [ "$#" -gt 1 ]; then
    die "usage: tools/dashboard_url.sh [open|copy|ip|check]"
fi

case "$cmd" in
    ip)
        addr=$(lan_ip) || die "could not read the LAN address"
        is_ipv4 "$addr" || die "could not read the LAN address"
        printf '%s\n' "$addr"
        ;;
    open)
        open_url "$(dashboard_url)"
        ;;
    copy)
        copy_url "$(dashboard_url)"
        ;;
    check)
        check_url "$(dashboard_url)"
        ;;
    *)
        die "usage: tools/dashboard_url.sh [open|copy|ip|check]"
        ;;
esac
