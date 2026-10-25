#!/usr/bin/env bash
# Mocked host-side test for wait_for_marker in scripts/bench-xbox-ort.sh.
# No console and no network: curl/sleep/date are mocked, the function text is
# extracted from the real script so the test cannot drift from the source.
#
# Covered: exact "done\n" success, no-newline "done" (trimmed), HTTP 404,
# partial 200 body, fail-then-succeed, deadline with timeout 0 (no requests
# started), and bounded polling under a partial body. Run:
#   bash tests/test_wait_for_marker.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT="$HERE/../scripts/bench-xbox-ort.sh"

FN=$(awk '/^wait_for_marker\(\) \{/{f=1} f{print} f&&/^\}/{exit}' "$SCRIPT")
[[ -n "$FN" ]] || { echo "extract failed: wait_for_marker not found in $SCRIPT" >&2; exit 1; }
eval "$FN"

MOCK_DIR=$(mktemp -d)
trap 'rm -rf "$MOCK_DIR"' EXIT
MOCK_NOW="$MOCK_DIR/now"
MOCK_COUNT="$MOCK_DIR/count"
# Used by the eval'd wait_for_marker body, invisible to shellcheck.
# shellcheck disable=SC2034
TMPDIR_LOCAL="$MOCK_DIR"
# shellcheck disable=SC2034
BASE_URL="https://mock"
# shellcheck disable=SC2034
CURL_AUTH=()
# shellcheck disable=SC2034
PFN="mock.pfn"

# --- mocks -----------------------------------------------------------------
curl() {
	local out=""
	while [[ $# -gt 0 ]]; do
		case "$1" in
		-o)
			out="$2"
			shift 2
			;;
		-w)
			shift 2
			;;
		*)
			shift
			;;
		esac
	done
	local n
	n=$(cat "$MOCK_COUNT")
	n=$((n + 1))
	echo "$n" >"$MOCK_COUNT"
	case "$MOCK_MODE" in
	done)
		printf 'done\n' >"$out"
		printf '200'
		;;
	done_no_newline)
		printf 'done' >"$out"
		printf '200'
		;;
	http_error)
		printf 'Not Found' >"$out"
		printf '404'
		;;
	partial_200)
		printf 'don' >"$out"
		printf '200'
		;;
	error_then_done)
		if ((n < 3)); then
			printf 'don' >"$out"
			printf '500'
		else
			printf 'done\n' >"$out"
			printf '200'
		fi
		;;
	esac
	return 0
}
sleep() {
	local s="${1:-0}"
	echo $(( $(cat "$MOCK_NOW") + s )) >"$MOCK_NOW"
}
date() {
	if [[ "${1:-}" == "+%s" ]]; then
		cat "$MOCK_NOW"
	else
		command date "$@"
	fi
}

# --- harness ---------------------------------------------------------------
pass=0
fail=0

run_case() { # mode timeout_s
	local mode="$1" timeout_s="$2"
	MOCK_MODE="$mode"
	echo 1000 >"$MOCK_NOW"
	echo 0 >"$MOCK_COUNT"
	local out rc calls
	out=$(wait_for_marker "marker.done" "$timeout_s" 2>&1) && rc=0 || rc=$?
	calls=$(cat "$MOCK_COUNT")
	printf '%s' "$out" >"$MOCK_DIR/last-output"
	echo "$rc $calls"
}

expect() { # desc got want
	if [[ "$2" == "$3" ]]; then
		pass=$((pass + 1))
		echo "PASS: $1"
	else
		fail=$((fail + 1))
		echo "FAIL: $1 (got '$2' want '$3')"
	fi
}

# 1. exact done-newline
read -r rc calls <<<"$(run_case 'done' 30)"
expect "done-newline succeeds" "$rc" "0"
expect "done-newline one request" "$calls" "1"

# 2. done without trailing newline must also be accepted (body trimmed)
read -r rc calls <<<"$(run_case 'done_no_newline' 30)"
expect "done-no-newline accepted" "$rc" "0"

# 3. HTTP 404 is a failed poll, never a marker
read -r rc calls <<<"$(run_case 'http_error' 30)"
expect "http-404 times out" "$rc" "1"
expect "http-404 did poll" "$((calls >= 1 ? 1 : 0))" "1"
expect "http-404 diagnostic present" "$(grep -c 'Timeout waiting' "$MOCK_DIR/last-output")" "1"

# 4. partial 200 body is not success; polling stays bounded by the deadline
read -r rc calls <<<"$(run_case 'partial_200' 30)"
expect "partial-200 times out" "$rc" "1"
expect "partial-200 bounded requests" "$((calls >= 2 && calls <= 4 ? 1 : 0))" "1"

# 5. two failed polls then a valid marker
read -r rc calls <<<"$(run_case 'error_then_done' 60)"
expect "fail-then-ok succeeds" "$rc" "0"
expect "fail-then-ok third request" "$calls" "3"

# 6. timeout 0 starts no request at all (deadline checked before the request)
read -r rc calls <<<"$(run_case 'partial_200' 0)"
expect "zero-timeout returns failure" "$rc" "1"
expect "zero-timeout no requests" "$calls" "0"

echo
echo "wait_for_marker mocked tests: ${pass} passed, ${fail} failed"
[[ "$fail" -eq 0 ]]
