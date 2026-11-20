#!/bin/bash
# SPEED-Bench (qualitative split) with NVIDIA AIPerf against an OpenAI-compatible endpoint,
# plus timestamped copies of the engine's per-request log lines so that draft acceptance
# can be attributed to categories afterwards (tools/bench/speed_bench_table.py).
# Results and method: ds41f-4xgx10-switchless/docs/speed-bench.md
#
# usage: speed_bench.sh NAME CONCURRENCY INPUT.jsonl [extra aiperf args...]
#   NAME         run name; artifacts go to runs/NAME/artifacts, logs to runs/NAME/logs
#   CONCURRENCY  1 for the one-stream table, 16 for the 16-stream table
#   INPUT.jsonl  a prepared SPEED-Bench file (AIPerf's `--format raw` rows: category + messages).
#                Prepare it on a machine with spare memory and disk (the sources are about 40 GB),
#                never on the serving host, and copy the 2 MB result. A later file holding only
#                some categories is run the same way under another NAME; the table script merges
#                run directories.
# environment:
#   AIPERF     path of the aiperf executable (0.14.0 or later: SPEED-Bench support)   [aiperf]
#   URL        endpoint base URL                                                      [http://127.0.0.1:8000]
#   MODEL      served model name                                                      [deepseek-v4.1-flash]
#   KEY_FILE   file holding the Bearer key (read at run time, never stored)           [none]
#   LOG_DIR    directory of the engine's rank.0.log and front.log                     [none: no acceptance]
#   MEM_MAX    memory cap of the AIPerf processes (systemd scope, no swap)            [3G]
#   OSL        output token cap                                                       [8192]
set -u
NAME=$1; CONC=$2; IN=$3; shift 3
AIPERF=${AIPERF:-aiperf}; URL=${URL:-http://127.0.0.1:8000}; MODEL=${MODEL:-deepseek-v4.1-flash}
MEM_MAX=${MEM_MAX:-3G}; OSL=${OSL:-8192}
D=runs/$NAME; mkdir -p "$D/logs"
ts() { while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%3N)" "$l"; done; }
if [ -n "${LOG_DIR:-}" ]; then
  ( tail -F -n 0 "$LOG_DIR/rank.0.log" 2>/dev/null | grep --line-buffered -E "serve: req [0-9]+ (drafts|session)" | ts > "$D/logs/$NAME.rank.tlog" ) &
  ( tail -F -n 0 "$LOG_DIR/front.log" 2>/dev/null | grep --line-buffered -E "^req [0-9]+ session" | ts > "$D/logs/$NAME.front.tlog" ) &
fi
KEYARG=()
if [ -n "${KEY_FILE:-}" ]; then KEYARG=(--api-key "$(cat "$KEY_FILE")"); fi
date -u +"START %FT%TZ" > "$D/logs/$NAME.time"
# No request count: AIPerf then makes one full pass over every conversation of the file, every
# turn once (a count smaller than the number of TURNS silently drops whole categories).
systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 \
  "$AIPERF" profile \
    --model "$MODEL" \
    --endpoint-type chat \
    --streaming \
    --url "$URL" \
    "${KEYARG[@]}" \
    --input-file "$IN" \
    --custom-dataset-type speed_bench_qualitative \
    --osl "$OSL" \
    --extra-inputs temperature:0 \
    --concurrency "$CONC" \
    --request-timeout-seconds 1800 \
    --no-server-metrics --no-gpu-telemetry \
    --export-level raw \
    --workers-max 2 --record-processors 1 \
    --ui none \
    --output-artifact-dir "$D/artifacts" "$@" > "$D/logs/$NAME.log" 2>&1
echo "EXIT $?" >> "$D/logs/$NAME.time"
date -u +"END %FT%TZ" >> "$D/logs/$NAME.time"
pkill -P $$ 2>/dev/null
exit 0
