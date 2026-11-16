#!/bin/bash
# The vision encoder as a part of the service (service.sh, DSGB10_VISION_CMD), end to end against the
# mock engine: a test instance of service.sh (DSGB10_MOCK=1) with its own home, ports and ring lock, and
# a stand-in for the encoder that speaks its wire format (docs/vision.md) with no torch and no GPU.
# The mock engine reads a request's prompt and leaves its image part alone, so an image request goes
# all the way: dsfront, the encoder, the expanded prompt, an answer. Nothing here touches the service's
# run directory, the ranks or the real ring lock: ssh and pkill are stand-ins that only log, and the
# test checks that none was called.
#   engine/server/tests/vision_test.sh       # VT_DIR (a short path: Unix sockets), VT_PORT (18896),
#                                            # VT_VISION_PORT (18946), VT_TOKENIZER (a tokenizer.json,
#                                            # where dsfront's default one is not there), VT_KEEP=1
set -u
D=$(cd "$(dirname "$0")/.." && pwd)
SVC=$D/service.sh RING=$D/ring.sh SRC=$(cd "$D/../.." && pwd)
FE=${DSGB10_FRONT_BIN:-$D/../../frontend/target/release/dsfront}
T=${VT_DIR:-${TMPDIR:-/tmp}/dsgb10-vt.$$}
PORT=${VT_PORT:-18896} VPORT=${VT_VISION_PORT:-18946}
pass=0 fail=0
ok() {
  echo "  ok: $*"
  pass=$((pass + 1))
}
bad() {
  echo "  FAIL: $*"
  fail=$((fail + 1))
}
until_() {  # seconds command...: poll every 0.25 s until the command succeeds
  local end=$((SECONDS + $1))
  shift
  until "$@"; do
    ((SECONDS < end)) || return 1
    sleep 0.25
  done
}

[ -x "$FE" ] || {
  echo "no $FE (cargo build --release in frontend/)"
  exit 2
}
for p in "$PORT" "$VPORT"; do
  ss -ltn | grep -q "127.0.0.1:$p " && {
    echo "port $p is in use"
    exit 2
  }
done
rm -rf "$T" && mkdir -p "$T/shim" && chmod 700 "$T"
for c in ssh pkill; do  # nothing here may reach a rank: these only log
  printf '#!/bin/sh\necho "%s $*" >> "%s/shim.calls"\nexit 1\n' "$c" "$T" > "$T/shim/$c" && chmod +x "$T/shim/$c"
done
export PATH=$T/shim:$PATH
[ "$(type -P ssh)" = "$T/shim/ssh" ] && [ "$(type -P pkill)" = "$T/shim/pkill" ] || {
  echo "the stand-ins for ssh and pkill are not first in PATH: stop"
  exit 2
}
head -c 24 /dev/urandom | base64 > "$T/key" && chmod 600 "$T/key"
printf 'Authorization: Bearer %s\n' "$(cat "$T/key")" > "$T/hdr" && chmod 600 "$T/hdr"

# The encoder's stand-in: its wire format, three rows of zeros an image (the shortest span dsfront takes).
cat > "$T/encoder.py" << 'EOF'
import argparse, json, os, socketserver, struct

p = argparse.ArgumentParser()
p.add_argument("--host")
p.add_argument("--port", type=int)
a = p.parse_args()
ROWS, DIM = 3, 5120


class H(socketserver.BaseRequestHandler):
    def need(self, n):
        b = b""
        while len(b) < n:
            c = self.request.recv(n - len(b))
            if not c:
                raise ConnectionError
            b += c
        return b

    def handle(self):
        try:
            while True:
                (n,) = struct.unpack("<I", self.need(4))
                images = json.loads(self.need(n))["images"]
                hdr = json.dumps({"ok": True, "images": [{"rows": ROWS, "hash": i + 1} for i in range(len(images))]}).encode()
                data = bytes(len(images) * ROWS * DIM * 2)
                self.request.sendall(struct.pack("<I", len(hdr)) + hdr + struct.pack("<Q", len(data)) + data)
        except ConnectionError:
            return


class S(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


print(f"stand-in: cwd {os.getcwd()} VISION_PRELOAD {os.environ.get('VISION_PRELOAD')}", flush=True)
srv = S((a.host, a.port), H)
print(f"listening on {a.host}:{a.port}", flush=True)
srv.serve_forever()
EOF
# A command that does not become the encoder but starts it and waits, as a container's client does:
# the encoder is then a child in the command's process group, which a stop has to reach.
printf '#!/bin/sh\npython3 "%s/encoder.py" "$@" &\nwait\n' "$T" > "$T/wrap.sh" && chmod +x "$T/wrap.sh"

TOK=${VT_TOKENIZER:+--tokenizer $VT_TOKENIZER}
export DSGB10_MOCK=1 DSGB10_HOME=$T/svc RING_LOCK=$T/ring.lock DSGB10_ADDR=127.0.0.1 DSGB10_PORT=$PORT
export DSGB10_KEY_FILE=$T/key DSGB10_FRONT_BIN=$FE DSGB10_ENGINE_ARGS="--delay-ms 1 $TOK" DSGB10_FRONT_ARGS="$TOK"
export DSGB10_VISION_PORT=$VPORT DSGB10_YIELD_IDLE_S=3 DSGB10_HOLD_S=30 START_WAIT=60 LOCK_WAIT=60 OWNER=vision_test
unset DSGB10_VISION_CMD VISION_PRELOAD
RUN=$T/svc/run LOG=$T/svc/logs/service.log VLOG=$T/svc/logs/vision.log

PROD_RUN=$HOME/dsgb10/service/run
prod() { echo "supervisor $(cat "$PROD_RUN/supervisor.pid" 2> /dev/null), $(cat "$PROD_RUN/restarts" 2> /dev/null)"; }
PROD0=$(prod)

TEXT='{"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "Say hello."}]}'
IMAGE='{"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": [{"type": "text", "text": "What is this?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]}'
post() {  # name body: a completion; its status on stdout, its answer in $T/<name>.json
  curl -s -m 60 -o "$T/$1.json" -w '%{http_code}' "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H @"$T/hdr" -H 'Content-Type: application/json' -d "$2"
}
listening() { (exec 3<> "/dev/tcp/127.0.0.1/$VPORT") 2> /dev/null; }
state() { cut -d' ' -f1 "$RUN/state" 2> /dev/null; }
serving() { [ "$(state)" = serving ]; }
healthy() { serving && [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ]; }
rank0() { cat "$RUN/rank0.pid" 2> /dev/null; }
vpid() { cat "$RUN/vision.pid" 2> /dev/null; }
encoders() { pgrep -f "$T/encoder.py" | tr '\n' ' '; }  # the stand-ins that run, whoever started them
front_args() { tr '\0' ' ' < "/proc/$(cat "$RUN/front.pid")/cmdline"; }
exits() { grep -c "vision encoder exited" "$LOG"; }

echo "1. without DSGB10_VISION_CMD: as before"
"$SVC" start > "$T/start1.out" 2>&1 && ok "the service starts" || bad "start: $(tail -n 3 "$T/start1.out" | tr '\n' '|')"
[ "$(post t1 "$TEXT")" = 200 ] && ok "a text request: 200" || bad "a text request: $(cat "$T/t1.json")"
c=$(post i1 "$IMAGE")
[ "$c" = 400 ] && grep -q "not supported" "$T/i1.json" && ok "an image request: 400, not supported" || bad "an image request: $c $(cat "$T/i1.json")"
[ ! -e "$RUN/vision.pid" ] && ! listening && [[ $(front_args) != *--vision-addr* ]] && ! grep -q vision "$T/start1.out" &&
  ok "no encoder, no --vision-addr, nothing about it in the status" || bad "an encoder without the setting: $(ls "$RUN") | $(front_args)"
# a service started by a service.sh from before the setting: its run/settings has neither variable
sed -i '/ VCMD=/d; / VPORT=/d' "$RUN/settings"
"$SVC" status > "$T/status1.out" 2>&1 && ! grep -qi "unbound\|vision" "$T/status1.out" && ok "status reads settings from before the encoder" ||
  bad "status with old settings: $(tr '\n' '|' < "$T/status1.out")"
"$SVC" stop > /dev/null 2>&1 && ok "stop" || bad "stop"

echo "2. with it: the encoder is the service's"
export DSGB10_VISION_CMD=$T/wrap.sh
"$SVC" start > "$T/start2.out" 2>&1 && ok "the service starts" || bad "start: $(tail -n 3 "$T/start2.out" | tr '\n' '|')"
until_ 15 listening && ok "the encoder listens on 127.0.0.1:$VPORT" || bad "the encoder does not listen: $(tail -n 3 "$VLOG" | tr '\n' '|')"
V1=$(vpid)
read -r pgid sid <<< "$(ps -o pgid= -o sid= -p "$V1")"
[ -n "$V1" ] && [ "$pgid $sid" = "$V1 $V1" ] && ok "run/vision.pid ($V1) leads its own session and process group" ||
  bad "vision.pid '$V1': process group $pgid, session $sid"
[[ $(front_args) == *"--vision-addr 127.0.0.1:$VPORT "* ]] && ok "dsfront has --vision-addr 127.0.0.1:$VPORT" || bad "dsfront's arguments: $(front_args)"
grep -q "stand-in: cwd $SRC VISION_PRELOAD 1\$" "$VLOG" && ok "it runs in the configuration's directory, with VISION_PRELOAD=1" || bad "vision.log: $(head -n 2 "$VLOG" | tr '\n' '|')"
grep -q "vision encoder: pid $V1, up .*127.0.0.1:$VPORT listening" "$T/start2.out" && ok "status: $(grep 'vision encoder' "$T/start2.out" | sed 's/^ *//')" ||
  bad "status: $(grep -i vision "$T/start2.out" | tr '\n' '|')"
c=$(post i2 "$IMAGE")
[ "$c" = 200 ] && grep -q '"finish_reason"' "$T/i2.json" && ok "an image request: 200" || bad "an image request: $c $(cut -c1-200 "$T/i2.json")"
grep -q "images: 1 (3 rows)" "$T/svc/logs/front.log" && ok "dsfront took its span from the encoder: $(grep -o 'images: 1 (3 rows)' "$T/svc/logs/front.log" | head -n 1)" ||
  bad "front.log: $(grep images "$T/svc/logs/front.log" | tail -n 1)"
[ "$(post t2 "$TEXT")" = 200 ] && ok "a text request: 200" || bad "a text request: $(cat "$T/t2.json")"

echo "3. the encoder dies: image requests answer 503, text goes on, and it is started again"
kill -9 $(encoders)
c=$(post i3 "$IMAGE")
[ "$c" = 503 ] && ok "an image request while it is down: 503 ($(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['error']['message'][:60])" "$T/i3.json"))" ||
  bad "an image request while it is down: $c $(cut -c1-200 "$T/i3.json")"
[ "$(post t3 "$TEXT")" = 200 ] && ok "a text request while it is down: 200" || bad "a text request while it is down: $(cat "$T/t3.json")"
until_ 20 eval '[ -n "$(vpid)" ] && [ "$(vpid)" != "$V1" ] && listening' && ok "started again (pid $V1 -> $(vpid))" || bad "not started again: $(tail -n 3 "$LOG" | tr '\n' '|')"
[ "$(post i3b "$IMAGE")" = 200 ] && ok "an image request: 200" || bad "an image request after the restart: $(cut -c1-200 "$T/i3b.json")"
# the command dies and leaves the encoder behind on the port: it goes with the command's process group
V2=$(vpid) E2=$(encoders)
kill -9 "$V2"
until_ 25 eval '[ -n "$(vpid)" ] && [ "$(vpid)" != "$V2" ] && listening' && ! kill -0 $E2 2> /dev/null &&
  ok "the command killed, its encoder left behind (pid ${E2% }): that one is ended and a new one listens (pid $(vpid))" ||
  bad "after the command's death: vision.pid $(vpid), encoders $(encoders), was $E2| $(tail -n 3 "$LOG" | tr '\n' '|')"
[ "$(post i3c "$IMAGE")" = 200 ] && ok "an image request: 200" || bad "an image request after the second restart: $(cut -c1-200 "$T/i3c.json")"
[ "$(exits)" = 2 ] && [ "$(cat "$RUN/restarts")" = "0 restarts" ] && serving &&
  ok "two exits in service.log, none counted as a failure of the service ($(cat "$RUN/restarts"), $(state))" ||
  bad "$(exits) exits, restarts: $(cat "$RUN/restarts"), state $(state)"

echo "4. the engine restarts, the ring is lent: the encoder stays as it is"
V3=$(vpid) R3=$(rank0)
kill -9 "$R3"  # the mock engine: a failure of the service, which starts it again
until_ 40 eval '[ -n "$(rank0)" ] && [ "$(rank0)" != "$R3" ] && healthy' && [ "$(vpid)" = "$V3" ] && listening &&
  ok "the engine started again (pid $R3 -> $(rank0), $(cut -d';' -f1 "$RUN/restarts")): the encoder is the process it was ($V3)" ||
  bad "after the engine's restart: rank 0 $(rank0), state $(state), vision.pid $(vpid) (was $V3)"
[ "$(post i4 "$IMAGE")" = 200 ] && ok "an image request: 200" || bad "an image request after the engine's restart: $(cut -c1-200 "$T/i4.json")"
"$RING" bash -c "date +%s > $T/job.start; sleep 4" > "$T/job.log" 2>&1 &  # a job that waits for the ring: lent once idle
job=$!
until_ 30 test -s "$T/job.start" && [ "$(state)" = lent ] && [ -z "$(rank0)" ] && [ "$(vpid)" = "$V3" ] && listening &&
  ok "the ring lent to a job, the engine stopped: the encoder is the process it was, and listens" ||
  bad "during the lend: state $(state), rank 0 '$(rank0)', vision.pid $(vpid) (was $V3)| $(tail -n 2 "$T/job.log" | tr '\n' '|')"
wait "$job"
until_ 40 healthy && [ "$(vpid)" = "$V3" ] && [ "$(post i4b "$IMAGE")" = 200 ] && ok "the ring back: serving, the same encoder, an image request: 200" ||
  bad "after the lend: state $(state), vision.pid $(vpid), image $(cut -c1-200 "$T/i4b.json")"

echo "5. stop takes the encoder along"
"$SVC" stop > /dev/null 2>&1 && ok "stop" || bad "stop"
[ ! -e "$RUN/vision.pid" ] && ! listening && [ -z "$(encoders)" ] && ok "no vision.pid, nothing on the port, no encoder left" ||
  bad "after stop: $(ls "$RUN" | tr '\n' ' ')| encoders: $(encoders)"

echo "6. an encoder that cannot start is no failure of the service"
export DSGB10_VISION_CMD=false MAX_FAILS=1  # MAX_FAILS: two failures of the service would end it
"$SVC" start > "$T/start5.out" 2>&1 && ok "the service starts" || bad "start: $(tail -n 3 "$T/start5.out" | tr '\n' '|')"
grep -q "vision encoder: not running" "$T/start5.out" && ok "status: $(grep 'vision encoder' "$T/start5.out" | sed 's/^ *//')" ||
  bad "status: $(grep -i vision "$T/start5.out" | tr '\n' '|')"
until_ 30 eval '[ "$(exits)" -ge 3 ]' && ok "$(exits) exits of the encoder, more than MAX_FAILS" || bad "$(exits) exits in 30 s"
serving && [ "$(cat "$RUN/restarts")" = "0 restarts" ] && ok "still serving, $(cat "$RUN/restarts")" || bad "state $(state), restarts: $(cat "$RUN/restarts")"
[ "$(post t5 "$TEXT")" = 200 ] && [ "$(post i5 "$IMAGE")" = 503 ] && ok "text 200, images 503" || bad "text $(cat "$T/t5.json") | image $(cut -c1-200 "$T/i5.json")"
"$SVC" stop > /dev/null 2>&1 && ok "stop" || bad "stop"
unset MAX_FAILS

echo "7. start refuses what cannot work"
export DSGB10_VISION_CMD=$T/wrap.sh
refused() {  # what words...: start must fail, saying the words, and leave no supervisor
  local what=$1 out
  shift
  out=$("$SVC" start 2>&1) && bad "$what: start went ahead" && "$SVC" stop > /dev/null 2>&1 && return
  [[ $out == *"$*"* ]] && [ ! -e "$RUN/supervisor.pid" ] && ok "$what: $(cut -c1-110 <<< "$out")" || bad "$what: $out"
}
DSGB10_FRONT_ARGS="$TOK --vision-addr 127.0.0.1:1" refused "--vision-addr in DSGB10_FRONT_ARGS as well" "take it out of DSGB10_FRONT_ARGS"
DSGB10_VISION_PORT=$PORT refused "the front-end's port as the encoder's" "is not a port of the encoder's own"
python3 "$T/encoder.py" --host 127.0.0.1 --port "$VPORT" > /dev/null 2>&1 &
other=$!
until_ 10 listening
refused "the encoder's port taken" "127.0.0.1:$VPORT is taken"
# a service without an encoder does not look at the encoder's port
(unset DSGB10_VISION_CMD && DSGB10_VISION_PORT=$PORT "$SVC" start > "$T/start7.out" 2>&1) && [ "$(post t7 "$TEXT")" = 200 ] &&
  ok "without the command, the front-end's port as the encoder's refuses nothing: the service starts" ||
  bad "a service without an encoder: $(tail -n 2 "$T/start7.out" | tr '\n' '|')"
"$SVC" stop > /dev/null 2>&1
kill "$other" && wait "$other" 2> /dev/null

echo "8. a supervisor that died: stop ends what it left, the encoder too"
"$SVC" start > "$T/start8.out" 2>&1 && until_ 15 listening && ok "the service starts, the encoder listens" || bad "start: $(tail -n 3 "$T/start8.out" | tr '\n' '|')"
kill -9 "$(cat "$RUN/supervisor.pid")"
sleep 1
listening && [ -n "$(encoders)" ] && ok "the supervisor killed: its encoder still runs (pid $(vpid))" || bad "the encoder went with the supervisor"
out=$("$SVC" stop 2>&1)
[[ $out == *"stopped what a dead supervisor left"* ]] && until_ 10 eval '! listening && [ -z "$(encoders)" ]' && [ ! -e "$RUN/vision.pid" ] &&
  ok "stop: no encoder left, no vision.pid" || bad "stop said: $out | encoders: $(encoders)| $(ls "$RUN" | tr '\n' ' ')"

echo "9. the service was not touched, nor any rank"
[ -s "$T/shim.calls" ] && bad "ssh or pkill was called: $(head -n 3 "$T/shim.calls" | tr '\n' '|')" || ok "no ssh, no pkill"
[ "$(prod)" = "$PROD0" ] && ok "its supervisor and restart count are as before: $PROD0" || bad "before: $PROD0 | after: $(prod)"
"$SVC" stop > /dev/null 2>&1
[ -z "$(encoders)" ] || kill -9 $(encoders)
[ "${VT_KEEP:-0}" = 1 ] || rm -rf "$T"
echo "$pass passed, $fail failed"
[ "$fail" = 0 ]
