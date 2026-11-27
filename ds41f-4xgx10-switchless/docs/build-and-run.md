# Building and running the line

This document states how to build this configuration's two binaries with no network, how to run the
line from the build image, how to install the Go runner as a systemd unit, how to run the runner's
tests, and how to read the line's streams, rates and context. Each step gives the command and the
output to expect, so a reader can tell a success from a failure without reading the code.

Every command below runs from `ds41f-4xgx10-switchless/` unless it says otherwise. The build host is
`dgx1` (192.168.1.51, aarch64, Ubuntu 24.04) unless a step says otherwise.

## 1. What the build needs

`build/dependencies.txt` is the single source of truth for what the build reads. Every dependency is
pinned by version and by sha256, and the file ends with one `count` line per kind:

```
count image 1
count apt-direct 11
count apt-transitive 47
count apt 57
count rustup-toolchain 1
count rustup-init 1
count crate 53
```

A record is `<kind> <name> <version> <sha256>`, and an apt record carries a fifth field, the pool URL.
A dependency that is not pinned is a defect: the build image has no network, so a dependency the file
does not name cannot be resolved at build time.

The base image is `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, pinned by the digest
`sha256:689c34b3fdc12303418e90828359fcdb3c7053bc8aa8d7681823186dde1fe572`. It is taken from the build
host's local image cache and never pulled, because the registry mirror at `192.168.14.12:13090`
answered `000` on 2026-11-26.

## 2. Fetch the dependencies

This is the only step that needs network. It runs on a host with network and fills `build/cache/`.

```
./build/fetch-dependencies.sh
```

The script downloads every pinned dependency that `build/cache/` does not hold yet, then checks every
file in the cache against the manifest. A complete cache prints one line and exits 0:

```
OK: 111 files in build/cache, every sha256 matches the manifest
```

A file that does not match stops the script with exit 1 and names that file. A missing or malformed
manifest stops it with exit 2. To check a cache without fetching anything:

```
./build/fetch-dependencies.sh --verify
```

`build/cache-manifest.txt` records the same 111 files with their sha256 and their size, so a cache can
be copied to another host and checked there. The cache holds 57 `.deb` files, 52 `.crate` files, the
Rust toolchain tarball, and `rustup-init.sh`.

## 3. Build the image with no network

```
docker build --network none -t dsgb10-build .
```

The build has 14 layers. The apt layer installs the 57 pinned `.deb` files with a `dpkg` retry loop,
because `apt-get install /tmp/cache/*.deb` fails on a relative path and on the install order. The
front-end layer builds against a cargo local registry made from the 52 `.crate` files, so cargo never
reaches crates.io. A successful build ends with:

```
#17 DONE 11.2s
#19 DONE 55.7s
```

and the log holds no `http`, `https`, `fetch` or `download` line. The image holds the two binaries:

```
docker run --rm dsgb10-build ls -l /out/
-rwxr-xr-x 1 root root 19614600 ... /out/dsfront
-rwxr-xr-x 1 root root 45296432 ... /out/dsgb10_rank
```

The two binaries are built with the pinned Rust 1.97.1, and their sha256 differs from the binaries the
2026-11-26 served line ran, which were built with Rust 1.100.0. The difference is the toolchain, not
the source: the same source built with 1.100.0 gives the served line's hashes.

## 4. Run the line from the image

The image's default command is `/bin/bash`, so a run names what it starts. `dsfront serve` answers
`/v1/models` with 200 and `/health` with 503 until an engine is attached:

```
docker run --rm --gpus all -v ~/dsgb10/weights:/weights:ro \
  -v ~/models/Qwen3.8-27B.Qwen.FP8.safetensors.260911.v1/tokenizer.json:/tokenizer.json:ro \
  dsgb10-build /out/dsfront serve --listen 0.0.0.0:8888 --engine-socket /run/engine.sock \
  --api-key-file /key --tokenizer /tokenizer.json --model-name dsgb10
```

```
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8888/health
503
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8888/v1/models
200
```

`dsgb10_rank` needs `libcuda.so.1` and the RDMA device the fleet names, so it runs only on a GB10 box
with `--gpus all`. On a box without that device it stops at `RDMA device rocep1s0f1 not found`, which
is the expected stop and not a build fault.

## 5. Install the Go runner as a unit

The Go runner is `runner/`, a Go module with four commands:

| Command | What it does |
| --- | --- |
| `dsgb10-runner-server-go` | the supervisor: starts the front-end and the engine, restarts either, lends the ring |
| `dsgb10-runner-server-cli-go` | prints the line's state, uptime, restarts, `/health` code and the last 20 log lines |
| `dsgb10-runner-server-webgui-go` | serves the same reading as a page and as `/state.json` |
| `dsgb10-runner-bootstart-go` | starts the supervisor at boot, detached from the unit's own process |

Build them:

```
cd runner && go build -o ~/dsgb10/bin/ ./cmd/...
```

Install the unit as the user that owns the line, not as root:

```
mkdir -p ~/.config/systemd/user
cp dsgb10-runner.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now dsgb10-runner.service
loginctl enable-linger "$USER"
```

`loginctl enable-linger` is what makes the unit start at boot without a login. Check it:

```
systemctl --user is-enabled dsgb10-runner.service
enabled
systemctl --user is-active dsgb10-runner.service
active
```

The supervisor reads its settings from the environment. The unit sets `FLEET`, `DSGB10_BIN` and
`DSGB10_FRONT_BIN`; the rest have defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLEET` | `fleet/homelab.conf` | the fleet file: the boxes, the ranks, the service directory |
| `DSGB10_ADDR` | `127.0.0.1` | the addresses the front-end listens on |
| `DSGB10_PORT` | `18893` | the front-end's port |
| `DSGB10_KEY_FILE` | none | the API key file; the supervisor refuses to start without it |
| `DSGB10_ENGINE_ARGS` | `--max-ctx 262144` | the arguments for `dsgb10_rank` |
| `DSGB10_BIN` | `dsgb10_rank` | the engine binary |
| `DSGB10_FRONT_BIN` | `dsfront` | the front-end binary |
| `DSGB10_YIELD_IDLE_S` | `600` | lend the ring after this many idle seconds |
| `DSGB10_HOLD_S` | `600` | hold a request this long while the ring is lent |
| `MIN_AVAIL_GIB` | `100` | do not start the engine below this much available memory |
| `DSGB10_KEEP_GIB` | `4` | the page cache to keep |
| `READY_S` | `600` | stop the engine when its socket does not appear in this time |
| `MAX_FAILS` | `5` | stop restarting after this many failures |
| `FAIL_WINDOW` | `3600` | the window those failures are counted in |
| `RUNNER_WEBGUI_LISTEN` | `127.0.0.1:18895` | the web GUI's address |

Read the line with the CLI:

```
dsgb10-runner-server-cli-go
service: serving since 2026-11-28 00:58:28: http://127.0.0.1:18893/v1; engine up since 2026-11-28 00:58:26
  uptime: 0m 5s
  0 restarts
  ring: 2026-11-28 00:58:24 pid 91071 service: dsgb10-runner-server-go, front-end http://127.0.0.1:18893/v1
  front-end: http://127.0.0.1:18893/v1, pid 91073, /health 200 (the engine is connected)
  rank 0: pid 91081, up 0s, MemAvailable 0 GiB
  last 6 lines of service.log:
    ...
```

The CLI exits 0 when the line serves and 1 when it does not, so a script can test it. `--json` prints
the same reading as JSON, which is what the web GUI serves at `/state.json`.

## 6. Run the tests

```
cd runner && go test ./...
ok  	github.com/Picom-code/pico-dgx-ds4/ds41f-4xgx10-switchless/runner	55.255s
```

The 12 tests are integration tests. They run the real binaries against a stub front-end and a stub
engine in a service directory of their own under the system temporary folder, so they touch no line
and no ring:

| Test | What it proves |
| --- | --- |
| `TestCLIReportsServingLine` | the CLI reports the state, the uptime, the restarts, `/health` 200 and exactly the last 20 log lines |
| `TestCLIReportsNotServingLine` | the CLI exits 1 and says so when `/health` is 503 and when no line runs |
| `TestCLIReportsHealthRecovering` | the CLI reports 500 three times and then 200, as the front-end recovers |
| `TestCLIJSON` | `--json` holds every field the web GUI shows |
| `TestWebGUIServesTheSameReading` | the page and `/state.json` report the same line as the CLI |
| `TestCLIReportsTheRates` | the CLI reports the 3 windows, and each rate matches the rate the stub serves |
| `TestCLIJSONCarriesTheMetrics` | `--json` carries the counters and the 3 windows, each resting on at least 2 readings |
| `TestWebGUIShowsTheRates` | the page and `/state.json` carry the streams, the context and the 3 windows |
| `TestSweepRisesWithConcurrency` | the sweep sends its streams together, reports a rate that rises with the concurrency, and repeats at a longer prompt |
| `TestSupervisorStartsAndStops` | the supervisor reaches `serving`, then stops both children on SIGINT |
| `TestSupervisorRestartsTheEngine` | killing rank 0 makes the supervisor start it again and count the restart |
| `TestBootStarterStartsTheSupervisor` | the boot starter starts a supervisor that then serves |

The tests need `python3` on the host, because the stubs are Python programs behind a shell script. A
Unix socket path may hold at most 104 bytes on macOS, so the tests make their service directory under
the system temporary folder with a short name; a long test name would push the socket path past that
limit and the stub engine's bind would fail.

## 7. Read the streams, the rates and the context

The front-end answers `/metrics` with the counters of the line. The CLI and the web GUI read them, and
the sweep command drives them.

```
curl -s http://127.0.0.1:18893/metrics
{"phase":"decoding","streams":3,"streams_encoding":0,"gate_in_flight":3,"gate_waiting":0,
 "gate_passed":5,"gate_refused":0,"encode_tokens":1912,"decode_tokens":19129,
 "context_tokens":9000,"max_context":262144,"engine_connected":true}
```

`streams` is how many streams are in progress now, and `streams_encoding` how many of them have not
sent their first token yet. `phase` is `encoding` while any stream waits for its first token,
`decoding` once one has sent a token, and `idle` when no stream is in flight. `context_tokens` is the
tokens held by the streams in flight, and `max_context` the context window the engine was started
with, so `context_tokens / max_context` is the line's context usage.

The 3 counters `encode_tokens`, `decode_tokens` and `context_tokens` are monotonic since the server
started, so a rate is the difference between 2 readings divided by the time between them, never one
sample. The CLI and the web GUI report the rates over 0.1 s, 0.5 s and 1 s:

```
dsgb10-runner-server-cli-go
  phase decoding, streams 3 (encoding 0), context 9000 of 262144 tokens (3.4%)
     0.1 s: encode    107.4 tok/s, decode   1005.2 tok/s
     0.5 s: encode    101.5 tok/s, decode   1001.5 tok/s
     1.0 s: encode    100.6 tok/s, decode   1000.7 tok/s
```

A window that the readings do not cover yet prints `not covered yet` rather than a rate computed from
one sample. The CLI samples for 1 s before it reports, because it starts with no readings.

The sweep command answers where throughput stops rising. It sends 1, 2, 4, 8, 16 and 32 chat
completions at the same time, reads the counters before and after each step, and repeats the whole
sweep at a longer prompt:

```
dsgb10-runner-sweep-go --levels=1,2,4,8,16,32 --prompt-tokens=512,8192 --max-tokens=128
prompt 512 tokens, answer up to 128 tokens
  streams   ok  fail   wall s   encode tok/s   decode tok/s   total tok/s   per stream   peak ctx
        1    1     0     2.61            196.2           49.0          245.2         49.0        640
        2    2     0     2.58            397.1           99.2          496.3         49.6       1280
```

`per stream` is the decode rate divided by the number of streams: when it falls, the line is
saturated. `peak context` is the largest context the line held during the step, sampled every 100 ms.
The command exits 1 when a stream failed or the counters could not be read, so a sweep can gate a
change.

## 8. What to read next

- `README.md` states what the configuration is and what it measures.
- `docs/engine.md` states what the service does and how a development job borrows the ring.
- `docs/operations.md` states what production runs and when to restart it.
- `build/dependencies.txt` states every dependency and its pin.