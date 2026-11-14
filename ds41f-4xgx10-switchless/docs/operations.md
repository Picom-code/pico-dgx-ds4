# Operations

How production runs on rank 0:8888: the build it runs and how that build was validated, the command that
starts it, when to restart it, and how to bring the SGLang kit back. Until 2026-10-13 this was the end of
the configuration [README](../README.md)'s Status section. What the service does is in
[engine.md § As a service](engine.md#as-a-service-servicesh), and how development jobs borrow the ring
in [engine.md § Borrowing the ring](engine.md#borrowing-the-ring).

**dsgb10 serves DeepSeek-V4.1-Flash (dealignai's variant, as in the [README](../README.md)) on rank 0:8888 since 2026-10-12 06:39** (M9, the owner's go-ahead), in
place of the SGLang kit, which was stopped on 2026-10-11 at ~14:55 and is dormant (nothing removed). It
runs from a pinned worktree, with
`~/dsgb10/prod/ds41f-4xgx10-switchless/engine/server/service.sh status | stop | restart` (settings in
its header). Since 2026-10-13 04:13 the worktree and its supervisor are at c247a78, and the binaries are release
candidate 5, built from it (the fix of the pipelined reduce-scatters' send-buffer race; B′, the decode step's paced L2
prefetch, on by default: 16.71 ms a token at k = 1 against 17.47 with `--no-l2pf`; the exact weight codes; B2's scheduler at
`--slots 1`) and validated on the ring (the 7 reference hashes, serve_test 17 of 17, B1's slice test, kept sessions across a
stop, a restart and builds, yield_test 76 of 76, the address list 25 of 25, and 538 stress requests with 0 faults): engine
b352b870…, dsfront ce116cea…, in `~/dsgb10/prodbin` (read-only; `ENGINE_BUILT_FROM` records them and the validation; no build
writes there; the previous binaries, release candidate 4's engine 2da015a5 and dsfront 5307a372, are in `prev-2da015a5/`, older
ones in `prev-198923bb/`, `prev-51f08122/` and `prev-c2dd1421/`). A new engine binary does not take back another binary's kept
sessions (run ids hash the binary): at a deploy the kept conversations come back cold once, with correct outputs. An engine
built from master since the fleet merge (5d79697) reads `fleet.conf` beside it, which only `service.sh` and `run4.sh` from
that merge on install: deploy a binary with its own commit's scripts (under an older `service.sh` it stops at its start,
"cannot read the fleet file"). Every engine
before this one had a rare write-after-read race in the pipelined reduce-scatters of prefill chunks of 2,433 rows or more, seen
once, as one wrong first-token logprob in RC4's first validation run. This one has the fix (1128401), and ring_test's defaults
force the race's window as a regression test
([ring § The pipelined reduce-scatters' send-buffer race](ring.md#the-pipelined-reduce-scatters-send-buffer-race-2026-10-13)).
The owner put development first (2026-10-12): the ring is lent after 2 idle minutes,
and a request held meanwhile waits up to 15. Production starts, from `~/dsgb10/prod/ds41f-4xgx10-switchless`, with Release candidate 7 (`~/dsgb10/candbin7`, built from f7c8f81: this engine's code with the teacher-forced
hash's fix b47c510 and the fleet file; `ENGINE_BUILT_FROM` there) is validated and not deployed: every reference hash on
the variant files (8 of 8) and on the official checkpoint (8 of 8: tf c7605e0b6ba8d9d2, k = 1 7941647b162df1a8 and the drafted
variants, sampled 02ef3e0858d7eb45, 16k CED 8790ee9388f4abd8, 28k CED de99bfce6ea96ba8), and serve_test `--ced` on the
official checkpoint 17 of 17 (2026-10-13, 12:55–16:31, between the kit's turns). Deploying it is the owner's call. Master
since 5666f78 and d8912e6 (2026-10-13) also has B3, drafts across sessions: with `--slots 4` up to eight requests decode
together, each with its own drafts (109 / 127 / 136 tok/s in all at 2 / 3 / 4 streams on the variant weights, one stream
unchanged; [batching.md](batching.md#b3-the-build-2026-10-13)); production stays on one slot until the owner deploys a build
from it.

```
DSGB10_PROD=1 DSGB10_ADDR="127.0.0.1 <tailnet address>" DSGB10_PORT=8888 DSGB10_KEY_FILE=~/ai/dsv41-4x-ring/state-tp4/api-key \
  DSGB10_ENGINE_ARGS="--max-ctx 262144" DSGB10_NOTE="production: dsgb10 on 127.0.0.1:8888 and <tailnet address>:8888 (M9), ~/dsgb10/prod" \
  DSGB10_BIN=~/dsgb10/prodbin/dsgb10_rank DSGB10_FRONT_BIN=~/dsgb10/prodbin/dsfront \
  DSGB10_KEEP_GIB=4 DSGB10_YIELD_IDLE_S=120 DSGB10_HOLD_S=900 engine/server/service.sh restart
```

Checked at 04:13 on both addresses: models, a plain answer, thinking at `max`, a tool call, streaming, and 401
without the key. Through dsfront, a "pong" sent while a 600-token answer decoded came back in 0.17 s (B1; the answer
took 8.01 s). The service was down about a minute. Restart only when no `*.held` want is in `/data/dsgb10/ring.wants`: a
job's driver that holds a want across its parts keeps the service from taking the ring back until the driver ends (release
candidate 4's start waited 44 minutes so). To bring the kit back: `service.sh stop`, then
`cd ~/ai/dsv41-4x-ring && nohup setsid ./start-tp4.sh serve > logs-tp4/serve-launch.log 2>&1 < /dev/null &`
on rank 0 (ready in ~6 min).

## The `/metrics` endpoint

dsfront answers `GET /metrics` on the same addresses and port as the API, unauthenticated like
`/health`, in the Prometheus text exposition format (`text/plain; version=0.0.4`). The series keep the
`tensorfold:` prefix of the serve they replace, so a scraper already reading the TensorFold counters
reads these unchanged:

| series | type | what it reports |
|---|---|---|
| `tensorfold:requests_running` | gauge | requests in flight in the gate |
| `tensorfold:requests_waiting` | gauge | requests waiting in the gate |
| `tensorfold:generation_tokens_total` | counter | output tokens, added as the engine streams them |
| `tensorfold:prompt_tokens_total` | counter | the full prompt length of each served request |
| `tensorfold:prompt_tokens_cached_total` | counter | the part of that prompt whose KV was reused |
| `tensorfold:time_to_first_token_seconds_sum`, `_count` | summary | time to first token over completed requests |
| `tensorfold:request_prefill_seconds_sum`, `_count` | summary | the engine's own prefill time (prefill start -> first token) over completed requests |

`prompt_tokens_cached_total` is a **subset** of `prompt_tokens_total`, not a counter beside it: it
counts the prompt tokens whose KV was reused, and those sit inside the prompt total. A scraper derives
the prefill rate as `prompt_tokens_total − prompt_tokens_cached_total`; if the two were a partition
instead, that rate would come out inflated by the cache share — on a cache-heavy agent lane, by more
than an order of magnitude. The front end adds the full prompt length to the prompt total and only the
engine's `Done.cached_tokens` to the cached counter, clamped at the update site to the prompt it was
handed ([protocol.md](protocol.md) defines the field), so the relation holds even if an engine ever
reports a reuse length longer than the prompt.

`generation_tokens_total` advances while a request decodes, so it includes the tokens of a request
still in flight; the prompt, cached and TTFT series are recorded once, when the request completes.

`request_prefill_seconds` is the prefill time the **engine** measured (`Done.prefill_us`, prefill started
to first token), not TTFT: TTFT is measured from the front end and also covers the gate and the submit
round trip, so it is the larger of the two and dividing by it understates the prefill rate. A scraper
derives the rate as `d(prompt_tokens_total - prompt_tokens_cached_total) / d(request_prefill_seconds_sum)`
over its window: the computed tokens divided by the time spent computing them. The prompt, cached,
prefill and TTFT series are all recorded once per completed request, so the deltas of one window cover
the same requests.

The counters are per-process and begin at zero when dsfront starts. **A restart resets them all to
zero, together with the session and prefix-cache state**, so a scraper should read a drop to zero as a
restart, not as a traffic drop: they return to zero together, never one without the others. There is
no per-slot or queue-depth gauge beyond the two request gauges.
