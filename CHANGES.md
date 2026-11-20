# Changes in pico-dgx-ds4

Fork of blaesus/dsgx10 by Andy Shu (MIT). Changes by Patrick Moorhead.
Base: upstream `a92b8f4` (master, 2026-10-18). All changes are in `ds41f-4xgx10-switchless/`.

This file is organised as versions. **v2** (2026-11-19) is this release; **v1** is everything that was public before
it and follows below unchanged, change by change.

---

# v2 (2026-11-19)

One line: the engine we serve (lossless speed changes for shared steps, prompt reading, sampled requests and long
code), the three community features merged on 2026-11-15 (thinking options, 100 images per request, video input)
and copy drafts with the `prediction` request field, in ONE tree, tested together end to end on the real 4-Spark
engine. The source in this repo is the source of the binaries we serve.

## v2.1 What changes for someone who uses the endpoint

Speed (4x DGX Spark, TP4 over a 200G RoCE switch, 1M context, 16 slots; decode tok/s):

| what | v1 | v2 | how it was measured |
|---|---|---|---|
| chat, 1 user, held-out prompts | about 90 | about 90 (about +1 %) | paired cold starts on the same prompts, same texts |
| chat, 1 user, 3-prompt bench | 84.6 | 84 (83.6-84.6) | four 9-run medians |
| code, 1 user, 3-prompt bench | 139-144 | about 147 (143-152) | four 9-run medians; this bench swings by about 3 |
| 16 users, a different prompt each | 368-376 | about 413 (408-419) | 8 runs |
| 8 users, a different prompt each | about 302-307 | 340-350 | two runs |
| reading a long prompt (prefill), 48.7K tokens cold | about 10.2K tok/s | about 10.8K tok/s | 8-10 cold starts each |
| a 948K-token prompt, 3 facts recalled | 3/3, 9.4-9.8K tok/s | 3/3, 10.2K tok/s | one run each |
| first token, short prompt | 99 ms | 86 ms | medians, same window |
| sampled agent sessions (temperature 1.0, top-p) | baseline | +5 to 8 % | seeded pairs on identical texts, 1-12 sessions |
| the same prompt sent again gets its first reply | 15 of 17 | 17 of 17 | |
| rewrite of a 1K / 4K / 8K-token file, 1 user | 164 | 365-370 / 459-465 / 468 | identical texts |
| numbered listing, the file sent as `prediction` | 163 | 432 | identical texts |

What did NOT improve, plainly:

- One-user chat. It sits at the hardware's limit (memory bandwidth); the gains of this release are in shared steps
  (several users), prompt reading, sampled requests, longer draft blocks on code, and replies that repeat their prompt.
- Short sampled chat replies (the sampled-request gain is on long agent sessions).
- Several rewrites in flight at once: a copy is only used in a step that holds one request, so two rewrites at
  once gain a few percent and four gain nothing.
- A reply that repeats nothing from its prompt or prediction (a wrong prediction costs nothing and gains nothing).
- The 3-prompt bench swings by about 3 tok/s between runs of one build, and the v1 column was measured on other
  days. Same-window pairs of v1's engine against v2's: 8 users 274 -> 297, 16 users 380 -> 406, long code +2.7 to 3.7 %.

Known costs:

- About 0.3 GiB more memory on the box that runs rank 0 (0.26 GiB of device memory per rank for the copy step graph).
- First token of a follow-up turn: equal on average; one shape (the third turn of a 77K-token conversation) is
  3 ms later, repeatably.
- A text request sent while two 40,000-token image prompts are being read waited up to 13 s (median 0.2 to 1 s).
  With `--max-image-rows-in-flight 40000` (what we serve with) only one such prompt is read at a time: up to 7 s.
  The price of that setting: an image or video request that arrives while a full 40,000-token request is being
  answered gets 429 with `Retry-After: 5` instead of running beside it.
- While several requests are in flight, a reply's wording can differ from the reply the same request gets alone.
  That was already so in v1 (`DSGB10_OEA` re-routes experts in steps shared by 9 or more rows, a lossy opt-in);
  copy drafts change WHEN a request's steps happen, so which requests share a step can change. Drop `DSGB10_OEA`
  for replies that do not depend on the other users.

New features:

| feature | what a client can now do |
|---|---|
| thinking options (pull request #4, MiaAI-Lab) | `reasoning_effort`: none, minimal (1), low (50), medium / high / xhigh (75), ultra (95), max (100), or a number 1-100. `thinking: {"type": "enabled" \| "disabled"}` (or true / false). Any explicit off-switch wins |
| 100 images per request (pull request #5, MiaAI-Lab) | up to 100 images in one request (was 16), bounded by image tokens: 40,000 per request (about 100 small or 40 full-size images) |
| video input (pull request #6, MiaAI-Lab) | `video_url` parts (also `video`, `input_video`): mp4 (h264, mpeg4) and webm, data: or public http(s) URLs, `max_frames` 1-100 (default 32), frames carry their timestamps; several clips, and clips with stills, in one request |
| predicted outputs | `prediction: {"type": "content", "content": "..."}` (OpenAI's shape). The text is used as drafts: same reply, faster when the prediction is right. Copying from the prompt needs no field at all |

## v2.2 Flags and variables

Front-end (`dsfront serve`), new in v2 (defaults in brackets):

    --max-image-rows N              image tokens one request may take [40000]; more is answered 400 before anything
                                    is encoded
    --max-image-rows-in-flight N    image tokens of all image requests being answered [80000 = two full requests];
                                    a request that does not fit gets 429 with Retry-After: 5 at once; text requests
                                    are never counted
    --max-body-mb N                 largest request body in MiB [64]; larger is answered 413
    --thinking / --no-thinking      what a request gets when it does not say [off]
    --prediction                    pass a request's `prediction` to the engine as drafts [off: the field is ignored]
    --prediction-max N              the most prediction tokens sent [65536]

Vision encoder (`tools/vision/vision_encoder.py`): `--max-images` [100], `--max-rows` [40000], `--max-video-bytes`
[128 MiB], `--video-timeout-s` [60]; needs `video_frames.py` beside it.

Engine variables new in v2 (`--setenv NAME=VALUE`; every one defaults to v1's behaviour; the values are the served ones):

| variable = served value | what it does | output |
|---|---|---|
| `DSGB10_REUSE_EXACT=1` | a re-sent prompt gets its first reply (the 1-8 leftover positions take the prefill path) | repeats now equal the first reply |
| `DSGB10_DRAFT_MAX=7`, `DSGB10_DRAFT_DYN=5` | a 7-draft block after a fully accepted step | identical |
| `DSGB10_WIDE_SH=1` | wide steps: the shared expert beside the gate and router | identical |
| `DSGB10_WIDE_AR=2` | wide steps: all-reduce as reduce-scatter + all-gather, the gather in bf16 | identical |
| `DSGB10_L2PF=d:.../c2:...` | L2 prefetch plan for 9-16-row steps and the draft pass | identical |
| `DSGB10_PF_TUNE=hc,small,down` | upstream's prefill tunings | identical |
| `DSGB10_RING_RSBAL=1` | reduce-scatter balanced over both RoCE functions (switch fleets only) | identical |
| `DSGB10_HEADG=2` | the wide LM head reads rows normalised once | identical |
| `DSGB10_GSAMPLE=1`, `DSGB10_GCOUPLE=0.8` | sampled requests WITHOUT a client seed draw through a Gumbel-max sampler and the drafter uses the same noise | same distribution |
| `DSGB10_GATE_SEP=1` | 2-4-row steps: a no-op kernel between the router gate and the MoE | identical |
| `DSGB10_COPY=1`, `DSGB10_COPY_ROWS=64`, `DSGB10_COPY_ENTER=32`, `DSGB10_COPY_SHARED=0`, `DSGB10_COPY_SW=1`, `DSGB10_COPY_DELTA=1`, `DSGB10_COPY_LOG=1` | copy drafts: a reply that repeats its prompt or the client's prediction is verified up to 64 rows a step, in steps that hold one request | identical |
| `DSGB10_EXACT_SOLO=136`, `DSGB10_NORM_EXACT=1`, `DSGB10_WIDE_EXACT=95`, `DSGB10_WIDE_EXACT_MT=2` | the copy steps' own graph with exact 32- and 64-row bodies (row-group kernels with the decode kernels' arithmetic); every other step runs the bodies of before | identical |

Every copy-drafts variable, its default and the design: [docs/copy.md](ds41f-4xgx10-switchless/docs/copy.md). The
full served argument line is in the README's "run it" block.

## v2.3 Behaviour changes (read before upgrading)

Intended, from the merged pull requests (front-end):

1. A `reasoning_effort` (or `thinking.type`) value that is not valid now returns HTTP 400. v1 silently used the
   default. Examples that now fail: "banana", "hihg", 0, 101, true, "0", [].
2. `minimal` means effort 1 and `ultra` means 95. v1 read both as unknown = 75.
3. The request field `thinking` is read. v1 ignored it.
4. When a request carries both an on-switch and an off-switch, off wins. v1 took whichever field it read first.
5. More than 16 images in one request are accepted (up to the image-token caps). v1 answered 400.
6. A new status for clients: an image / video request can get 429 with `Retry-After: 5` when the image requests in
   flight are at the cap. OpenAI-style clients retry it by themselves.
7. Video parts are accepted. v1 answered 400.

Unchanged: thinking is OFF unless a request or `--thinking` turns it on (as in v1); the body limit stays 64 MiB.

Intended, from the engine:

8. A sampled request WITHOUT a seed (the default for most clients) draws from another random stream than v1 (same
   distribution; checked on 6,303 tokens against their rows' probabilities and on 7,200 replies). Such requests
   were never repeatable, so nothing visible changes.
9. The same prompt sent again now gets its first reply (v1: 15 of 17).
10. With `--prediction`, the `prediction` field is read (v1 ignored it). It never changes a reply's text.

Unchanged: greedy requests and requests WITH a seed are token-identical to the same engine without the v2 variables,
copy drafts included (per-token logit bits compared, 0 differ).

## v2.4 Upgrade notes

- Engine, front-end and vision encoder must be updated TOGETHER. The v2 front-end sends requests the v1 engine and
  the v1 encoder refuse (more than 16 images, video, prediction ids).
- The first start of the new engine binary drops the kept sessions once (every open conversation's next turn is
  read again from the start). Changing `--setenv` values later needs `service.sh stop --forget`: variables are not
  part of the kept sessions' identity.
- Memory on the rank-0 box, which also runs the front-end and the encoder (128 GB DGX Spark, 12 GiB session store,
  one encoder, a fresh start, file rewrites running at the same time): about 11.5 GiB free when idle; lowest
  during 100 small images 9.5 GiB, during one request of 40 full-size images 9.2 GiB, during two such requests at
  once (the default `--max-image-rows-in-flight 80000`) 8.8 GiB. Memory follows image tokens, not the number of
  images: on a box with less room lower `--max-image-rows`. Over long uptime the engine's n-gram cache grows to its
  4 GiB per rank (`--engram-cache-mib`, as in v1; under heavy mixed load we saw about 0.3 GiB in 30 minutes, the
  rate falling), so plan for about 4 GiB less than a fresh start shows.
- `DSGB10_RING_DIRECT=2` and `DSGB10_RING_RSBAL=1` only on a switch; the `DSGB10_L2PF` budgets and `DSGB10_GATE_SEP`
  were fitted on GB10.
- Rolling back is a binary + launcher + encoder-file swap; no data format changed.

## v2.5 How it was tested

One request at a time, the build we served before against v2, cold against cold: 54/54 identity prompts, 140/140
held prompts, 36/36 rewrites and numbered listings, predictions right, wrong, partial and malformed, seeded sampled
sessions with a fixed arrival order, 45/45 fixed-answer request shapes, 15/15 images; per-token logit bits equal on
185,707 tokens (0 differ), 58,707 of them computed in 32- and 64-row copy steps. With nobody copying, 2-16 users get
the same texts (68/68 greedy, 32/32 seeded sampled). Rewrites and predictions were also sent together with images
and video and with every thinking variant (34/34 replies equal to the same binaries with copy drafts off), and
while image and video requests were in flight (replies equal except two of about sixty whose wording differed: see
the last known cost above). 100 small / 40 full-size image requests, the caps and the 429 path, the video suite
(39/39); quality eval 71/75 at 4 and 16 users (= v1); 3/3 facts from a 948K-token prompt; a 30-minute soak with
everything mixed (text, sampled agent sessions, rewrites, predictions, images, video, thinking variants,
cancellations): 2,366 requests, no failed request (the only non-200 answers to valid requests were 19 of the 429s
described above).

## v2.7 SPEED-Bench (2026-11-20, measurement only, no code change in the engine)

NVIDIA's SPEED-Bench qualitative split (880 prompts, 1,138 turns, 11 categories) with NVIDIA AIPerf 0.14.0, output cap 8192, temperature 0, on the v2.0 build: one stream 112 tok/s decode as the mean of the categories (coding 140, roleplay 80), 3.05 tokens per verify step; 16 closed-loop streams 245 tok/s in total, held down by a late end of stream that the run exposed (a finished request is released only when its group ends). Tables, method and limits: [docs/speed-bench.md](ds41f-4xgx10-switchless/docs/speed-bench.md); scripts `tools/bench/speed_bench.sh` and `tools/bench/speed_bench_table.py`.

## v2.8 A stream ends when its reply ends: `DSGB10_END_EARLY=1` (2026-11-20; engine, off by default)

The fault (found by the SPEED-Bench run of v2.7): requests that decode together form a group, and the engine sent a
finished request's closing chunk (finish reason, usage, `[DONE]`) only when every request of the group had ended or
a new request arrived. A client that waits for the end of its stream before it sends its next request, as agents
and closed-loop benchmarks do, was held by the longest reply it shared the engine with; with every slot's client
waiting, nobody could be the new request.

The change (`engine/server/serve.cpp`, one file): with `DSGB10_END_EARLY=1` a request's DONE goes out at the token
where it ends (its stop id, its `max_tokens`, or its cancel seen), while the others go on. Its session stays in
flight until the group's run returns, where it is kept exactly as before; from the moment a request has ended, the
next request that arrives pauses the group as an arrival does when a slot is free, and takes a slot. No step's
rows, order or arithmetic change, and without the variable the engine behaves as v2.0.

Measured on a test instance (same boxes, served arguments; closed-loop clients, each sends its next request when
its stream has ended):

| | v2.0 | `DSGB10_END_EARLY=1` |
|---|---:|---:|
| 16 clients, replies of unequal length: last token to end of stream, median / largest | 7.3 s / 18.1 s | under 1 ms / 0.34 s |
| the same: streams producing tokens, of 16 | 8.4 | 15.5 |
| the same: total output, tok/s (16 / 8 / 4 / 2 clients) | 196 / 164 / 131 / 109 | 238 / 191 / 148 / 110 |
| SPEED-Bench, 16 streams, one fifth of the prompts, same window: total | 240 tok/s | 307 tok/s |
| SPEED-Bench, 16 streams, all prompts: total (v2.0: one pass; new: five passes) | 245 tok/s | 303 tok/s |
| SPEED-Bench, 16 streams: last token to end of stream, median / mean | 9.3 s / 24 s | under 1 ms / 34 ms |
| 16 users started together (the README's bench), 4 runs each, alternating | 406-421 | 413-419 |
| one user, 3-prompt bench x9, chat / code medians, 2 runs each, alternating | 84.9, 84.8 / 146, 144 | 85.2, 85.1 / 147, 154 |

What it does not do, plainly:

- It does not make 16 mixed closed-loop streams reach the 16-user figure of the README (about 413). That bench
  starts 16 code prompts together. In a closed loop every new request has its prompt read while the group
  waits, and mixed text accepts fewer draft tokens than code.
- A request's own decode rate at 16 streams is lower than in v2.0's run (about 20 tok/s against 26 on SPEED-Bench):
  16 streams now share the steps where 8 did. The gain is the end of the stream, the total, and the time a set of
  requests takes.
- Texts. A request that runs alone is bit-identical (logit bits compared). Requests that arrive in a fixed order
  with nothing arriving while they run give v2.0's texts. Where clients send as soon as their stream ends, requests
  now arrive at other moments, so other requests share a step; with the re-routing (`DSGB10_OEA`) and the plain wide
  body of shared steps a reply's wording depends on who shares its steps, so texts differ from a v2.0 run there, as
  two v2.0 runs differ from each other (same build twice, 16 closed-loop clients: 33 of 103 replies equal). With
  those two levers off (`DSGB10_OEA` removed, `DSGB10_EXACT_SOLO` removed so that shared wide steps are exact) the
  closed-loop texts of v2.0 and of `DSGB10_END_EARLY=1` are identical (72 of 72 replies) although their timing is
  not.

How it was tested (test instance, cold starts, against v2.0 as served): with the variable off, 54 of 54 fixed
prompts and their logit bits equal. With it on: one request at a time 54 of 54, 140 held-out prompts, rewrites with
copy drafts, predictions, seeded sampled requests, the Gumbel-path pairs and 45 request shapes all equal, logit
bits of over 180,000 tokens with no difference; 2 to 16 users started together and the seeded sampled ruler up to 16
sessions equal; quality set 73 and 71 of 75 at 4 and 16 users (v2.0: 71 and 71); a 948K-token prompt 3 of 3; images,
video and 100-image checks pass; 24 requests on 16 slots answered; agent sessions (8 at 80-300K tokens, 4 turns) finish in the same time (181 s against 182-183 s, two pairs);
a 30-minute mixed soak (text, sampled agent sessions, rewrites, images, video, thinking variants, tool calls, closed-loop
streams, about 300 abandoned or cancelled requests): about 2,600 requests, no failed request (the only non-200 answers
to valid requests were 8 of the image cap's 429s), no error line in the engine's log; memory at the end of the same
load equal to v2.0's.

| variable = value | what it does | output |
|---|---|---|
| `DSGB10_END_EARLY=1` (default 0) | a request's DONE is sent at the step where it ends, not when its group's run returns | identical for a request alone and for fixed arrivals; see above for closed-loop clients |

Tables and method: [docs/speed-bench.md](ds41f-4xgx10-switchless/docs/speed-bench.md), "16 streams with `DSGB10_END_EARLY=1`".

## v2.6 Credits

Thinking options, 100-image requests and video input are pull requests #4, #5 and #6 by MiaAI-Lab, merged with
maintainer commits on top. The engine, kernels, RoCE ring and drafter are Andy Shu's dsgx10 (MIT); `DSGB10_PF_TUNE`
switches on tunings that upstream built. The `prediction` field follows OpenAI's predicted-outputs shape.

---

# v1 (public before 2026-11-19)

Everything below is v1, as it was published.

Measured on four GB10 boxes (TP=4) over a 200G switch, serving DeepSeek-V4.1-Flash-1017, `--max-ctx 1048576`.
Speeds are decode tok/s (total across concurrent requests). "agent" = 8 concurrent sessions with
80K-300K-token prompts, 512 tokens per turn. qeval = 75-question quality eval, bar >= 70.

| # | Change | Files | Default | Lossless | Before -> after |
|---|---|---|---|---|---|
| 1 | 17-bit block index in `select_bins` (prefix split `nlow`, guard `bkeys_stride` <= 131072) | `engine/attn/indexer.cu` | on | yes | max context 512K -> 1M; needle at 947,826 tokens 3/3; prefill 9.2k tok/s; qeval 71/75; c1 unchanged |
| 2 | `DefaultBodyLimit` 64 MB (axum default 2 MB) | `frontend/src/server.rs` | on | yes | HTTP 413 above ~400K tokens -> 948K-token prompts accepted |
| 3 | 16 slots, big/small slot units (`--big-slots`), slot-aware admission, big-slot wait instead of die | `engine/model/*`, `engine/server/{main,serve}.cpp` | `--slots`/`--big-slots` | yes | code c16 172 -> 234-247, c8 170 -> 195-200; c1 unchanged; qeval 71 |
| 4 | 9..16-row normed GEMVs as rmsnorm + plain r16 GEMV (`DSGB10_FUSED16=1` reverts) | `engine/model/model.cu` | on | yes (same bits) | c16 234 -> 253, c4 187 -> 204, step 77.3 -> 74.8 ms |
| 5 | Batch-aware expert re-routing `DSGB10_OEA=K0,KP,MINROWS` | `engine/kernels/{common.cuh,moe.cu}` | off | **no** | with 4,12,9: c16 253 -> 286, agent 180.7 -> 203.4; qeval 71/72 |
| 6 | Cross-session draft row budget `DSGB10_BUDGET=1` | `engine/model/{spec.cu,ops.h,model.h}`, `engine/server/cmd.cpp` | off | yes | c16 253 -> 259, agent 180.7 -> 189.5 |

## End to end

| Build | chat c1 | code c1 | c4 | c8 | c16 | agent | qeval c4/c16 | max ctx |
|---|---|---|---|---|---|---|---|---|
| upstream, 256K, 8 slots | 80 | 140 | - | - | 177 | - | 71 | 256K |
| + 1 + 2, 1M, 4 slots | 78-82 | 134-140 | - | 170 | 172 | - | 71 | 1M |
| + 3, 16 slots (4 big) | 79 | 135-140 | 187 | 197 | 234 | 180.1 | 71/71 | 1M |
| + 4 + 6 (`DSGB10_BUDGET=1`) | 80.4 | 137.8 | 211 | 212 | 259 | 189.5 | 71/71 | 1M |
| + 5 (`DSGB10_OEA=4,12,9`) | 77.5-78.5 | 137.8-139.3 | 239 | 236 | 296 | 215.1 | 71/73 | 1M |

Engine args for the last row: `--max-ctx 1048576 --slots 16 --big-slots 4 --setenv DSGB10_BUDGET=1 --setenv DSGB10_OEA=4,12,9`.

## Wide step (branch wide64, now main)

| # | Change | Default | Lossless | Before -> after |
|---|---|---|---|---|
| 7 | Wide batch step, 17-64 rows (`DSGB10_WIDE=W`): prefill GEMMs + `moe_prefill`, indexer/compressor/head in 16-row groups | off | ~1 ulp | c16 296 -> 350, agent 215 -> 240 |
| 8 | Cost-aware wide gate (`DSGB10_WIDE_COST`) | on with WIDE | yes | wide only when drafts pay |
| 9 | One-pass drafter (`DSGB10_DRAFT1=6`) + coded-head GEMM (`DSGB10_HEADG`) | off / on | ~1 ulp / same bits | 48-row step 120 -> 109 ms |
| 10 | Gate re-fit + `WIDE=64` | - | yes | c16 350 -> 438, c8 295 -> 330, agent 240 -> 260; qeval 71/71; needle 3/3 at 948K |

## Vision

| # | Change | Files | Default | Lossless | Before -> after |
|---|---|---|---|---|---|
| 11 | Image input: side encoder (the checkpoint's `inference/vision.py` + `image_processor.py`, SDPA on contiguous 4D views: MATH -> flash, 3.1 s -> 0.39 s per 444-token image); dsfront `--vision-addr` accepts `image_url`/`image` parts, expands each placeholder into its span; SUBMIT flag 8 carries the spans' rows; prefill injects them, span rows route with `ffn.gate.bias_vl`, Engram dead on spans; decode untouched | `tools/vision/`, `frontend/src/{vision,server,protocol,main}.rs`, `engine/common/frontend_protocol.h`, `engine/model/{prefill,model,ops}.cu`, `engine/server/{serve,cmd}.cpp`, `engine/engram/hash.cpp` | off (needs `--vision-addr`) | text: same path | images refused (400) -> 15/15 image checks, ~0.26 s TTFT per image; text: chat/code c1 79.7/138.4 -> 81.9/144.4, distinct c16 367 -> 380, agent 260 -> 266, qeval 71/71 -> 71/72, needle 948K 3/3 (noise) |

## Single user (c1fast)

The served build is the exact path: chat c1 ~80-82, code c1 ~138-144 (9-run medians). Change 12 below was a lossy
experiment; the code stays in tree, off by default, and is not used.

| # | Change | Files | Default | Lossless | Before -> after |
|---|---|---|---|---|---|
| 12 | Relaxed accept for one-session greedy decode (`DSGB10_RELAX_P=P`: a draft the target gives >= P probability is kept as the row's token; `DSGB10_RELAX=G` logit-gap form) + `DSGB10_RELAX_THRESH=T` fixed verify gate for one-session greedy decodes. Batch steps and sampled requests keep exact accept and the default gate | `engine/model/sample.cu`, `engine/server/cmd.cpp` | off | **no** | **REMOVED from the build 2026-11-09 (outputs must be bit-identical; this was lossy).** Measured with P=0.003, T=0.05: chat c1 81.6 -> 102.8, code c1 140.1 -> 162.9 (our bench, 9-run medians); sparkDash prose 101 -> 145, code 166 -> 172; qeval c1/c4/c16 72/71/71; distinct c16 381, copies 434, distinct c8 311, agent 260 (vis1 same day 262), needle 948K 3/3, images 15/15 |

Method check (sparkDash 1.8.8 at 754f40a, upstream's own sdrun.mjs harness): upstream's README 95 / 151 vs our 80 / 140
was the method (one easier prompt, 256 tokens), not the engine: the exact build scores 101 / 166 on sparkDash. Full
sweep and profile: [docs/c1fast.md](ds41f-4xgx10-switchless/docs/c1fast.md).


## stack4: three small single-user levers (2026-11-10)

All three act on one-session steps only; batch (multi-user) steps run as before. Proof per lever: paired runs on the same 320
held-out prompts (same texts), GPU ms per step by rows, and the 54-prompt identity set on a test port. Design and proofs:
[docs/stack4.md](ds41f-4xgx10-switchless/docs/stack4.md).

| # | Change | Files | Default | Same output | Before -> after |
|---|---|---|---|---|---|
| 13 | `DSGB10_HEAD4_C1=1` with upstream's `--draft-head4 --draft-markov8`: MXFP4 drafter head and FP8 Markov head for one-session drafts, the FP8 head copy kept for batch draft passes | `engine/dspark/dspark.cu`, `engine/model/spec.cu` | off | yes: drafts only; 374/374 test outputs identical | draft pass -0.30..-0.38 ms; 320 prompts chat 86.19 -> 87.17, code 128.20 -> 129.29, math 133.70 -> 134.93; 16 distinct 363-370 (unchanged) |
| 14 | Confidence MLP gate `DSGB10_CF_MLP=FILE`, `DSGB10_CF_ALPHA=A` (one-session drafts): per depth the shipped logit, top-1 margin and top-1 log p (a scan kernel per rank, one all-gather, one MLP kernel), 64 hidden units; trained offline on 661K anchors | `engine/dspark/dspark.{h,cu}`, `engine/model/{spec.cu,ops.h}`, `engine/server/cmd.cpp`, `tools/stack4/` | off | every token is still the verifier's top choice; the gate changes which drafts are verified, which regroups steps and can flip a close call (alpha 1.0: 374/374 identical; alpha 0.9: 319/320 held texts) | accepted drafts per step chat 1.17 -> 1.24; chat 87.17 -> 88.17, code 129.29 -> 130.50, math 134.93 -> 136.11 |
| 15 | Certified LM head `DSGB10_CERT=1` (one-session steps whose rows are greedy without logprobs): FP8 coarse GEMV, per-row rigorous bound, cross-rank threshold, exact coded-head kernel on surviving 16-row groups (`gemv_bf16x_norm_masked`), sampler unchanged; `DSGB10_CERT_CHECK=1` runs the full head beside it | `engine/model/{model.cu,spec.cu,impl.h,model.h}`, `engine/kernels/{gemv16.cu,dsk.h}`, `engine/dspark/dspark.{h,cu}`, `engine/server/{cmd,serve}.cpp` | off | yes: same steps, same token; 1,344,416 real step rows checked on each rank, 0 mismatches, fallback exercised | head -0.18..-0.28 ms a step; chat 88.17 -> 89.45, code 130.50 -> 130.88, math 136.11 -> 137.05 |
| - | Measurement tools: `DSGB10_STEP_LOG`, `DSGB10_FORCE_NTOP`, `DSGB10_CF_DBG`, `DSGB10_CERT_FORCE`, `DSGB10_CERT_BSCALE`, `DSGB10_MH_DUMP` | `engine/server/{cmd,serve}.cpp`, `engine/model/spec.cu` | off | - | - |

Stack vs the exact build, same binary: 320 held prompts chat +3.78 %, code +2.09 %, math +2.51 %; our bench 27-run medians
79.6 / 136.6 -> 80.1 / 142.4; sparkDash c1 prose 99.1 -> 100.5, code 162.9 -> 162.8, catalogs 76.3 / 125.2 / 151.2 ->
79.5 / 130.7 / 154.0; 16 distinct 365-366, copies 446, 8 distinct 302, agent 262, prefill 10.0k, qeval 71 / 71, needle 3/3 at
948K, images 15/15.

Measured and not built: the target's output at a rejected row as the next draft (right 18 % of the time against the drafter's
65 %), a vocabulary shortlist for the drafter's head (+0.3 % simulated on top of #13), 6-7 bit coarse codes for #15 (about
0.07 ms), 7-bit packing of the FP8 dense weights (upstream's costing: no gain at the measured decode cost).

Carryover pilot (2026-11-10, offline only): carried rejected-row states with the shipped drafter frozen raise the accepted
length after a rejection by 2.2 % (3.5 % with small gated adapters) on 12,220 held anchors; realised speed with the current
confidence gate is about zero, so it is not in the engine. Numbers and code: docs/stack4.md, tools/stack4/carryover/.

Live build re-measured on our bench (27 runs, 2026-11-10, no other traffic): chat 82.7, code 142.6 (medians). README
table and charts now show 83 / 143.

Setup and bench (2026-11-10): README "run it" now lists the steps for the exact served build; tools/bench/bench.py is the
bench behind the README numbers. 16 users re-measured on the live build: 368 distinct, 436 copies (medians of 3).

Front-end fix (2026-11-11): a message that contains the image placeholder as ordinary text (a doc, a log, an error quoted
back) used to get HTTP 400 on every later turn. The placeholder is now rewritten with ASCII bars in request text and served
as plain text; image parts are unchanged; error texts no longer quote the token. `frontend/src/encoding.rs`, `server.rs`.

## stack5: sync without relays; coded weights and a 4-bit drafter body measured, not built (2026-11-11)

Three levers got a decisive measurement before any engine work; one shipped. Design, proofs and the two negative results:
[docs/stack5.md](ds41f-4xgx10-switchless/docs/stack5.md).

| # | Change | Files | Default | Same output | Before -> after |
|---|---|---|---|---|---|
| 16 | Direct ring `DSGB10_RING_DIRECT=1\|2` (`comm::Options::direct`, `ring_test --direct`) for fleets where every rank reaches every other in one hop (a switch): each rank also connects to the opposite rank and writes its contribution there itself; proxies relay nothing. Mode 2 also sends a small contribution to a neighbour as one write and one write of both flags. Buffers, flags, device kernels and the order of every sum are unchanged | `engine/comm/{ring.cpp,ring.h}`, `engine/comm/tests/ring_test.cu` | off | yes: same bits; 54/54 identity prompts, 320/320 held texts, identical steps | 10 KB all-reduce 14.45 -> 9.89 us, a step's 80 all-reduces 1.16 -> 0.79 ms (ring bench); engine GPU step -0.21 ms at one row .. -0.59 ms at six; 320 prompts chat 89.16 -> 90.28, code 131.32 -> 133.62, math 136.99 -> 139.46 |
| - | Bench only: a fixed-length lossless FP8 weight code decoded inside the GEMV (`k_gemv_fp8x`), next to upstream's fused Huffman kernel; `bench_wcomp` is now a CMake target | `engine/kernels/tests/bench_wcomp.cu`, `engine/kernels/CMakeLists.txt` | - | bit-exact (bench) | no gain: see below |

Gates with #16 on (test port; relay ring in the same window or stack4's gate run in brackets): 16 distinct 372.5, 373.9
(367.3, 361.6); copies 458.7 (439.1); 8 distinct 307.4 (302); agent 269.5 (262.4); our bench 27-run 82.4 / 143.7
(81.5 / 145.3: within its noise); sparkDash c1 prose 101.9-102.3 (100.3-100.9), code 159.9-165.3 (162.5-163.4), catalogs
80.7 / 132.4 / 156.1 (78.8 / 128.8 / 154.1); prefill 9.0k / 10.2k; qeval 71 / 71; needle 3/3 at 948K; images 15/15.

Measured and not built:
- Lossless coding of the dense FP8 weights, decoded inside the GEMV, on one Spark with our weights. Fused Huffman (upstream's
  kernel, 17 % fewer bytes): 10-48 % slower than today's kernels, as upstream found. A fixed-length 7-bit code with a
  side list for outliers (8.3-8.8 % fewer bytes, every byte exact, outputs bit-identical): -6 % to +6 % against today's
  kernels by site and row count, about par on average; it reads its stored bytes at 216-237 GB/s against 248-258 GB/s raw.
  The bar was 3 % of the dense time.
- The drafter's FP8 body as MXFP4, post hoc, in the torch mirror: accepted length -1.9 % chat, -1.3 % code, -1.4 % math for a
  pass that would be 0.30 ms shorter; first-order simulated gain 0 to +0.2 %. A 22-minute quantisation-aware pilot on
  166 held-out sequences did not close the gap (chat accepted length 1.463 -> 1.464, FP8 1.493); a full fine-tune was not run.

The chat target of 100 tok/s on the 320 held prompts was not reached: 90.3.

## Thinking options, 100 images per request, video input (2026-11-15; pull requests #4, #5, #6 by MiaAI-Lab)

Merged with maintainer commits on top. Tested on the real 4-Spark engine on a test port: output identity 54/54,
images 15/15, 16 users 375-378 tok/s (unchanged), encoder rows byte-identical to the previous encoder's.

| Change | Default | Notes |
|---|---|---|
| `reasoning_effort` is validated; `thinking: {"type": "enabled" \| "disabled"}` is read | thinking stays OFF unless a request or `--thinking` turns it on | unknown values now return 400; any explicit off-switch wins; `minimal` = 1, `ultra` = 95 |
| Up to 100 images per request (was 16) | `--max-image-rows 40000` per request, `--max-image-rows-in-flight 80000` across requests | over the per-request cap: 400 before anything is encoded; over the in-flight cap: 429 with `Retry-After`; body limit stays 64 MiB (`--max-body-mb`) |
| Video input (`video_url`, timestamped frame sampling, 32 frames by default) | on when the vision encoder runs | frames count against the same row caps; unreadable clips, local paths and private addresses return 400 |

Memory on a 4-Spark head: about 1.3 GiB for 100 small images, 0.8 GiB for 40 full-size images in one request (measured).
Two full-size requests at once were not measured: set `--max-image-rows-in-flight 40000` on a head with less than ~10 GiB free.
The engine, the front-end and the vision encoder must be updated together: the new front-end sends requests the older engine and encoder refuse.
