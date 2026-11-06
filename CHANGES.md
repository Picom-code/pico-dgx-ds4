# Changes in pico-dgx-ds4

Fork of blaesus/dsgx10 by Andy Shu (MIT). Changes by Patrick Moorhead.
Base: upstream `a92b8f4` (master, 2026-10-18). All changes are in `ds41f-4xgx10-switchless/`.

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

## GLP steering (branch glp/steer-v1)

| # | Change | Default | Lossless | Before -> after |
|---|---|---|---|---|
| 11 | GLP spec-v1 steering at `residual_stream_post_layer` (`WEIGHTLESS_STEER_PATH`) | off | off: no launch, no read; `alpha 0` is the identity | not measured yet |

A GLP file holds one unit direction per decoder layer. At the end of layer L, after its write has been
folded into the four hc streams (`h = bf16(post[L] sub + sum_m comb[m] res[m])`), each stream is projected
off `direction.L`:

    h[n] <- bf16(h[n] - alpha * (h[n] . d) * d),   n = 0..3

which is the operation Owen's engine applies in `models/deepseek_v4_1/glp.py` + `tilelang/glp.py`
(spec v1, hook `residual_stream_post_layer`, `alpha` from `glp.alpha_default`, layers from the file). The
direction is applied after layer L and left of layer 0 unsteered, i.e. `direction.1..39` steer layers 1..39.

Where it runs: the decode/verify step steers the streams `ops::allreduce_hc` materialised for the next block
and re-collapses `yraw` from the projected streams with the same `pre` (so the next sublayer's norm, and the
head's final collapse on the last layer, read the steered stream; DSpark's `spec_hidden` and an Engram
layer's write already come after it); a prefill chunk splits the boundary in two, `hc_post` -> `glp_steer` ->
the next sublayer's `hc_pre` + norm + mixes, so both paths steer the same point of the forward.

Off unless `WEIGHTLESS_STEER_PATH` names a GGUF v3 file, which is read once at startup with the same
fail-closed checks Owen's reader makes: `glp.mode == "project"`, `glp.spec_version == 1`, rank 1, hook
`residual_stream_post_layer`, no `glp.dir_scales`/`glp.layer_scales`, one `direction.<L>` per layer of width
5120 (1-D F32, finite, unit norm within 1e-3), a layer list and `glp.content_sha256` that agree with the
tensors, and nothing but alignment padding after the last tensor. `WEIGHTLESS_STEER_ALPHA` and
`WEIGHTLESS_STEER_LAYERS` override the file and are refused when malformed. A file the reader cannot apply
exactly stops the rank at startup instead of serving unsteered.

`engine/model/tests/glp_test.cpp` (host only, no GPU) checks the reader against the real GLP-39 export
(39 directions, layers 1..39, alpha 0.75, worst |norm-1| 2.9e-8), six synthetic refusals, the env contract
(unset = off, overrides applied and refused), `alpha 0` bit identity and the projection's orthogonality.
