# Copy drafts, the `prediction` request field, and exact wide steps

Status: in the tree behind `DSGB10_COPY` (engine) and `--prediction` (front-end). Both default to off; with them
unset the engine runs the graphs, steps and commands of a build without this feature. The served set of variables
is in the README's "run it" block.

## What it does

When a reply repeats text that is already in the prompt (a file rewrite, an edit, an agent echoing code) or in the
client's `prediction`, the tokens that followed the same text there are offered to the verify step as drafts, in
place of the draft module's few. The model verifies every one of them as it verifies any draft (greedy: the longest
prefix equal to its own samples; sampled: the sampler's own draw per position), so a wrong copy costs rows and never
puts a token in the reply. A request that copies nothing is computed by the same graph bodies as before.

Measured on 4x DGX Spark (TP4, 1M context, 16 slots), one request at a time, decode tok/s:

| workload | off | on |
|---|---|---|
| rewrite of a 1K-token file | 164 | 365-370 |
| rewrite of a 4K-token file | 164 | 459-465 |
| rewrite of an 8K-token file | 169 | 468 |
| numbered listing of a 1K file, the file sent as `prediction` | 163 | 432 |
| the same, a prediction that covers part of the reply | 162 | 225 |
| the same, a WRONG prediction | 164 | 166 |
| chat, code, math (nothing to copy) | 107.1 | 107.2 |

Texts are identical in every row (the logit bits of every emitted token were compared: 0 of 219,356 differ).
Several rewrites in flight at once gain almost nothing: a copy window is only used in a step that holds ONE request
(`DSGB10_COPY_SHARED=0`), so that steps shared by several users stay exactly the steps of before.

## The `prediction` field

`/v1/chat/completions` accepts OpenAI's predicted-outputs shape when the front-end runs with `--prediction`:

    "prediction": {"type": "content", "content": "the text you expect"}
    "prediction": {"type": "content", "content": [{"type": "text", "text": "..."}, ...]}

- It only ever drafts. The reply is the reply the same request gets without the field, token for token; a right
  prediction makes it faster, a wrong one costs nothing measurable.
- Any other shape (another type, wrong value types, empty text, more than 4 MiB) is no prediction, never an error.
- At most `--prediction-max` tokens (65536) are sent to the engine; a longer prediction is cut there.
- A request with images or video ignores the field (the image path sends no prediction).
- Without `--prediction` the field is ignored, as it always was.

## Why it is exact: the two graphs and the exact wide kernels

A position's logits must not depend on how many rows share its step. Steps of 1..8 rows are bit-equal to one-row
steps; the plain wide body (17..64 rows: prefill GEMMs, `moe_prefill`) is not, and neither is a 9..16-row step
(the separate RMSNorm launch sums in another order than the fused 8-row kernels). So copy steps get bodies of
their own:

- `DSGB10_EXACT_SOLO=mask` (bit i = the wide bucket `ops::wide_bucket_rows(i)`: 20, 24, 28, 32, 40, 48, 56, 64 rows).
  The batch step graph is captured as in a build without copy drafts, node for node. Verify steps that carry a copy
  window of ONE request run a separate copy step graph: bodies of 1..8 rows (the kernels of before) plus one body
  per bucket of the mask, captured with the exact kernels, plus the copy tail (long accepts, carry repair).
- `DSGB10_NORM_EXACT=1` and `DSGB10_WIDE_EXACT=95` select the exact kernels and, with the mask set, act in those
  bodies ONLY: the RMSNorm sums in the fused kernels' order, and every family of the step (attention GEMMs, wo_a +
  wo_b, gate, MoE, Engram) runs row-group kernels that keep the decode kernels' arithmetic and read each weight once.
- The layout (`spec_batch.h copy_fit`, the same function on device and host) keeps a one-request step to row counts
  that have an exact body: at most 8 rows, or a bucket of the mask. With mask 136 a copy grows 8 -> 32 -> 64 rows and
  a one-request step of 9..16 rows never runs.
- The expert re-routing (`DSGB10_OEA`) never applies to a step whose rows are one request's.

Unit proof: `test_kernels DIR rg` (every row-group kernel against 8-row calls, bit for bit, at 9 / 16 / 17 / 32 / 48 /
64 rows). Engine proof without a service: a teacher-forced session at R rows a step must print the `samples hash` of
the 1-row run (`--mode batch --batch-tf --setenv DSGB10_TF_SESSROWS=1 --batch-rows R --steps S`, R x S constant).

## Pieces

| piece | where | what |
|---|---|---|
| matcher | `server/copy.h` (host, every rank alike) | an index of 4-token runs over the prompt (kept per session slot, extended when the next prompt starts with the old one) and over the prediction; the longest backward match of the reply's tail picks the source position; a copy goes on while the model's tokens equal the source's |
| copy window | `model/ops.h` `CopyWin`, `StepIn::copy*` | "entry e's token at position p0 + j is tok[j]"; [0] is a token the host knows |
| layout | `model/spec_batch.h` `spec_copy_pick`, `spec_batch_layout`, `copy_fit` | on the device: a window is used when its token at the entry's first row equals the real token there. The host replays the same function for the rows' Engram lookups and dies if the device laid out other rows |
| long accepts, carries | `model/spec.cu` `k_copy_acc`, `k_copy_next`, `k_copy_save`, `k_copy_repair` | accepts longer than 8 rows; the compressor carry written back after a wide step of one entry |
| one request alone | `server/cmd.cpp` `copy_loop` | the speculative loop until `COPY_ENTER` tokens repeat a source, then copy steps, back when copying stops |
| changes of step kind | `DSGB10_COPY_SW=1` | a change waits for the queued step and uses its tokens, no token is fed twice; an unproductive visit to copy steps ends at its first step without a copy, and doubles the match the next visit needs (up to 8x) |
| prompt on every rank | `serve.cpp`, `DSGB10_COPY_DELTA=1` | every rank's matcher needs the prompt; only the tokens the ranks' history of the slot does not hold are sent, with a hash of the rest. A rank whose history differs dies with `copy delta: ...` (as the engine dies on a layout mismatch) rather than draft from another text |
| protocol | `common/frontend_protocol.h`, `frontend/src/protocol.rs` | `kFlagPrediction = 16`: the prediction's ids follow `prompt[]` (before an image part), their count in `SubmitHeader::reserved`. Only requests with a prediction set it |

## Variables

Engine (`--setenv`, the same on every rank). "served" is the value in the README's argument block.

| variable | default | served | meaning |
|---|---|---|---|
| `DSGB10_COPY` | 0 | 1 | 1: on. 0 / unset: the graphs, steps and commands of before |
| `DSGB10_COPY_ROWS` | 16 | 64 | rows of a step at most when one request copies |
| `DSGB10_COPY_SHARED` | = ROWS | 0 | the same when several requests share the step; 0: no copy window in a shared step |
| `DSGB10_COPY_ENTER` | 16 | 32 | one request alone: matched tokens that move it to copy steps |
| `DSGB10_COPY_SW` | 0 | 1 | cheap changes of step kind (above) |
| `DSGB10_COPY_DELTA` | 0 | 1 | the prompt's delta to the other ranks (above) |
| `DSGB10_EXACT_SOLO` | 0 | 136 | mask of wide buckets with an exact body for one-request steps (136 = 32 and 64 rows) |
| `DSGB10_NORM_EXACT` | 0 | 1 | RMSNorm in the fused kernels' order (with the mask: in the exact bodies only) |
| `DSGB10_WIDE_EXACT` | 0 | 95 | bits 1 attention GEMMs, 2 wo_a + wo_b, 4 gate, 8 MoE, 16 Engram through the decode kernels' arithmetic; + 64: the row-group kernels (each weight read once) |
| `DSGB10_WIDE_EXACT_MT` | 1 | 2 | wo_b's row-group kernel with two tiles of weight rows per warp (the same bits) |
| `DSGB10_COPY_LOG` | 0 | 1 | a `copy:` line per request that ran alone, in rank 0's log |
| `DSGB10_COPY_MATCH` | 8 | - | tokens that must match before a copy is offered |
| `DSGB10_COPY_N0` | 15 | - | drafts of the first copy step; a copy accepted whole doubles it |
| `DSGB10_COPY_RESET` | 0 | - | after a rejected copy: back to N0 (0) or half the width (1) |
| `DSGB10_COPY_MODE` | 0 | - | 1: only when the draft module's own gated drafts agree with the copy |
| `DSGB10_COPY_EXIT` | 24 | - | one request alone: steps without a copy that move a productive visit back |
| `DSGB10_COPY_BACKOFF` | 1 | - | each copy in a row that is not accepted whole doubles the match the next one needs |
| `DSGB10_COPY_MINTOK` | 0 | - | a reply's first N tokens never copy |
| `DSGB10_OEA_SOLO` | 0 | - | 1: re-route one-request steps of 9+ rows too (lossy for copy steps; do not set) |
| `DSGB10_BITS_LOG`, `DSGB10_TF_SESSROWS`, `DSGB10_TF_SOLO`, `DSGB10_R16_EXACT`, `DSGB10_WIDE_EXACT_GQ` | unset | - | test hooks |

Front-end: `--prediction` (off: the field is ignored), `--prediction-max N` (65536 tokens).

## Costs to know

- Device memory: the copy step graph takes about 0.26 GiB on every rank at start (10 bodies). On the box that also
  runs the front-end and the encoder, free memory is about 0.3 GiB lower than without the feature.
- Host memory: the matcher's per-slot prompt index, about 32 bytes a prompt token per slot, bounded once every slot
  has held its longest prompt (about 0.2 GB a rank at 4 x 1M + 12 x 128K).
- First token of a follow-up turn: equal on average (within 2 ms); one shape (the third turn of a 77K-token
  conversation) was 3 ms later in 3 of 3 pairs.
- One `copy:` line per request in rank 0's log with `DSGB10_COPY_LOG=1`.
- While several requests share steps a reply's wording can differ from its solo reply, as without the feature
  (`DSGB10_OEA` re-routes experts in shared steps of 9+ rows); copy drafts change when a request's steps happen.
- A new way to die, by design: `copy delta: ...` when a rank's slot history differs from rank 0's (never seen in
  test, including soaks with cancellations, 16-user bursts and slot reuse).
