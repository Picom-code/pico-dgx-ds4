# SPEED-Bench (qualitative split) with NVIDIA AIPerf

Measured 2026-11-20 on the build of tag `v2.0`, as it serves. One run per cell. The 16-stream run exposed a late
end of stream; the engine variable `DSGB10_END_EARLY=1` removes it, and the 16-stream section has the
same benchmark with it ([16 streams with `DSGB10_END_EARLY=1`](#16-streams-with-dsgb10_end_early1)).

SPEED-Bench is NVIDIA's benchmark for speculative decoding
([dataset](https://huggingface.co/datasets/nvidia/SPEED-Bench)). Its *qualitative* split is 880 prompts in
11 kinds of text, 80 each; some are conversations of several turns, 1,138 turns in all. It answers a
question our own bench does not: how fast is the engine on each kind of text, when the reply may be as
long as the model wants. Speculative drafts make that depend on the text: the more predictable the next
words, the more draft tokens a step accepts.

## The short version

| | one stream | 16 streams, `v2.0` | 16 streams, `DSGB10_END_EARLY=1` |
|---|---:|---:|---:|
| decode tok/s per request, mean over the 11 categories | 112.3 | 26.2 | 20.0 |
| the same, median request | 109.9 | 24.3 | 19.3 |
| fastest and slowest category (mean) | 140.5 coding, 80.0 roleplay | 36.0 math, 20.6 roleplay | 26.7 math, 14.4 roleplay |
| first token, median | 137 ms | 1,051 ms | 486 ms |
| tokens per verify step, all steps | 3.05 | 2.69 | 2.83 |
| draft acceptance rate, all steps | 75.9 % | 80.4 % | 81.1 % |
| streams producing tokens, on average | 1 | 8.3 of 16 | 15.4 of 16 |
| last token to end of stream, median (mean) | 24 ms | 9.3 s (24 s) | under 1 ms (34 ms) |
| total output of the endpoint | - | 245 tok/s | 303 tok/s |
| requests sent / failed | 1,138 / 0 | 1,138 / 0 | 1,135 / 0 |

Decode tok/s is measured per request from its first token to its last. The `v2.0` 16-stream total is held
down by a late end of stream: see [what the 16-stream total shows](#what-the-16-stream-total-shows). With
`DSGB10_END_EARLY=1` all 16 streams produce tokens, so each one's share of a step is smaller (20 tok/s per
request against 26) while the endpoint's total is higher (303 against 245) and a set of requests is done
sooner.

## What was run

| | |
|---|---|
| benchmark | SPEED-Bench v1.1, `qualitative` config, dataset revision `fa4e669`: all 880 prompts, 1,138 turns |
| tool | NVIDIA AIPerf 0.14.0, built from its git main at `ebcdc98` (the release on PyPI that day, 0.13.0, has no SPEED-Bench loader) |
| method | AIPerf's SPEED-Bench tutorial: chat endpoint, streaming, temperature 0, one full pass over every turn, a conversation's earlier replies fed back as history. One change, on purpose: output cap 8192 tokens instead of the tutorial's 4096 |
| model | DeepSeek V4.1 Flash, thinking off (the endpoint's default; 0 reasoning tokens in every reply) |
| engine | this repository at tag `v2.0`, served as in the README's "run it": speculative drafts from the model's own draft head (up to 7 draft tokens a step), copy drafts on |
| hardware | 4x NVIDIA DGX Spark, tensor parallel over a 200G RoCE switch |
| slots | 16 sessions (4 of 1M context, 12 of 128K) |
| client | AIPerf on rank 0's own CPU, against the engine's port on localhost |

```bash
aiperf profile \
    --model deepseek-v4.1-flash \
    --endpoint-type chat \
    --streaming \
    --url http://127.0.0.1:PORT \
    --api-key "$KEY" \
    --input-file qualitative.jsonl \
    --custom-dataset-type speed_bench_qualitative \
    --osl 8192 \
    --extra-inputs temperature:0 \
    --concurrency 1 \
    --tokenizer MODEL_DIR \
    --request-timeout-seconds 1800 \
    --no-server-metrics --no-gpu-telemetry \
    --export-level raw \
    --workers-max 2 --record-processors 1 \
    --ui none \
    --output-artifact-dir ./artifacts/c1
```

The 16-stream run is the same command with `--concurrency 16`. No request count is given, so AIPerf makes
one full pass (a count is in turns, and one that is too small silently drops whole categories). After each
run every category was checked: 80 prompts and its full number of turns. `--osl` is sent as
`max_completion_tokens`; the engine honours it (a counting prompt ran to exactly 8,192 tokens,
`finish_reason: length`). AIPerf ends a conversation when one of its turns is cut at the cap; that
skipped no turn here (the replies that reached the cap were last turns).

The prompt file was prepared with AIPerf's own preparation code (`--format raw`) on another machine and
copied over: preparing it downloads about 40 GB of source datasets, which does not belong on a serving
host. One source, `cais/hle`, is gated: its terms must be accepted on Hugging Face before the 208 prompts
that come from it (72 humanities, 62 math, 74 stem) can be fetched.

**Tokens are the server's.** Every tok/s here uses `usage.completion_tokens` from the reply. AIPerf's own
count, made with the model's tokenizer on the client, is one lower on almost every request (the end token;
the difference ranged from -3 to +5 tokens), which matters only for very short replies.

## One stream

One request at a time; each number is a per-request figure averaged over the category.

| category | prompts | requests (turns) | decode tok/s, mean | median | end-to-end tok/s, mean | first token ms, median | p90 | output tokens, mean | median | replies at the 8192 cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coding | 80 | 89 | 140.5 | 140.3 | 134.1 | 171 | 212 | 875 | 618 | 1 |
| humanities | 80 | 88 | 103.6 | 97.2 | 96.7 | 117 | 188 | 1,332 | 466 | 6 |
| math | 80 | 98 | 134.1 | 133.9 | 127.6 | 132 | 181 | 2,129 | 1,055 | 10 |
| multilingual | 80 | 80 | 122.8 | 121.6 | 113.0 | 138 | 169 | 369 | 202 | 0 |
| qa | 80 | 80 | 101.0 | 97.1 | 93.3 | 81 | 87 | 296 | 157 | 1 |
| rag | 80 | 105 | 121.1 | 119.0 | 90.5 | 233 | 254 | 118 | 65 | 0 |
| reasoning | 80 | 190 | 113.9 | 107.8 | 104.9 | 96 | 183 | 347 | 242 | 1 |
| roleplay | 80 | 158 | 80.0 | 78.8 | 73.3 | 105 | 186 | 172 | 122 | 0 |
| stem | 80 | 86 | 113.0 | 111.7 | 109.2 | 138 | 192 | 1,678 | 857 | 4 |
| summarization | 80 | 80 | 108.1 | 107.8 | 95.0 | 215 | 304 | 201 | 155 | 0 |
| writing | 80 | 84 | 97.3 | 93.9 | 95.2 | 191 | 336 | 2,085 | 1,582 | 2 |
| **all 11, macro-average** (each category counts once) | 880 | 1138 | 112.3 | 109.9 | 103.0 | 147 | 208 | 873 | 502 |  |
| **all 1,138 requests pooled** | 880 | 1138 | 110.9 | 107.5 | 101.4 | 137 | 238 | 789 | 241 | 25 |

- *decode tok/s*: (output tokens - 1) / (time from the first token to the last).
- *end-to-end tok/s*: output tokens / (time from sending the request to the last token).
- *macro-average*: the mean of the 11 category values, the way the SPEED-Bench paper gives an overall
  figure. *Pooled* treats the 1,138 requests as one set, so categories with more turns weigh more.
- Time weighted (all output tokens / all decode seconds): 116.0 tok/s.

## 16 streams

### `v2.0`

AIPerf keeps 16 requests in flight: when a stream ends it sends the next turn. The prompts go in the
dataset's order, so mostly one or two categories share the engine at a time.

| category | prompts | requests (turns) | decode tok/s, mean | median | end-to-end tok/s, mean | first token ms, median | p90 | output tokens, mean | median | replies at the 8192 cap | last token to end of stream, s, median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coding | 80 | 89 | 32.7 | 31.8 | 30.5 | 1,424 | 2,297 | 897 | 606 | 1 | 21.7 |
| humanities | 80 | 88 | 25.5 | 21.2 | 23.3 | 989 | 1,752 | 1,246 | 462 | 5 | 37.9 |
| math | 80 | 98 | 36.0 | 34.0 | 33.3 | 1,185 | 1,941 | 2,053 | 1,146 | 9 | 59.2 |
| multilingual | 80 | 80 | 28.4 | 27.4 | 25.3 | 1,010 | 1,839 | 326 | 193 | 0 | 9.1 |
| qa | 80 | 80 | 22.4 | 21.3 | 19.8 | 748 | 1,209 | 299 | 145 | 1 | 10.9 |
| rag | 80 | 105 | 21.0 | 18.6 | 15.0 | 1,436 | 3,101 | 125 | 64 | 0 | 5.6 |
| reasoning | 80 | 190 | 26.3 | 24.8 | 23.5 | 948 | 1,672 | 325 | 242 | 0 | 10.3 |
| roleplay | 80 | 158 | 20.6 | 18.4 | 18.1 | 865 | 1,719 | 171 | 116 | 0 | 4.3 |
| stem | 80 | 86 | 29.6 | 25.4 | 28.2 | 1,212 | 2,025 | 1,803 | 968 | 6 | 50.8 |
| summarization | 80 | 80 | 23.1 | 22.8 | 20.2 | 1,042 | 1,848 | 191 | 154 | 0 | 4.9 |
| writing | 80 | 84 | 23.2 | 21.7 | 22.4 | 1,527 | 2,771 | 2,061 | 1,590 | 2 | 54.5 |
| **all 11, macro-average** (each category counts once) | 880 | 1138 | 26.2 | 24.3 | 23.6 | 1,126 | 2,016 | 863 | 517 |  | 24.5 |
| **all 1,138 requests pooled** | 880 | 1138 | 25.9 | 24.1 | 23.3 | 1,051 | 2,038 | 778 | 246 | 24 | 10.3 |

Total output: 889,019 tokens in 3,627 s = **245 tok/s** (AIPerf's own
figure with client-side tokens: 244.8).

### What the 16-stream total shows

The last column is the reason the total is 245 and not higher. At 16 streams the engine sent the
closing chunk of a stream (`finish_reason`, usage, `[DONE]`) a median of 9.3 s after
that stream's last token (mean 24 s, 90th percentile 71 s); at one stream it is
24 ms. All the text had arrived; only the end of the stream was late. The cause is in how the
engine groups requests (`engine/server/serve.cpp`, `run()`): requests that decode together form a run, and
a run returns when every request in it has ended or when a new request arrives. A client that waits for
the end of its stream before sending the next request cannot be that new request. With 16 such clients on
16 slots, each group of 16 drains down to its longest reply before anyone is released, so on average
8.3 of the 16 streams were producing tokens.

What this means for the numbers:

- The per-request decode tok/s above are what a user sees while text is arriving. They are not affected.
- The total is a floor for this workload, not the engine's capacity. The README's 16-user figure (about
  413 tok/s) starts 16 requests together with the same fixed length, so all 16 end together and nothing
  waits.
- For a real client the cost is a reply that looks finished but whose stream stays open; an agent that
  waits for the end of the stream is held up by the longest request it shares the engine with. Any new
  request arriving ends the wait, so it shows only when all slots' clients are waiting at once.

This is a finding of this benchmark, not a property anyone asked for. It is not fixed in `v2.0`;
`DSGB10_END_EARLY=1` fixes it (next section).

## 16 streams with `DSGB10_END_EARLY=1`

`DSGB10_END_EARLY=1` (off by default; [CHANGES.md](../../CHANGES.md), v2.8) makes the engine send a request's
closing chunk at the step where the request ends, while the rest of its group goes on. The session is kept
as before; the next request that arrives pauses the group, as an arrival always did, and takes a slot.
No step's arithmetic changes.

How this run differs from the `v2.0` run above, plainly:

- It ran on a test instance of the same four boxes (the engine built from this repository with the
  variable set, the served front-end and arguments), not on the live endpoint, in time-boxed test
  windows. One pass over all 880 prompts does not fit a window, so the file was split into five parts of
  176 prompts (every fifth prompt, in the dataset's order: 16 prompts of each category in every part) and
  each part ran as its own AIPerf pass at 16 streams, from a cold start, with the same command.
- One part reached its window's time limit with 3 turns still in flight (1 multilingual, 2 writing): 1,135
  of the 1,138 turns are in the table.
- The same first part was also run on `v2.0` in the same window, as a like-for-like pair (below).

| category | prompts | requests (turns) | decode tok/s, mean | median | end-to-end tok/s, mean | first token ms, median | p90 | output tokens, mean | median | replies at the 8192 cap | last token to end of stream, ms, mean | p90 | tokens per verify step | acceptance rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coding | 80 | 89 | 25.6 | 25.5 | 24.1 | 1,304 | 2,505 | 1,012 | 664 | 2 | 33 | 130 | 3.68 | 86.3 % |
| humanities | 80 | 88 | 17.4 | 15.8 | 16.3 | 479 | 550 | 1,269 | 451 | 5 | 61 | 141 | 2.82 | 82.5 % |
| math | 80 | 98 | 26.7 | 26.7 | 25.5 | 473 | 532 | 2,217 | 1,216 | 9 | 38 | 127 | 3.31 | 83.3 % |
| multilingual | 79 | 79 | 21.2 | 21.5 | 20.1 | 494 | 549 | 319 | 200 | 0 | 21 | 126 | 3.04 | 79.6 % |
| qa | 80 | 80 | 17.0 | 16.3 | 15.6 | 427 | 447 | 189 | 140 | 0 | 37 | 126 | 2.15 | 74.5 % |
| rag | 80 | 105 | 21.4 | 20.7 | 17.4 | 587 | 803 | 121 | 66 | 0 | 25 | 129 | 3.43 | 84.0 % |
| reasoning | 80 | 190 | 20.6 | 19.2 | 19.1 | 460 | 557 | 360 | 256 | 1 | 37 | 130 | 2.94 | 81.0 % |
| roleplay | 80 | 158 | 14.4 | 13.2 | 13.7 | 444 | 615 | 174 | 114 | 0 | 33 | 115 | 1.75 | 68.4 % |
| stem | 80 | 86 | 20.1 | 19.3 | 19.7 | 479 | 550 | 1,688 | 836 | 5 | 29 | 127 | 3.04 | 82.1 % |
| summarization | 80 | 80 | 17.9 | 17.9 | 16.9 | 583 | 728 | 207 | 160 | 0 | 25 | 128 | 2.76 | 77.2 % |
| writing | 78 | 82 | 17.4 | 15.8 | 17.1 | 527 | 680 | 1,901 | 1,566 | 1 | 32 | 127 | 2.18 | 74.6 % |
| **all 11, macro-average** (each category counts once) | 877 | 1135 | 20.0 | 19.3 | 18.7 | 569 | 774 | 860 | 515 |  |  |  | 2.83 | 79.4 % |
| **all 1,135 requests pooled** | 877 | 1135 | 19.8 | 18.4 | 18.5 | 486 | 677 | 780 | 251 | 23 | 34 | 128 | 2.83 | 81.1 % |

Total output: 885,529 tokens in 2,923 s of the five passes = **303 tok/s**. Each pass has its own
end-of-file tail, where fewer than 16 streams are left; up to the moment the last request of a pass starts,
the rate is 316 to 330 tok/s (323 over the five), with 15.4 of 16 streams producing tokens. The closing
chunk follows the last token by 34 ms on average (median under 1 ms, 90th percentile 128 ms, largest
468 ms).

The like-for-like pair, the first part (176 prompts, 224 turns), same window, each from a cold start:

| | `v2.0` | `DSGB10_END_EARLY=1` |
|---|---:|---:|
| total output | 240 tok/s | 307 tok/s |
| time to finish the 224 requests | 743 s | 520 s |
| streams producing tokens, on average | 7.3 | 15.5 |
| last token to end of stream: median / mean / largest | 11.7 s / 27.1 s / 119 s | under 1 ms / 41 ms / 454 ms |
| first token, median | 1,126 ms | 488 ms |
| decode tok/s per request, mean | 26.0 | 19.7 |
| requests failed | 0 | 0 |

How to read it:

- The end of a stream is no longer late, and the endpoint gives about a quarter more output on this
  workload, because every slot is working.
- A request's own decode rate is lower (about 20 tok/s against 26): its steps are now shared with 15
  others, not with 7. What a client gains is the time from sending a request to the end of its stream
  and the time to finish a set of requests.
- The total is still under the README's 16-user figure (about 413). That figure is 16 code prompts
  started together; here the text is mixed (2.83 tokens per verify step over all steps against about 3.7
  for coding alone), and every new request first has its prompt read while the group waits: with a
  request ending about every two seconds, that is the rest of the difference.
- Replies are not word for word those of the `v2.0` run. When several requests share a step, a reply's
  wording depends on who shares it (the re-routing `DSGB10_OEA`, and the plain wide body of shared steps);
  that is so in `v2.0` too, where the same 16-stream run twice does not give the same texts. Requests now
  arrive at other moments, so other requests share a step. With those two levers off, the 16-stream
  closed-loop texts with and without `DSGB10_END_EARLY` are identical; a request that runs alone is
  bit-identical either way.

## Draft acceptance, from the engine

AIPerf reads acceptance from the reply or from a metrics endpoint; this engine offers neither, so the
figures come from the engine's log. It writes one line per request (`serve: req N drafts: V verify steps,
X drafts verified and Y accepted per step`). During each run those lines were copied with a timestamp
and matched to AIPerf's records by prompt tokens, completion tokens and end time: 1,136 of 1,138 requests
matched in each run (the two others are 2-token replies, which have no verify step). Sums are taken
over a category's steps before dividing, as AIPerf's own report does.

| category | one stream: tokens per verify step | drafts verified per step | drafts accepted per step | acceptance rate | 16 streams: tokens per verify step | drafts verified per step | drafts accepted per step | acceptance rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coding | 3.90 | 3.50 | 2.89 | 82.7 % | 3.57 | 2.98 | 2.57 | 86.1 % |
| humanities | 3.18 | 2.81 | 2.18 | 77.6 % | 2.70 | 2.05 | 1.69 | 82.6 % |
| math | 3.51 | 3.07 | 2.51 | 81.6 % | 3.12 | 2.55 | 2.12 | 83.1 % |
| multilingual | 3.15 | 2.90 | 2.14 | 74.0 % | 2.83 | 2.28 | 1.82 | 79.8 % |
| qa | 3.06 | 2.81 | 2.04 | 72.8 % | 2.54 | 1.94 | 1.53 | 79.0 % |
| rag | 3.69 | 3.54 | 2.66 | 74.9 % | 3.24 | 2.65 | 2.21 | 83.5 % |
| reasoning | 3.21 | 2.94 | 2.20 | 74.9 % | 2.78 | 2.25 | 1.77 | 78.8 % |
| roleplay | 1.96 | 1.67 | 0.95 | 56.9 % | 1.66 | 0.97 | 0.64 | 66.1 % |
| stem | 3.18 | 2.80 | 2.18 | 77.9 % | 2.95 | 2.38 | 1.95 | 81.8 % |
| summarization | 3.01 | 2.83 | 1.99 | 70.5 % | 2.62 | 2.11 | 1.61 | 76.2 % |
| writing | 2.40 | 2.08 | 1.40 | 67.5 % | 2.10 | 1.49 | 1.10 | 74.0 % |
| **macro-average** | 3.11 | 2.81 | 2.10 | 73.8 % | 2.74 | 2.15 | 1.73 | 79.2 % |
| **all verify steps pooled** | 3.05 | 2.69 | 2.04 | 75.9 % | 2.69 | 2.10 | 1.69 | 80.4 % |

- *tokens per verify step* (acceptance length): output tokens / verify steps, about 1 + accepted drafts.
- *acceptance rate*: drafts accepted / drafts verified. The engine's gate chooses per step how many drafts
  to verify (0 to 7), so a low-confidence text has both fewer drafts verified and fewer accepted.

## Reading it

**Speed follows how predictable the text is.** At one stream a step costs roughly the same whatever the
text, so tok/s tracks tokens per step: coding accepts 2.89 drafts a step
(3.90 tokens) and decodes at 140 tok/s; roleplay accepts
0.95 (1.96 tokens) and decodes at 80.

- *Fast*: coding (140), math (134), multilingual (123), rag (121). Code,
  formulas, translations and answers that quote their context repeat structure and source text, which the
  draft head predicts well.
- *Middle*: reasoning (114), stem (113), summarization (108), humanities (104),
  qa (101). Explanatory prose.
- *Slow*: writing (97) and roleplay (80). Free creative text has the least predictable next
  word; the gate verifies fewer drafts (1.67 a step in roleplay against
  3.50 in coding) and fewer of them are right (57 % against 83 %).

The spread between the fastest and the slowest category is 1.76x. A single tok/s figure for this
engine is therefore a statement about the prompts as much as about the engine.

**First token.** A first turn reads its whole prompt: median 137 ms over all requests at one stream
(rag and summarization, the long prompts, about 233 ms). A follow-up turn reuses the
conversation the engine already holds and starts in about 85 ms (median). At 16 streams a new request
first pauses the group that is decoding, and shares the engine with it: median 1,051 ms.

**16 streams.** Each stream gets fewer drafts verified per step (2.10 against
2.69: the step's rows are shared out between the sessions) and the ones kept are
the surer ones (acceptance 80.4 % against 75.9 %). The order of the categories stays the same.

## Against the README's numbers

The README's headline for one user is about 90 tok/s for chat (mean over 320 held-out prompts) and about
147 for code (a 3-prompt bench, 512 tokens). Here:

| | README | this benchmark |
|---|---:|---|
| code, one user | about 147 | coding 140.5 (89 requests, replies up to 8,192 tokens) |
| chat, one user | about 90 | roleplay 80, writing 97, qa 101, humanities 104, summarization 108; all 11 categories 112 |
| 16 users | about 413 | 245 total (`v2.0`); 303 with `DSGB10_END_EARLY=1` |

- Code agrees within the spread of the prompts (140 against 147: other prompts, longer replies).
- Chat: the README's figure sits inside the range of the conversational categories here. The overall
  112 is higher than 90 because 4 of the 11 categories (coding, math, multilingual, rag) are
  the predictable kind, not because anything changed. Neither number is "the" speed; they are different
  mixes of text.
- 16 users: not the same measurement. The README starts 16 different code prompts together, 512 tokens
  each, and counts tokens while all 16 decode. Here 16 clients work through mixed prose, each reply as
  long as the model makes it, a new prompt read whenever one ends, and in `v2.0` with the late end of
  stream described above. The per-stream rates are comparable in kind (README: about 26 per user on code; here
  coding 33), the totals are not.

## Limits

- One run per cell. A second one-stream pass over three categories gives the run-to-run spread:

| category | requests compared | first pass, decode tok/s mean | repeat pass | difference | replies with the same length |
|---|---:|---:|---:|---:|---:|
| coding | 71 | 140.7 | 140.7 | -0.0 % | 71 |
| summarization | 80 | 108.1 | 108.9 | +0.7 % | 80 |
| roleplay | 146 | 79.6 | 79.6 | -0.0 % | 144 |

  Requests that shared the engine with another user in either pass are left out of this comparison.
  The repeat pass gave replies of the same length on all but two requests, so this is the spread of
  the timing alone: under 1 %. It says nothing about other prompts or another day.
- The endpoint was live and shared. Requests from other users were logged with their times. At one stream
  39 of our requests overlapped one of them (they ran about a quarter slower); those conversations were
  run again while the endpoint was quiet and the tables use the second measurement. At 16 streams the
  last three categories of the first pass (multilingual, summarization, roleplay) overlapped two long
  requests of another user and were run again as a group of three, twice, until every request in the table had run without another user's request beside it. The
  16-stream total is from the first full pass, including those minutes.
- The client ran on rank 0's own CPU (under 1.3 GB of memory); it is not a separate machine.
- The engine's session cache was not cleared between runs. First turns were cold in the full passes (the
  16-stream pass found a cached prefix on 22 of 880 first turns); the re-run requests named above had
  their prompts cached, so their first-token times are warm.
- Output cap 8192: 25 replies at one stream and 24 at 16 streams reached it and were
  cut there (math, humanities and stem prompts from `cais/hle` are the ones that run long).
- Greedy decoding only. The tutorial notes that temperature 0 is the best case for draft acceptance;
  sampled requests accept less.
- AIPerf's `speed-bench-report` cannot make its acceptance matrix from this engine (no per-request
  acceptance field in the reply, no Prometheus endpoint); the acceptance table is computed by
  `tools/bench/speed_bench_table.py` from the engine's log with the same formulas.

## Reproduce

1. AIPerf 0.14.0 or later (`pip install "git+https://github.com/ai-dynamo/aiperf@ebcdc98"` in a
   virtualenv is what was used).
2. Accept the terms of [cais/hle](https://huggingface.co/datasets/cais/hle) and log in (`hf auth login`).
   Then, on a machine with about 40 GB of free disk and spare memory, not the serving host:
   `python -m aiperf.dataset.loader.vendor.speed_bench_prepare --config qualitative --dataset-revision v1.1 --format raw --output-dir ./speed-bench-data`.
   Check the file: 880 lines, 80 per `category`.
3. Copy `qualitative.jsonl` to the host and run, with nothing else on the endpoint:

```bash
export AIPERF=/path/to/venv/bin/aiperf URL=http://127.0.0.1:PORT KEY_FILE=/path/to/keyfile
export LOG_DIR=/path/to/service/logs      # the directory of rank.0.log and front.log
tools/bench/speed_bench.sh c1 1 qualitative.jsonl --tokenizer /path/to/model_dir
tools/bench/speed_bench.sh c16 16 qualitative.jsonl --tokenizer /path/to/model_dir
python3 tools/bench/speed_bench_table.py out_c1 runs/c1
python3 tools/bench/speed_bench_table.py out_c16 runs/c16
```

   The script runs AIPerf inside a 3 GB memory-capped systemd scope and copies the engine's per-request
   log lines with timestamps while it runs. The table script writes the per-request CSV and the tables
   above; several run directories can be given (a file with only some categories, or a re-run of
   disturbed requests, is merged by conversation turn), and a file of other users' request times
   (`SPANS=file`) marks the requests that shared the engine.
4. One stream takes about two and a quarter hours (most of it the long replies), 16 streams about one hour.
