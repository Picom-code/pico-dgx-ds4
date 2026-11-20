# SPEED-Bench (qualitative split) with NVIDIA AIPerf

Measured 2026-11-20 on the build of tag `v2.0`, as it serves. One run per cell.

SPEED-Bench is NVIDIA's benchmark for speculative decoding
([dataset](https://huggingface.co/datasets/nvidia/SPEED-Bench)). Its *qualitative* split is 880 prompts in
11 kinds of text, 80 each; some are conversations of several turns, 1,138 turns in all. It answers a
question our own bench does not: how fast is the engine on each kind of text, when the reply may be as
long as the model wants. Speculative drafts make that depend on the text: the more predictable the next
words, the more draft tokens a step accepts.

## The short version

| | one stream | 16 streams |
|---|---:|---:|
| decode tok/s per request, mean over the 11 categories | 112.3 | 26.2 |
| the same, median request | 109.9 | 24.3 |
| fastest and slowest category (mean) | 140.5 coding, 80.0 roleplay | 36.0 math, 20.6 roleplay |
| first token, median | 137 ms | 1,051 ms |
| tokens per verify step, all steps | 3.05 | 2.69 |
| draft acceptance rate, all steps | 75.9 % | 80.4 % |
| total output of the endpoint | - | 245 tok/s |
| requests sent / failed | 1,138 / 0 | 1,138 / 0 |

Decode tok/s is measured per request from its first token to its last. The 16-stream total is lower than
this engine can give 16 users at once: see [what the 16-stream total shows](#what-the-16-stream-total-shows).

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

This is a finding of this benchmark, not a property anyone asked for. It is not fixed in `v2.0`.

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
| 16 users | about 413 | 245 total |

- Code agrees within the spread of the prompts (140 against 147: other prompts, longer replies).
- Chat: the README's figure sits inside the range of the conversational categories here. The overall
  112 is higher than 90 because 4 of the 11 categories (coding, math, multilingual, rag) are
  the predictable kind, not because anything changed. Neither number is "the" speed; they are different
  mixes of text.
- 16 users: not the same measurement. The README starts 16 different code prompts together, 512 tokens
  each, and counts tokens while all 16 decode. Here 16 clients work through mixed prose, each reply as
  long as the model makes it, a new prompt read whenever one ends, and with the late end of stream
  described above. The per-stream rates are comparable in kind (README: about 26 per user on code; here
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
