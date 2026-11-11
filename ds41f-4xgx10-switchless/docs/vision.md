# Vision input (image_url / image parts)

DeepSeek-V4.1-Flash-1017 ships a vision tower (`vision.*`, `aligner.*`, `image_start/newline/end`, 259 tensors,
~0.4B parameters) and per-layer image routing biases (`ffn.gate.bias_vl`). This build serves it with no change to
the text path: text requests run the same code, the same kernels and the same memory as before.

## Design

```
client --image_url/base64--> dsfront --records--> vision encoder (tools/vision, torch, 127.0.0.1:18950, ~1 GB GPU)
                                  <--span rows (bf16 [rows][5120])--
dsfront: placeholder <｜deepseek_image｜> -> rows x 129264, SUBMIT flag 8 + spans + rows --> rank 0 --kCmdImages--> ranks 1-3
engine prefill chunk: embed, then inject the span rows (4 hc copies); span rows route with bias_vl; Engram dead on spans
decode: untouched (no flag, no branch, same graphs)
```

- **Encoder** (`tools/vision/vision_encoder.py`) imports the checkpoint's own `inference/vision.py` and
  `inference/image_processor.py` (mounted read-only), so preprocessing (min 544x544 pixels, gray padding,
  (x/255-0.5)/0.5, patch 14, 2D RoPE, 3x3 aligner, <= 1024 tokens per image) and the span layout
  `[IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h + [IMAGE_END]` are the publisher's. It returns every row of
  the span, delimiters included (the learned vectors). One shim: the reference calls
  `scaled_dot_product_attention` on 3D strided views, which only PyTorch's MATH backend accepts (42 ms per layer at
  3.8k patches on GB10); the shim passes contiguous 4D views and gets the flash kernel (0.65 ms), same values to
  bf16 rounding. http(s) URLs are fetched with a size cap and only from public addresses (no SSRF into the LAN).
- **dsfront**: with `--vision-addr HOST:PORT`, `image_url` and `image` parts are accepted (without it they are
  refused as before). The chat encoding already wrote one placeholder per image (encoding.py's rules). An image
  request gets a session of its own that is not kept and never enters the text session table, so text prefix reuse
  is unaffected (and image turns are not prefix-cached; v1).
- **Protocol**: `SubmitFlags::kFlagImages = 8`; after `prompt[]`: `u32 n`, `ImageSpan {u32 start, u32 rows,
  u64 hash}[n]`, then the rows. A text SUBMIT is byte-identical to before.
- **Engine**: rank 0 checks that every span covers image ids only, broadcasts the spans and rows (`kCmdImages`,
  ~25-45 ms for 200-450 rows), and marks span positions -1 in the Engram history. The prefill chunk embeds as
  before, then overwrites the span rows; the MoE router reruns `router_topk` with `bias_vl` on span rows only
  (model.py `Gate`); Engram's value contribution is zeroed on span rows and their n-grams are dead
  (model.py `token_mask`), via the existing `live` rule in `hash_positions` (a negative id is dead). Span rows always
  take the chunk path (the small-rows step path has no injection). After the prompt, the spans are cleared.
  Every new line of engine code is behind `!img_span.empty()` or the flag.

The served checkpoint's vision tensors and `bias_vl` are byte-identical to stock 1017.

## Measured (2026-11-07, 4x GB10, same scripts as the 05f572f gates)

Text, test build vs the live 05f572f gate runs:

| bench | 05f572f | + vision | |
|---|---|---|---|
| chat c1 (9-run median) | 79.7 | 81.9 | noise |
| code c1 (9-run median) | 138.4 | 144.4 | noise |
| abbench chat / code c1 | 78.1 / 133.8 | 77.8 / 139.7 | noise |
| code c4 / c8 / c16 (copies) | 222.1 / 330.4 / 438.4 | 238.0 / 333.7 / 438.0 | |
| 16 distinct users | 367 | 380.4 | |
| agent load (8 x 80-300K) | 259.9 | 265.9 | |
| prefill 32K | 10.0k tok/s | 10.04k tok/s | |
| needle 948K | 3/3, 9.56k tok/s | 3/3, 9.70k tok/s | |
| qeval c4 / c16 | 71 / 71 of 75 | 71 / 72 of 75 | |
| session store, slots | 12 GiB, 16 (4 at 1M) | 12 GiB, 16 (4 at 1M) | |

Images (15 checks, temperature 0, through the OpenAI API): 10 known-answer images (OCR sign, receipt, red circle
count, squares/triangles count, bar chart, table, code screenshot, flag colours, arithmetic, position), two images in
one request, an image after 28K tokens of text with a fact in the middle, an http URL, a broken image (400), and a
text request afterwards: 15/15 on the test port and 15/15 through the public gateway.

| | per image |
|---|---|
| encoder (warm) | 100-160 ms for 200-450 tokens; ~0.4-0.55 s for the first image after a start |
| TTFT added vs the same question without the image | 0.17-0.34 s (median ~0.26 s) test port; 0.18-0.47 s public |
| rows per image here | 214-460 tokens (max 1024) |

Concurrency, 8 distinct text users decoding 512 tokens, one request fired 4 s in (2 runs each):

| | text per-user tok/s | during that request's prefill | after it |
|---|---|---|---|
| no extra request | 43.1 / 43.0 | - | - |
| + image request (455 tokens) | 40.9 / 40.9 | 20.2 / 16.2 | 43.05 / 43.05 |
| + text request of the same length (control) | 41.1 / 41.1 | 22.0 / 20.8 | 42.1 / 42.1 |

An image costs the text users what a text prompt of the same token count costs, only while it prefills.

## Memory

Encoder on rank 0's box: 0.95 GiB of GPU memory (bf16 weights + activations), ~0.4 GiB host RSS. Rank 0's free
device memory at ready 11.15 -> 10.31 GiB; the session store reservation (12 GiB on every rank, the smallest offer
across ranks) and the slot layout are unchanged. Workers are untouched.

## Running it

Each chat-completions request accepts up to **100 images across all messages**, including prior
turns. Image URLs and base64 data URLs are supported. The JSON body limit is 256 MiB; image tokens
and text must together fit the context window. Images are encoded sequentially. The encoder and
engine frame limits include room for all 100 images' embedding rows. Large image batches take
longer than single-image requests; the encoder timeout scales with image count to allow URL fetches.

```
docker run -d --name dsgx10-vision --restart unless-stopped --gpus all --net=host --memory=6g -e VISION_PRELOAD=1 \
  -v $MODEL:/model:ro -v $PWD/tools/vision:/app:ro --entrypoint python3 <image with torch + PIL + safetensors> \
  /app/vision_encoder.py --host 127.0.0.1 --port 18950 --max-images 100
dsfront serve ... --vision-addr 127.0.0.1:18950
```

If the encoder is down, image requests answer 503 and text requests are unaffected. Tests: `tools/vision/imgtest.py`
(correctness + TTFT), `tools/vision/conc_img.py` (8 users + one image / text control).

For a running vision-enabled endpoint, the 100-image smoke test generates numbered OCR cards
locally (Pillow and a TrueType font required), checks the first and last image of the full batch,
verifies that 101 images and broken image bytes return 400, and checks text afterwards:

```bash
python3 tools/vision/smoke_100.py http://127.0.0.1:PORT/v1/chat/completions --key-file ~/api-key --output vision-100.json
```

Omit `--key-file` for an unauthenticated test server; `--font PATH` selects another font.

## Video input

Chat-completions requests also accept `video_url` content parts. Public HTTP(S) URLs and base64
data URLs work, for example:

```json
{"messages": [{"role": "user", "content": [
  {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,...", "max_frames": 32}},
  {"type": "text", "text": "Describe what happens in this clip, in order."}
]}]}
```

The encoder needs PyAV (`av`) as well as the image dependencies. It samples up to 32 frames per clip
by default, uniformly from the start through the end; `max_frames` accepts integers from 1 to 100.
For several clips, the available frame budget is divided between them. Video frames and still
images together share the 100-image limit across all messages. Each frame is resized to fit within
768x768 pixels and is inserted as an image with its timestamp and the clip duration, preserving
its place in the conversation. Frame extraction runs in a separate process with a 60-second timeout.
Clips are limited to 128 MiB; the HTTP JSON limit remains 256 MiB. Nested media protocols are disabled,
and video URLs use the same public-address-only fetch rules as images.

This is visual video analysis through the existing vision tower. Audio tracks are ignored. Streaming
and non-streaming chat requests both work; video decoding happens before generation starts.

CPU-only decoder tests (PyAV and Pillow required):

```bash
python3 -m unittest discover -s tools/vision -p test_video_frames.py -v
```

For a running endpoint, test a local clip with streaming and non-streaming requests, bad inputs,
the shared frame budget, then ordinary image and text requests:

```bash
python3 tools/vision/smoke_video.py http://127.0.0.1:PORT/v1/chat/completions clip.mp4 --key-file ~/api-key --output video-results.json
```

With a clip showing known codes in successive scenes, add `--expected 1111 2222 3333` to assert
their order. The live smoke script needs Pillow; generating or decoding clips on the client is not required.

## Not done (v1)

- No prefix reuse for image turns (each image request prefills its whole prompt).
- Images are encoded one at a time on one GPU; no batching across requests.
- `input_image` (Responses API) and audio parts are refused.

