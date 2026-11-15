"""dsgx10 vision encoder: DeepSeek-V4.1 1017's ViT + aligner as a small side process.

It turns images into the LLM's embedding rows for an image span, using the checkpoint's own
reference code (inference/vision.py, inference/image_processor.py, mounted read-only), so the
preprocessing, the 2D RoPE, the 3x3 aligner and the span layout are the publisher's:

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every row of the span is returned (bf16, 5120 wide): the delimiters are the learned image_start /
image_newline / image_end vectors, the IMAGE slots the aligner rows in reading order. The engine puts
them in place of the token embeddings at the span's positions (prefill only).

Protocol (TCP, 127.0.0.1 only): request  = u32 len, JSON {"images": [record, ...], "max_rows": N}
                                reply    = u32 len, JSON {"ok", "error", "images": [{rows, hash, ...}]},
                                           u64 len, the rows of every image, bf16, in order.
A record is what encoding.py's image blocks give: {"url": data-URL | http(s) URL} or {"data": b64}.
"max_rows" (optional) lowers --max-rows for one request: the cap on the rows of all its images together.
Over it the reply is {"ok": false, "code": "too_many_rows", "error"} and encoding stops at that image.
"""

import argparse
import hashlib
import json
import os
import socket
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = 40_000_000  # PIL refuses images over twice this (decompression bombs)

p = argparse.ArgumentParser()
p.add_argument("--model", default="/model")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=18950)
p.add_argument("--max-images", type=int, default=100)
# The rows (image tokens) of all the images of one request together. Memory follows rows, not the number
# of images: each row is 10 KiB here, again in the front-end and again in the engine. dsfront sends its
# own cap (--max-image-rows, the same default) with each request; the lower of the two applies.
p.add_argument("--max-rows", type=int, default=40000)
p.add_argument("--max-bytes", type=int, default=32 << 20)  # per fetched image
p.add_argument("--max-video-bytes", type=int, default=128 << 20)
p.add_argument("--video-timeout-s", type=float, default=60)
p.add_argument("--idle-unload-s", type=float, default=0)   # 0: keep the weights resident
a = p.parse_args()

sys.path.insert(0, os.path.join(a.model, "inference"))
import image_processor as ip  # noqa: E402  (the checkpoint's own code)
import vision as _vision  # noqa: E402
from vision import Aligner, ViT  # noqa: E402

# The reference calls scaled_dot_product_attention on 3D strided views ([heads, n, d] after a
# transpose), which only the MATH backend takes (42 ms per layer at 3.8k patches on GB10). The same
# function on a contiguous 4D view runs the flash kernel (0.65 ms); outputs agree to bf16 rounding.
import torch.nn.functional as _F  # noqa: E402


class _FShim:
    def __getattr__(self, n):
        return getattr(_F, n)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, *a, **kw):
        if q.dim() == 3:
            return _F.scaled_dot_product_attention(q.unsqueeze(0).contiguous(), k.unsqueeze(0).contiguous(),
                                                   v.unsqueeze(0).contiguous(), *a, **kw)[0]
        return _F.scaled_dot_product_attention(q, k, v, *a, **kw)


if os.environ.get("VISION_SDPA_REF", "0") != "1":
    _vision.F = _FShim()

cfg = json.load(open(os.path.join(a.model, "inference", "config.json")))
args = SimpleNamespace(**cfg)
assert args.vision_n_layers > 0

_model = None
_lock = threading.Lock()
_video_lock = threading.Lock()
_last = time.time()


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def load():
    """ViT, aligner and the three delimiter vectors, bf16 on the GPU (the reference's dtypes)."""
    global _model
    if _model is not None:
        return _model
    from safetensors import safe_open

    t0 = time.time()
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    want = [k for k in idx if k.startswith(("vision.", "aligner.", "image_"))]
    sd = {}
    for f in sorted({idx[k] for k in want}):
        with safe_open(os.path.join(a.model, f), framework="pt", device="cpu") as h:
            for k in want:
                if idx[k] == f:
                    sd[k] = h.get_tensor(k)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")  # as generate.py: the RoPE tables (get_vision_cos_sin) are made on it
    vit, aligner = ViT(args), Aligner(args)
    vit.load_state_dict({k[len("vision."):]: v for k, v in sd.items() if k.startswith("vision.")}, strict=True)
    aligner.load_state_dict({k[len("aligner."):]: v for k, v in sd.items() if k.startswith("aligner.")}, strict=True)
    vit.eval(), aligner.eval()
    delim = {k: sd[k].to("cuda", torch.bfloat16) for k in ("image_start", "image_newline", "image_end")}
    _model = (vit, aligner, delim)
    log(f"loaded {len(sd)} tensors in {time.time() - t0:.1f} s; cuda allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    return _model


class MediaError(ValueError):
    """A request's image or video cannot be taken; the message is ours and goes to the client as it is."""


def _public(host, kind="image"):
    """Only public addresses are fetched (no loopback, private, link-local or reserved: no SSRF into the
    cluster); VISION_ALLOW_PRIVATE=1 lifts it for tests. `kind` names the input in the message."""
    import ipaddress

    if os.environ.get("VISION_ALLOW_PRIVATE", "0") == "1":
        return
    for fam, _, _, _, sa in socket.getaddrinfo(host, None):
        ip = ipaddress.ip_address(sa[0])
        if not ip.is_global or ip.is_multicast:
            raise MediaError("%s URL host %s is not a public address" % (kind, host))


def fetch(record, max_bytes=None, kind="image"):
    """Image (or, for kind "video", clip) bytes from a record (image_processor.load_image_bytes, with a
    size cap for URLs). The messages name the input by `kind`."""
    limit = a.max_bytes if max_bytes is None else max_bytes
    url = record.get("url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        from urllib.parse import urlsplit
        from urllib.request import HTTPRedirectHandler, Request, build_opener

        class Redir(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, hdrs, newurl):
                _public(urlsplit(newurl).hostname or "", kind)
                return super().redirect_request(req, fp, code, msg, hdrs, newurl)

        _public(urlsplit(url).hostname or "", kind)
        with build_opener(Redir).open(Request(url, headers={"User-Agent": "dsgx10-vision/1"}), timeout=20) as r:
            b = r.read(limit + 1)
        if len(b) > limit:
            raise MediaError("%s larger than %d bytes" % (kind, limit))
        return b
    if isinstance(url, str) and url and not url.startswith("data:"):
        raise MediaError("%s url must be a data: URL or http(s)" % kind)  # no local paths over the API
    src = record.get("source")
    if isinstance(src, dict) and src.get("url") and not src.get("data"):
        return fetch({"url": src["url"]}, limit, kind)
    raw = ip.load_image_bytes(record)
    if len(raw) > limit:
        raise MediaError("%s larger than %d bytes" % (kind, limit))
    return raw


def fetch_video(record):
    """A clip's bytes. Our own refusals (MediaError) go to the client as they are; anything a library
    raises while fetching or unpacking (urllib's "<urlopen error [Errno -2] ...>", base64's "Incorrect
    padding", a socket timeout) is logged and replaced by plain words."""
    from urllib.error import HTTPError

    try:
        return fetch(record, a.max_video_bytes, "video")
    except MediaError:
        raise
    except HTTPError as e:
        raise MediaError("could not fetch the video: the server answered HTTP %d" % e.code) from e
    except Exception as e:  # noqa: BLE001
        log("video fetch failed:", repr(e))
        url = record.get("url") or (record.get("source") or {}).get("url") if isinstance(record, dict) else None
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            raise MediaError("could not fetch the video from its URL") from e
        raise MediaError("the video data is not valid base64") from e


def decode_videos(req):
    videos = req.get("videos")
    budget = req.get("frame_budget", a.max_images)
    if not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= a.max_images:
        raise MediaError("invalid video frame budget")
    if not isinstance(videos, list) or not 1 <= len(videos) <= budget:
        raise MediaError("not enough frame budget for the videos")
    results = []
    with _video_lock:
        for record in videos:
            count = record.get("max_frames", 32)
            if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= a.max_images:
                raise MediaError("max_frames must be an integer between 1 and %d" % a.max_images)
            count = min(count, budget // len(videos))
            raw = fetch_video(record)
            with tempfile.TemporaryDirectory(prefix="vision-video-") as tmp:
                clip = Path(tmp) / "clip"
                clip.write_bytes(raw)
                try:
                    result = subprocess.run(
                        [sys.executable, str(Path(__file__).with_name("video_frames.py")), str(clip), "--max-frames", str(count)],
                        capture_output=True, timeout=a.video_timeout_s, check=False,
                    )
                except subprocess.TimeoutExpired as e:
                    raise MediaError("video decoding timed out") from e
            # The decoder answers in plain words (video_frames.VideoError). Whatever else comes out of the
            # subprocess (a crash, no JSON, a library's error text on stderr) is logged here and the
            # client is told the clip is not a readable video.
            if result.stderr.strip():
                log("video decoder stderr:", result.stderr[-400:].decode(errors="replace").strip())
            try:
                decoded = json.loads(result.stdout)
            except (ValueError, UnicodeError):
                decoded = None
            if not isinstance(decoded, dict) or result.returncode or not decoded.get("ok"):
                said = decoded.get("error") if isinstance(decoded, dict) else None
                if not isinstance(said, str) or not said or result.returncode not in (0, 1):
                    log("video decoder failed: exit %s, %d B of output" % (result.returncode, len(result.stdout)))
                    said = "not a readable video"
                raise MediaError(said)
            if not isinstance(decoded.get("duration"), (int, float)) or not isinstance(decoded.get("frames"), list):
                raise MediaError("not a readable video")
            results.append({"duration": decoded["duration"], "frames": decoded["frames"]})
            log(f"video {len(raw)} B, {decoded['duration']:.3f} s -> {len(decoded['frames'])} timestamped frames")
    return results


@torch.inference_mode()
def encode(raw: bytes):
    """One image's span rows [n, 5120] bf16 and its grid (the reference's load_image + encode_image +
    merge_image_embeddings, for one span)."""
    vit, aligner, delim = load()
    with torch.device("cuda"):  # set_default_device is per thread; the RoPE tables are made here
        return _encode(raw, vit, aligner, delim)


def _encode(raw, vit, aligner, delim):
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image({"data": raw}, args)
    types = ip.image_token_types(n_llm_h, n_llm_w)
    feats = aligner(vit(patches.to("cuda"), n_vit_h, n_vit_w), n_vit_h, n_vit_w)  # [n_llm_h * n_llm_w, 5120]
    assert feats.shape[0] == n_llm_h * n_llm_w, (feats.shape, n_llm_h, n_llm_w)
    t = types.to("cuda")
    span = torch.empty((t.numel(), feats.shape[1]), dtype=torch.bfloat16, device="cuda")
    span[t == ip.IMAGE_START] = delim["image_start"]
    span[t == ip.IMAGE_END] = delim["image_end"]
    span[t == ip.IMAGE_NEW_LINE] = delim["image_newline"]
    span[t == ip.IMAGE] = feats.to(torch.bfloat16)
    return span, (n_vit_h, n_vit_w, n_llm_h, n_llm_w)


class TooManyRows(ValueError):
    """The images of one request take more rows than the cap (the reply's code: too_many_rows)."""


def row_limit(req):
    """The cap on a request's rows in total: --max-rows, or the request's own max_rows when lower."""
    limit = a.max_rows
    asked = req.get("max_rows")
    if isinstance(asked, int) and not isinstance(asked, bool) and asked > 0:
        limit = min(limit, asked)
    return limit


def handle(req):
    recs = req.get("images") or []
    if not recs or len(recs) > a.max_images:
        raise ValueError("between 1 and %d images per request" % a.max_images)
    limit, total = row_limit(req), 0
    out, blobs = [], []
    with _lock:
        for k, r in enumerate(recs, 1):
            t0 = time.time()
            raw = fetch(r)
            span, g = encode(raw)
            total += span.shape[0]
            if total > limit:  # stop here: nothing more is encoded or kept for this request
                raise TooManyRows("at most %d image tokens per request; the first %d of %d images already take %d"
                                  % (limit, k, len(recs), total))
            torch.cuda.synchronize()
            ms = 1e3 * (time.time() - t0)
            blobs.append(span.view(torch.int16).cpu().numpy().tobytes())
            h = hashlib.sha256(raw).digest()
            out.append({"rows": span.shape[0], "hash": int.from_bytes(h[:8], "little"), "grid": g, "ms": round(ms, 1),
                        "bytes": len(raw)})
            log(f"image {len(raw)} B -> {g[2]}x{g[3]} tokens ({span.shape[0]} rows) in {ms:.0f} ms")
    return out, b"".join(blobs)


def recv_exact(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(min(1 << 20, n - len(b)))
        if not c:
            raise ConnectionError("closed")
        b += c
    return bytes(b)


class H(socketserver.BaseRequestHandler):
    def handle(self):
        global _last
        s = self.request
        while True:
            try:
                (n,) = struct.unpack("<I", recv_exact(s, 4))
            except ConnectionError:
                return
            if n > (256 << 20):
                return
            try:
                req = json.loads(recv_exact(s, n))
                if "videos" in req:
                    try:
                        hdr, data = {"ok": True, "videos": decode_videos(req)}, b""
                    except MediaError:
                        raise
                    except Exception as e:  # noqa: BLE001  (not one of our refusals: its text stays in the log)
                        log("video error:", repr(e))
                        raise MediaError("the video could not be processed") from e
                else:
                    imgs, data = handle(req)
                    hdr = {"ok": True, "images": imgs}
            except Exception as e:  # noqa: BLE001  (the error goes back to the client as a 400)
                log("error:", repr(e))
                hdr, data = {"ok": False, "error": str(e)[:500]}, b""
                if isinstance(e, TooManyRows):
                    hdr["code"] = "too_many_rows"
            _last = time.time()
            j = json.dumps(hdr).encode()
            s.sendall(struct.pack("<I", len(j)) + j + struct.pack("<Q", len(data)) + data)


class S(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def unloader():
    global _model
    while a.idle_unload_s > 0:
        time.sleep(30)
        with _lock:
            if _model is not None and time.time() - _last > a.idle_unload_s:
                _model = None
                torch.cuda.empty_cache()
                log("idle: weights unloaded")


if __name__ == "__main__":
    if os.environ.get("VISION_PRELOAD", "0") == "1":
        load()
    threading.Thread(target=unloader, daemon=True).start()
    srv = S((a.host, a.port), H)
    log(f"listening on {a.host}:{a.port}")
    srv.serve_forever()
