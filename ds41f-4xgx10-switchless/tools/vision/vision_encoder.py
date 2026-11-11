"""dsgx10 vision encoder: DeepSeek-V4.1 1017's ViT + aligner as a small side process.

It turns images into the LLM's embedding rows for an image span, using the checkpoint's own
reference code (inference/vision.py, inference/image_processor.py, mounted read-only), so the
preprocessing, the 2D RoPE, the 3x3 aligner and the span layout are the publisher's:

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every row of the span is returned (bf16, 5120 wide): the delimiters are the learned image_start /
image_newline / image_end vectors, the IMAGE slots the aligner rows in reading order. The engine puts
them in place of the token embeddings at the span's positions (prefill only).

Protocol (TCP, 127.0.0.1 only): request  = u32 len, JSON {"images": [record, ...]}
                                reply    = u32 len, JSON {"ok", "error", "images": [{rows, hash, ...}]},
                                           u64 len, the rows of every image, bf16, in order.
A record is what encoding.py's image blocks give: {"url": data-URL | http(s) URL} or {"data": b64}.
"""

import argparse
import hashlib
import json
import os
import socket
import socketserver
import struct
import sys
import threading
import time
from types import SimpleNamespace

import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = 40_000_000  # PIL refuses images over twice this (decompression bombs)

p = argparse.ArgumentParser()
p.add_argument("--model", default="/model")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=18950)
p.add_argument("--max-images", type=int, default=100)
p.add_argument("--max-bytes", type=int, default=32 << 20)  # per fetched image
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


def _public(host):
    """Only public addresses are fetched (no loopback, private, link-local or reserved: no SSRF into the
    cluster); VISION_ALLOW_PRIVATE=1 lifts it for tests."""
    import ipaddress

    if os.environ.get("VISION_ALLOW_PRIVATE", "0") == "1":
        return
    for fam, _, _, _, sa in socket.getaddrinfo(host, None):
        ip = ipaddress.ip_address(sa[0])
        if not ip.is_global or ip.is_multicast:
            raise ValueError("image URL host %s is not a public address" % host)


def fetch(record):
    """Image bytes from a record (image_processor.load_image_bytes, with a size cap for URLs)."""
    url = record.get("url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        from urllib.parse import urlsplit
        from urllib.request import HTTPRedirectHandler, Request, build_opener

        class Redir(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, hdrs, newurl):
                _public(urlsplit(newurl).hostname or "")
                return super().redirect_request(req, fp, code, msg, hdrs, newurl)

        _public(urlsplit(url).hostname or "")
        with build_opener(Redir).open(Request(url, headers={"User-Agent": "dsgx10-vision/1"}), timeout=20) as r:
            b = r.read(a.max_bytes + 1)
        if len(b) > a.max_bytes:
            raise ValueError("image larger than %d bytes" % a.max_bytes)
        return b
    if isinstance(url, str) and url and not url.startswith("data:"):
        raise ValueError("image url must be a data: URL or http(s)")  # no local paths over the API
    src = record.get("source")
    if isinstance(src, dict) and src.get("url") and not src.get("data"):
        return fetch({"url": src["url"]})
    return ip.load_image_bytes(record)


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


def handle(req):
    recs = req.get("images") or []
    if not recs or len(recs) > a.max_images:
        raise ValueError("between 1 and %d images per request" % a.max_images)
    out, blobs = [], []
    with _lock:
        for r in recs:
            t0 = time.time()
            raw = fetch(r)
            span, g = encode(raw)
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
                imgs, data = handle(json.loads(recv_exact(s, n)))
                hdr = {"ok": True, "images": imgs}
            except Exception as e:  # noqa: BLE001  (the error goes back to the client as a 400)
                log("error:", repr(e))
                hdr, data = {"ok": False, "error": str(e)[:500]}, b""
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
