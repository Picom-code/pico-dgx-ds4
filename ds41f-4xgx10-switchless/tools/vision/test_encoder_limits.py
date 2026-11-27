"""CPU tests of the encoder's request limits; needs torch (CPU is enough), no model, no GPU, no network.

    python3 -m unittest discover -s tools/vision -p test_encoder_limits.py -v

vision_encoder.py is imported with stand-ins for the checkpoint's own modules (image_processor, vision)
and for the ViT, so only the request handling runs: how many rows a request may take in total, where
encoding stops, and what the reply says.
"""
import base64
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import types
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_encoder(max_rows=100, cache_mb=0):
    """vision_encoder imported against a stand-in checkpoint directory, with --max-rows MAX_ROWS and
    --cache-mb CACHE_MB. The limit tests pass 0 (default): they count ViT calls and rows read, which a
    cache would short-circuit, and the module keeps its cache across the tests of one class."""
    model = tempfile.mkdtemp(prefix="vision-test-model-")
    os.makedirs(os.path.join(model, "inference"))
    with open(os.path.join(model, "inference", "config.json"), "w") as f:
        json.dump({"vision_n_layers": 1}, f)
    ip = types.ModuleType("image_processor")

    def load_image_bytes(record):
        data = record.get("data") or record.get("url", "").split(",", 1)[-1]
        return base64.b64decode(data)

    ip.load_image_bytes = load_image_bytes
    vision = types.ModuleType("vision")
    vision.Aligner = vision.ViT = object
    vision.F = None
    saved = {k: sys.modules.get(k) for k in ("image_processor", "vision", "vision_encoder")}
    sys.modules.update({"image_processor": ip, "vision": vision})
    sys.modules.pop("vision_encoder", None)
    argv, sys.argv = sys.argv, ["vision_encoder.py", "--model", model, "--max-rows", str(max_rows),
                                "--cache-mb", str(cache_mb)]
    sys.path.insert(0, HERE)
    try:
        import vision_encoder as enc
    finally:
        sys.argv = argv
        sys.path.remove(HERE)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    enc.calls = []

    def measure(raw):  # the "image" is its row count in ASCII: b"30" is an image of 30 rows
        enc.measured.append(raw)
        return raw, torch.zeros(int(raw), dtype=torch.int64), (1, 1, 1, 1)

    def encode(patches, types, grid):  # the ViT's stand-in: counts what is really encoded
        enc.calls.append(patches)
        return torch.zeros((types.numel(), 8), dtype=torch.bfloat16), grid

    enc.measured = []
    enc.measure = measure
    enc.encode = encode
    enc.torch.cuda.synchronize = lambda: None
    return enc


def image(rows):
    return {"url": "data:image/png;base64," + base64.b64encode(str(rows).encode()).decode()}


class RowLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enc = load_encoder(max_rows=100)

    def setUp(self):
        self.enc.calls.clear()
        self.enc.measured.clear()

    def test_default_cap_matches_the_front_end(self):
        self.assertEqual(self.enc.p.parse_args([]).max_rows, 40000)  # dsfront --max-image-rows

    def test_rows_up_to_the_cap_pass(self):
        out, data = self.enc.handle({"images": [image(30), image(30), image(40)]})
        self.assertEqual([i["rows"] for i in out], [30, 30, 40])
        self.assertEqual(len(data), 100 * 8 * 2)

    def test_rows_over_the_cap_stop_at_the_image_that_crosses_it(self):
        with self.assertRaises(self.enc.TooManyRows) as e:
            self.enc.handle({"images": [image(30)] * 3 + [image(11)] + [image(30)] * 6})
        self.assertEqual(str(e.exception), "at most 100 image tokens per request; the first 4 of 10 images already take 101")
        self.assertEqual(len(self.enc.measured), 4, "the six images after it are not even read")
        self.assertEqual(self.enc.calls, [b"30"] * 3, "the image that crosses the cap is refused before the ViT runs on it")

    def test_41_full_size_images_cost_40_encodes(self):
        enc = load_encoder(max_rows=40000)
        with self.assertRaises(enc.TooManyRows) as e:
            enc.handle({"images": [image(994)] * 41})
        self.assertEqual(str(e.exception), "at most 40000 image tokens per request; the first 41 of 41 images already take 40754")
        self.assertEqual(len(enc.calls), 40)
        enc.calls.clear()
        out, _ = enc.handle({"images": [image(994)] * 40})
        self.assertEqual((len(out), len(enc.calls)), (40, 40))

    def test_a_request_can_lower_the_cap_but_not_raise_it(self):
        self.assertEqual(self.enc.row_limit({"max_rows": 50}), 50)
        with self.assertRaisesRegex(self.enc.TooManyRows, "at most 50 image tokens"):
            self.enc.handle({"images": [image(30), image(30)], "max_rows": 50})
        self.assertEqual(self.enc.row_limit({"max_rows": 100000}), 100)
        for ignored in (None, True, "50", 0, -1, 2.5, [50]):
            self.assertEqual(self.enc.row_limit({"max_rows": ignored}), 100, ignored)

    def test_the_reply_says_too_many_rows(self):
        srv = self.enc.S(("127.0.0.1", 0), self.enc.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            def ask(req):
                with socket.create_connection(srv.server_address, timeout=10) as s:
                    body = json.dumps(req).encode()
                    s.sendall(struct.pack("<I", len(body)) + body)
                    (n,) = struct.unpack("<I", self.enc.recv_exact(s, 4))
                    hdr = json.loads(self.enc.recv_exact(s, n))
                    (m,) = struct.unpack("<Q", self.enc.recv_exact(s, 8))
                    return hdr, self.enc.recv_exact(s, m)

            hdr, data = ask({"images": [image(60), image(60)], "max_rows": 40000})
            self.assertEqual((hdr["ok"], hdr["code"], data), (False, "too_many_rows", b""))
            self.assertTrue(hdr["error"].startswith("at most 100 image tokens per request"))
            hdr, data = ask({"images": [image(60), image(40)], "max_rows": 40000})
            self.assertTrue(hdr["ok"])
            self.assertEqual(len(data), 100 * 8 * 2)
            hdr, data = ask({"images": []})
            self.assertEqual((hdr["ok"], hdr.get("code")), (False, None), "other errors carry no code")
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
