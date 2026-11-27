"""CPU tests of the encoder's span-row cache (tools/vision/vision_encoder.py).

    python3 -m unittest discover -s tools/vision -p test_encoder_cache.py -v

The same stand-ins as test_encoder_limits.py: no model, no GPU, no network. The rows are bf16 x 5120
there, so a 40-row image is 400 KiB and the budget arithmetic below is the real one.
"""
import base64
import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_encoder_limits import load_encoder  # noqa: E402  (the shared stand-in harness)


def image(rows, variant=0):
    """An 'image' whose bytes are its row count (variant pads it: same rows, other bytes)."""
    return {"url": "data:image/png;base64," +
            base64.b64encode(((" " * variant) + str(rows)).encode()).decode()}


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enc = load_encoder(max_rows=40000, cache_mb=1)
        enc = cls.enc

        def encode(patches, types, grid):  # 10 KiB a row, as bf16 x 5120
            enc.calls.append(patches)
            return torch.zeros((types.numel(), 5120), dtype=torch.bfloat16), grid

        enc.encode = encode

    def setUp(self):
        self.enc.calls.clear()
        self.enc.measured.clear()
        self.enc._cache.clear()
        self.enc._cache_bytes = 0

    def test_a_repeated_image_is_served_from_the_cache(self):
        req = {"images": [image(40)]}
        out1, data1 = self.enc.handle(req)
        out2, data2 = self.enc.handle(req)
        self.assertFalse(out1[0].get("cached"), "the first turn encodes")
        self.assertTrue(out2[0].get("cached"), "the second is a hit")
        self.assertEqual(out2[0]["ms"], 0.0)
        self.assertEqual(out2[0]["hash"], out1[0]["hash"])
        self.assertEqual(data1, data2, "the rows handed to the engine are identical byte for byte")
        self.assertEqual(len(self.enc.calls), 1, "the ViT ran once")
        self.assertEqual(len(self.enc.measured), 1, "and the image was read once")

    def test_another_image_is_encoded(self):
        self.enc.handle({"images": [image(40)]})
        self.enc.handle({"images": [image(40, 1)]})  # the same rows, other bytes
        self.assertEqual(len(self.enc.calls), 2)
        self.assertEqual(len(self.enc._cache), 2)

    def test_an_image_over_the_budget_is_not_kept(self):
        # 200 rows are 2.0 MiB: over --cache-mb 1, so caching it would break the limit
        out1, _ = self.enc.handle({"images": [image(200)]})
        self.assertFalse(out1[0].get("cached"))
        self.assertEqual(self.enc._cache_bytes, 0, "nothing is retained past the budget")
        self.assertEqual(len(self.enc._cache), 0)
        out2, _ = self.enc.handle({"images": [image(200)]})
        self.assertFalse(out2[0].get("cached"), "and the next turn encodes again")
        self.assertEqual(len(self.enc.calls), 2)

    def test_the_oldest_goes_at_the_budget(self):
        # 40 rows are 400 KiB: two fit in 1 MiB, the third evicts the first
        for v in (0, 1, 2):
            self.enc.handle({"images": [image(40, v)]})
        self.assertEqual(len(self.enc._cache), 2, "the budget holds two")
        self.assertEqual(self.enc.calls.__len__(), 3)
        self.enc.handle({"images": [image(40, 0)]})  # the first was evicted: encoded again
        self.assertEqual(len(self.enc.calls), 4)
        self.enc.handle({"images": [image(40, 2)]})  # the newest is still there: no ViT
        self.assertEqual(len(self.enc.calls), 4)
        self.assertLessEqual(self.enc._cache_bytes, 1 << 20, "the budget always holds")

    def test_the_cache_is_off_at_budget_zero(self):
        enc = load_encoder(max_rows=40000, cache_mb=0)

        def encode(patches, types, grid):
            enc.calls.append(patches)
            return torch.zeros((types.numel(), 5120), dtype=torch.bfloat16), grid

        enc.encode = encode
        enc.handle({"images": [image(40)]})
        out2, _ = enc.handle({"images": [image(40)]})
        self.assertFalse(out2[0].get("cached"))
        self.assertEqual(len(enc.calls), 2, "no cache, no hit")


if __name__ == "__main__":
    unittest.main()
