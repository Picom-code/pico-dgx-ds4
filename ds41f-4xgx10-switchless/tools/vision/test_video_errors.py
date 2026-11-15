"""CPU tests of what a client is told when a video (or image) input is refused; needs torch (CPU), no
model, no GPU, no network. The decoder tests also need PyAV and are skipped without it.

    python3 -m unittest discover -s tools/vision -p test_video_errors.py -v
"""
import base64
import unittest

from test_encoder_limits import load_encoder


def data_url(raw, mime="video/mp4"):
    return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode())


class WordingTests(unittest.TestCase):
    """A refused video is called a video; a refused image is still called an image."""

    @classmethod
    def setUpClass(cls):
        cls.enc = load_encoder()

    def video_error(self, record):
        with self.assertRaises(ValueError) as e:
            self.enc.decode_videos({"videos": [record], "frame_budget": 32})
        return str(e.exception)

    def test_local_paths_are_refused_as_video(self):
        for url in ("/etc/passwd", "file:///etc/passwd", "clip.mp4", "ftp://example.com/a.mp4", "concat:/etc/passwd"):
            self.assertEqual(self.video_error({"url": url}), "video url must be a data: URL or http(s)", url)

    def test_private_addresses_are_refused_as_video(self):
        for host in ("127.0.0.1", "10.220.0.2", "192.168.1.10", "169.254.169.254", "[::1]", "0.0.0.0"):
            e = self.video_error({"url": "http://%s/clip.mp4" % host})
            self.assertEqual(e, "video URL host %s is not a public address" % host.strip("[]"), host)
        e = self.video_error({"source": {"url": "http://127.0.0.1:18900/v1/models"}})
        self.assertEqual(e, "video URL host 127.0.0.1 is not a public address")

    def test_a_clip_over_the_size_limit_is_refused_as_video(self):
        limit, self.enc.a.max_video_bytes = self.enc.a.max_video_bytes, 1000
        try:
            self.assertEqual(self.video_error({"url": data_url(b"x" * 1001)}), "video larger than 1000 bytes")
        finally:
            self.enc.a.max_video_bytes = limit

    def test_images_are_still_called_images(self):
        for record, want in [
            ({"url": "/etc/passwd"}, "image url must be a data: URL or http(s)"),
            ({"url": "http://127.0.0.1/a.png"}, "image URL host 127.0.0.1 is not a public address"),
            ({"url": data_url(b"x" * (self.enc.a.max_bytes + 1), "image/png")}, "image larger than %d bytes" % self.enc.a.max_bytes),
        ]:
            with self.assertRaises(ValueError) as e:
                self.enc.fetch(record)
            self.assertEqual(str(e.exception), want)

    def test_no_message_for_a_video_says_image(self):
        for record in ({"url": "/etc/passwd"}, {"url": "http://10.0.0.1/a.mp4"}, {"url": "http://localhost/a.mp4"}):
            self.assertNotIn("image", self.video_error(record).lower(), record)


if __name__ == "__main__":
    unittest.main()
