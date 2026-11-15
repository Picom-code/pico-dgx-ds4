"""CPU tests of what a client is told when a video (or image) input is refused; needs torch (CPU), no
model, no GPU, no network. The decoder tests also need PyAV and are skipped without it.

    python3 -m unittest discover -s tools/vision -p test_video_errors.py -v
"""
import base64
import io
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
import wave

from test_encoder_limits import load_encoder

try:
    import av  # noqa: F401
    from test_video_frames import clip
except ImportError:
    av = None

HERE = os.path.dirname(os.path.abspath(__file__))
# what must never reach a client: the decoder's, the OS's or Python's own error text
RAW = ("Errno", "<none>", "Traceback", "av.error", "urlopen", "Invalid data", "object has no attribute")
# the decoder's own messages
PLAIN = ("not a readable video", "the clip has no video stream", "video duration is unavailable",
         "could not decode a requested video frame")


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


class PlainFailureTests(unittest.TestCase):
    """Whatever fails while a clip is fetched or decoded, the client gets plain words."""

    @classmethod
    def setUpClass(cls):
        cls.enc = load_encoder()

    def video_error(self, record, **patches):
        saved = {k: getattr(self.enc, k) for k in patches}
        for k, v in patches.items():
            setattr(self.enc, k, v)
        try:
            with self.assertRaises(self.enc.MediaError) as e:
                self.enc.decode_videos({"videos": [record], "frame_budget": 32})
        finally:
            for k, v in saved.items():
                setattr(self.enc, k, v)
        msg = str(e.exception)
        for raw in RAW:
            self.assertNotIn(raw, msg)
        return msg

    def decoder(self, returncode, stdout, stderr=b""):
        """The encoder with a decoder subprocess that ends this way."""
        fake = types.SimpleNamespace(run=lambda *a, **k: types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr),
                                     TimeoutExpired=subprocess.TimeoutExpired)
        return {"subprocess": fake}

    def test_a_decoder_that_crashes_or_says_nothing_useful(self):
        clip_url = {"url": data_url(b"anything")}
        raw = b"av.error.InvalidDataError: [Errno 1094995529] Invalid data found when processing input: '<none>'"
        for rc, out in [(-11, b""), (1, b""), (1, b"Traceback (most recent call last):"), (0, b"[]"), (1, b'{"ok": false}'),
                        (0, b'{"ok": true}'), (2, b'{"ok": false, "error": "[Errno 2] x"}'), (1, b'{"ok": false, "error": 5}')]:
            self.assertEqual(self.video_error(clip_url, **self.decoder(rc, out, raw)), "not a readable video", (rc, out))

    def test_the_decoders_own_words_pass(self):
        out = json.dumps({"ok": False, "error": "the clip has no video stream"}).encode()
        self.assertEqual(self.video_error({"url": data_url(b"x")}, **self.decoder(1, out)), "the clip has no video stream")

    def test_a_decoder_that_hangs(self):
        def run(*a, **k):
            raise subprocess.TimeoutExpired("video_frames.py", 60)

        fake = types.SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired)
        self.assertEqual(self.video_error({"url": data_url(b"x")}, subprocess=fake), "video decoding timed out")

    def test_fetch_failures(self):
        def fails_with(e):
            def fetch(record, max_bytes=None, kind="image"):
                raise e
            return fetch

        url = {"url": "https://example.com/clip.mp4"}
        e = urllib.error.URLError(OSError(-2, "Name or service not known"))
        self.assertEqual(self.video_error(url, fetch=fails_with(e)), "could not fetch the video from its URL")
        self.assertEqual(self.video_error(url, fetch=fails_with(TimeoutError("timed out"))), "could not fetch the video from its URL")
        e = urllib.error.HTTPError(url["url"], 404, "Not Found", None, None)
        self.assertEqual(self.video_error(url, fetch=fails_with(e)), "could not fetch the video: the server answered HTTP 404")
        e = base64.binascii.Error("Incorrect padding")
        self.assertEqual(self.video_error({"url": "data:video/mp4;base64,A"}, fetch=fails_with(e)), "the video data is not valid base64")

    def test_over_the_wire(self):
        srv = self.enc.S(("127.0.0.1", 0), self.enc.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            def ask(req):
                with socket.create_connection(srv.server_address, timeout=10) as s:
                    body = json.dumps(req).encode()
                    s.sendall(struct.pack("<I", len(body)) + body)
                    (n,) = struct.unpack("<I", self.enc.recv_exact(s, 4))
                    return json.loads(self.enc.recv_exact(s, n))

            for req, want in [
                ({"videos": ["not a record"]}, "the video could not be processed"),
                ({"videos": [{"url": "/etc/passwd"}]}, "video url must be a data: URL or http(s)"),
                ({"videos": [{"url": "http://169.254.169.254/latest/meta-data"}]}, "video URL host 169.254.169.254 is not a public address"),
                ({"videos": [{"url": data_url(b"x"), "max_frames": 101}]}, "max_frames must be an integer between 1 and 100"),
                ({"videos": [], "frame_budget": 8}, "not enough frame budget for the videos"),
            ]:
                hdr = ask(req)
                self.assertEqual((hdr["ok"], hdr["error"]), (False, want), req)
        finally:
            srv.shutdown()
            srv.server_close()


@unittest.skipUnless(av, "PyAV is not installed")
class DecoderTests(unittest.TestCase):
    """The real decoder (video_frames.py) on inputs that are not clips."""

    @classmethod
    def setUpClass(cls):
        cls.enc = load_encoder()
        cls.clip = clip()

    def outcome(self, raw):
        """("frames", n) or ("error", message) for a clip sent as base64 through the encoder's video path."""
        try:
            out = self.enc.decode_videos({"videos": [{"url": data_url(raw), "max_frames": 3}], "frame_budget": 32})
            return "frames", len(out[0]["frames"])
        except self.enc.MediaError as e:
            for raw_text in RAW:
                self.assertNotIn(raw_text, str(e))
            return "error", str(e)

    def test_a_real_clip_still_decodes(self):
        self.assertEqual(self.outcome(self.clip), ("frames", 3))

    def test_bytes_that_are_not_a_video(self):
        for raw in (b"not a video", b"", os.urandom(4096)):
            self.assertEqual(self.outcome(raw), ("error", "not a readable video"), raw[:16])
        # a container header with nothing in it may open and then have no stream: plain words either way
        for raw in (b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64, self.clip[:40]):
            kind, what = self.outcome(raw)
            self.assertEqual(kind, "error", raw[:16])
            self.assertIn(what, PLAIN, raw[:16])

    def test_truncated_clips_fail_in_plain_words_or_decode(self):
        for cut in (0.25, 0.5, 0.75, 0.95):
            kind, what = self.outcome(self.clip[: int(len(self.clip) * cut)])
            if kind == "error":
                self.assertIn(what, PLAIN, cut)

    def test_audio_only(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"\x00" * 16000)
        self.assertEqual(self.outcome(buf.getvalue()), ("error", "the clip has no video stream"))

    def test_playlists_and_scripts_do_not_reach_files_or_the_network(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as local:
            local.write(self.clip)
            local.flush()
            for raw in (
                ("#EXTM3U\n#EXT-X-TARGETDURATION:3\n#EXTINF:3,\nfile://%s\n#EXT-X-ENDLIST\n" % local.name).encode(),
                ("#EXTM3U\n#EXTINF:3,\n%s\n#EXT-X-ENDLIST\n" % local.name).encode(),
                b"#EXTM3U\n#EXTINF:3,\nhttp://127.0.0.1:18900/v1/models\n#EXT-X-ENDLIST\n",
                b"#EXTM3U\n#EXTINF:3,\n/etc/passwd\n#EXT-X-ENDLIST\n",
                ("ffconcat version 1.0\nfile '%s'\n" % local.name).encode(),
                b"ffconcat version 1.0\nfile '/etc/passwd'\n",
                ("[playlist]\nFile1=%s\nNumberOfEntries=1\n" % local.name).encode(),
            ):
                kind, what = self.outcome(raw)
                self.assertEqual(kind, "error", "a playlist must not yield frames of a local file: %r" % raw[:40])
                self.assertIn(what, PLAIN)

    def test_the_decoder_alone_answers_in_plain_words_and_logs_the_cause(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"not a video")
            f.flush()
            r = subprocess.run([sys.executable, os.path.join(HERE, "video_frames.py"), f.name], capture_output=True, timeout=60)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(json.loads(r.stdout), {"ok": False, "error": "not a readable video"})
        self.assertIn(b"video decoder:", r.stderr, "the decoder's own error goes to stderr, for the encoder's log")
        r = subprocess.run([sys.executable, os.path.join(HERE, "video_frames.py"), "/nonexistent/clip"], capture_output=True, timeout=60)
        self.assertEqual(json.loads(r.stdout), {"ok": False, "error": "not a readable video"})

    def test_sample_raises_only_video_errors(self):
        from video_frames import VideoError, sample

        for raw, count in ((b"not a video", 3), (self.clip[:40], 3), (self.clip, 0), (self.clip, 101)):
            with self.assertRaises(VideoError):
                sample(raw, count)


if __name__ == "__main__":
    unittest.main()
