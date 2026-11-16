"""CPU tests for video sampling; requires PyAV and Pillow, no model or GPU."""
import base64
from fractions import Fraction
import io
import unittest
import wave

import av
from PIL import Image
from video_frames import sample


def clip():
    buf = io.BytesIO()
    with av.open(buf, 'w', format='mp4') as output:
        stream = output.add_stream('mpeg4', rate=1)
        stream.width = stream.height = 64
        stream.pix_fmt = 'yuv420p'
        for i, color in enumerate(['red', 'green', 'blue']):
            frame = av.VideoFrame.from_image(Image.new('RGB', (64, 64), color))
            frame.pts, frame.time_base = i, Fraction(1, 1)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return buf.getvalue()


class SamplingTests(unittest.TestCase):
    def test_timestamps_and_scene_order(self):
        result = sample(clip(), 3)
        self.assertEqual(result['duration'], 3.0)
        self.assertEqual([f['timestamp'] for f in result['frames']], [0.0, 1.0, 2.0])
        for frame, channel in [(result['frames'][0], 0), (result['frames'][-1], 2)]:
            image = Image.open(io.BytesIO(base64.b64decode(frame['url'].split(',', 1)[1])))
            self.assertEqual(image.size, (64, 64))
            pixel = image.convert('RGB').getpixel((32, 32))
            self.assertGreater(pixel[channel], 200)
            self.assertTrue(all(v < 30 for i, v in enumerate(pixel) if i != channel))

    def test_frame_count_caps_and_single_frame(self):
        raw = clip()
        self.assertEqual(len(sample(raw, 100)['frames']), 3)
        one = sample(raw, 1)['frames']
        self.assertEqual(len(one), 1)
        self.assertEqual(one[0]['timestamp'], 0.0)
        for count in [0, 101]:
            with self.assertRaises(ValueError):
                sample(raw, count)

    def test_broken_clip(self):
        with self.assertRaises(Exception):
            sample(b'not a video', 3)

    def test_audio_without_video_is_rejected(self):
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(b'\x00' * 16000)
        with self.assertRaisesRegex(ValueError, 'no video stream'):
            sample(buf.getvalue(), 3)


if __name__ == '__main__':
    unittest.main()
