"""Decode a bounded set of timestamped video frames. Run in a subprocess with a timeout."""
import argparse
import base64
import io
import json
import math
import sys

import av
from PIL import Image


class VideoError(ValueError):
    """The clip cannot be used; the message is ours, in plain words, and safe to show to a client."""


def sample(raw, max_frames):
    """Up to max_frames timestamped frames of the clip. Whatever goes wrong, the error is a VideoError:
    one of our own messages, or "not a readable video" for anything the decoder or the image library
    raises (their text, such as "[Errno 1094995529] Invalid data found when processing input: '<none>'",
    stays out of the answer; the cause is chained for the log)."""
    try:
        return _sample(raw, max_frames)
    except VideoError:
        raise
    except Exception as e:  # noqa: BLE001  (PyAV, Pillow, MemoryError, ...)
        raise VideoError("not a readable video") from e


def _sample(raw, max_frames):
    if not 1 <= max_frames <= 100:
        raise VideoError("max_frames must be an integer between 1 and 100")
    # BytesIO supplies the clip. Nested network/file protocols (e.g. playlists) stay disabled.
    with av.open(io.BytesIO(raw), options={"protocol_whitelist": ""}) as container:
        if not container.streams.video:
            raise VideoError("the clip has no video stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        start = float((stream.start_time or 0) * stream.time_base)
        duration = float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / av.time_base
        if not math.isfinite(duration) or duration <= 0:
            raise VideoError("video duration is unavailable")
        fps = float(stream.average_rate or 0)
        available = stream.frames or (max(1, math.ceil(duration * fps)) if fps > 0 else max_frames)
        count = min(max_frames, available)
        end = max(0.0, duration - (1 / fps if fps > 0 else 0.001))
        frames = []
        total_bytes = 0
        for i in range(count):
            target = start + (end * i / (count - 1) if count > 1 else 0)
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            selected = None
            for frame in container.decode(stream):
                selected = frame
                if frame.pts is not None and float(frame.pts * stream.time_base) >= target - 1e-6:
                    break
            if selected is None:
                raise VideoError("could not decode a requested video frame")
            image = selected.to_image().convert("RGB")
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            data = base64.b64encode(buf.getvalue()).decode()
            total_bytes += len(data)
            if total_bytes > 64 << 20:
                raise VideoError("decoded video frames exceed 64 MiB")
            timestamp = max(0.0, float(selected.pts * stream.time_base) - start) if selected.pts is not None else target - start
            frames.append({"timestamp": round(timestamp, 3), "url": "data:image/jpeg;base64," + data})
        return {"duration": round(duration, 3), "frames": frames}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--max-frames", type=int, default=32)
    args = parser.parse_args()
    try:
        with open(args.input, "rb") as f:
            result = sample(f.read(), args.max_frames)
        print(json.dumps({"ok": True, **result}))
    except VideoError as e:
        if e.__cause__ is not None:  # the decoder's own words: for the encoder's log, not for the client
            print("video decoder: %r" % (e.__cause__,), file=sys.stderr)
        print(json.dumps({"ok": False, "error": str(e)}))
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001  (reading the file)
        print("video decoder: %r" % (e,), file=sys.stderr)
        print(json.dumps({"ok": False, "error": "not a readable video"}))
        raise SystemExit(1)
