#!/usr/bin/env python3
"""Exact /metrics checks against a scripted rank-zero peer (stdlib only).

    python3 frontend/tools/metrics_test.py frontend/target/release/dsfront tokenizer.json

The test starts its own frontend on loopback. Frame layouts are in docs/protocol.md.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import http.client
import json
from pathlib import Path
from select import select
import socket
import struct
import subprocess
import sys
import tempfile
import time


def send_frame(peer, kind, payload):
    peer.sendall(struct.pack("<IHH", len(payload), kind, 0) + payload)


def read_frame(reader):
    size, kind, _ = struct.unpack("<IHH", reader.read(8))
    payload = reader.read(size)
    assert len(payload) == size, "truncated rank-zero frame"
    return kind, payload


def submit_frame(reader):
    while True:
        kind, payload = read_frame(reader)
        if kind == 0x02:  # SUBMIT
            return struct.unpack_from("<Q", payload)[0], struct.unpack_from("<I", payload, 64)[0]
        assert kind == 0x04, f"expected SUBMIT, received {kind:#x}"  # RELEASE


def exercise(port, peer, reader, token):
    def get_samples():
        with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=2)) as client:
            client.request("GET", "/metrics")
            reply = client.getresponse()
            assert reply.status == 200
            return {name: float(value) for line in reply.read().decode().splitlines()
                    if line.startswith("tensorfold:")
                    for name, value in [line.removeprefix("tensorfold:").split()]}

    def wait_for_samples(predicate):
        deadline = time.monotonic() + 5
        while True:
            try:
                samples = get_samples()
                if predicate(samples):
                    return samples
            except ConnectionRefusedError:
                pass
            assert time.monotonic() < deadline, "expected metrics were not observed"
            time.sleep(0.01)

    def post(client, content, stream=False):
        client.request("POST", "/v1/chat/completions", json.dumps({"stream": stream, "messages": [
            {"role": "user", "content": content}]}), {"Content-Type": "application/json"})

    def chat(content):
        with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=6)) as client:
            post(client, content)
            reply = client.getresponse()
            return reply.status, json.load(reply)

    def tokens(request, ids):
        send_frame(peer, 0x82, struct.pack(f"<QII{len(ids)}I", request, len(ids), 0, *ids))

    def done(request, reason, prompt, cached, generated, prefill, error=b""):
        send_frame(peer, 0x83, struct.pack("<QIIIIQQQII", request, reason, prompt, cached,
                                          generated, 0, prefill, 0, len(error), 0) + error)

    def completed(samples):
        return tuple(samples[name] for name in (
            "prompt_tokens_total", "prompt_tokens_cached_total", "request_prefill_seconds_sum",
            "request_prefill_seconds_count", "time_to_first_token_seconds_count"))

    generation = "generation_tokens_total"
    wait_for_samples(lambda samples: samples["requests_running"] == 0)
    with ThreadPoolExecutor(1) as executor:
        # Offsets and values come from the wire contract, independent of HTTP usage.
        good = executor.submit(chat, "known metrics")
        request, prompt = submit_frame(reader)
        assert prompt > 2
        time.sleep(0.1)  # Make frontend TTFT strictly larger than the scripted 70 ms prefill.
        tokens(request, [token])
        done(request, 0, prompt, 2, 1, 70_000)
        status, body = good.result()
        assert status == 200 and body["usage"]["prompt_tokens"] == prompt, body
        samples = get_samples()
        assert completed(samples) == (prompt, 2, 0.07, 1, 1), f"wire prompt, cache and prefill: {samples}"
        assert samples["time_to_first_token_seconds_sum"] > 0.07, samples

        for name, ids, reason, expected_status in (
            ("no tokens", None, 2, 200), ("empty frame", [], 0, 200), ("engine error", [token], 4, 500),
        ):
            reply = executor.submit(chat, name)
            request, length = submit_frame(reader)
            if ids is not None:
                tokens(request, ids)
            done(request, reason, length, 0, len(ids or ()), 30_000, b"scripted error" if reason == 4 else b"")
            assert reply.result()[0] == expected_status, name
            current = get_samples()
            assert completed(current) == completed(samples), f"{name} must not enter the completed cohort: {current}"
            assert current["time_to_first_token_seconds_sum"] == samples["time_to_first_token_seconds_sum"], name
            assert current[generation] == samples[generation] + len(ids or ()), name
            samples = current

        clamped = executor.submit(chat, "cache clamp")
        request, length = submit_frame(reader)
        tokens(request, [token])
        done(request, 1, length, length + 5, 1, 0)
        assert clamped.result()[0] == 200
        samples = get_samples()
        assert completed(samples) == (prompt + length, 2 + length, 0.07, 2, 2), f"cached prefix clamp: {samples}"

    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=3)) as client:
        post(client, "abandoned stream", stream=True)
        request, length = submit_frame(reader)
        tokens(request, [token])
        with client.getresponse() as reply:
            assert reply.status == 200
            for line in reply:
                if line.startswith(b"data:") and b'"content":"x"' in line:
                    break
            else:
                raise AssertionError("stream ended before the first content token")

    # Give Gen an event with which to observe the closed HTTP stream. CANCEL identifies the request.
    sent = 1
    deadline = time.monotonic() + 5
    while not select([peer], [], [], 0.05)[0]:
        assert time.monotonic() < deadline, "frontend did not cancel the abandoned stream"
        tokens(request, [token])
        sent += 1
    assert read_frame(reader) == (0x03, struct.pack("<Q", request)), "wrong cancellation"
    tokens(request, [token] * 4)
    done(request, 2, length, 0, sent + 4, 0)
    # A later completion proves the reader has processed both abandoned-request frames.
    with ThreadPoolExecutor(1) as executor:
        marker_reply = executor.submit(chat, "after disconnect")
        marker, length = submit_frame(reader)
        done(marker, 0, length, 0, 0, 0)
        assert marker_reply.result()[0] == 200
    after = get_samples()
    assert after[generation] == samples[generation] + sent + 4, f"all late token IDs must count: {after}"
    assert completed(after) == completed(samples), f"disconnect must not enter the completed cohort: {after}"
    assert after["time_to_first_token_seconds_sum"] == samples["time_to_first_token_seconds_sum"], after
    print("Exact rank-zero prompt, cache, prefill, completion exclusions and post-disconnect tokens passed.")


def main(binary, tokenizer):
    token = json.loads(Path(tokenizer).read_text())["model"]["vocab"]["x"]
    with tempfile.TemporaryDirectory(prefix="pico-metrics-", dir="/tmp") as directory, socket.socket(socket.AF_UNIX) as listener:
        endpoint = str(Path(directory) / "engine.sock")
        listener.settimeout(6)
        listener.bind(endpoint)
        listener.listen(1)
        with socket.socket() as spare:
            spare.bind(("127.0.0.1", 0))
            port = spare.getsockname()[1]
        server = subprocess.Popen([binary, "serve", "--engine-socket", endpoint, "--tokenizer", tokenizer,
                                   "--listen", f"127.0.0.1:{port}"])
        try:
            peer, _ = listener.accept()
            with peer, peer.makefile("rb") as reader:
                peer.settimeout(6)
                assert read_frame(reader) == (0x01, struct.pack("<II", 1, 0))  # HELLO
                name = b"metrics-test"
                send_frame(peer, 0x81, struct.pack("<IIIIII", 1, 4096, 150000, 1, 64, len(name)) + name)
                exercise(port, peer, reader, token)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    if len(sys.argv) != 3 or not __debug__:
        sys.exit("usage: python3 metrics_test.py DSFRONT TOKENIZER_JSON")
    main(*sys.argv[1:])
