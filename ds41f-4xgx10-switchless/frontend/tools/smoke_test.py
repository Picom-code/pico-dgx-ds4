#!/usr/bin/env python3
"""End-to-end checks of `dsfront serve` against `dsfront mock-engine` (stdlib only).

    dsfront mock-engine --socket S &
    dsfront serve --engine-socket S --api-key-file K --admin-socket A &
    smoke_test.py http://127.0.0.1:18890 K [LOG] [--admin-socket A]

Covers the OpenAI surface pi uses: models, auth, streaming and not, reasoning_content, tool calls
and a tool-result turn (with prefix reuse reported in cached_tokens), usage, stop strings,
max_tokens, the loop guard, bad requests, a client that disconnects, and concurrent requests.
Use the mock engine's default prefill speed and token delay. LOG is dsfront's stderr for its
"plan:" lines. The optional admin socket must belong to this isolated mock fixture.
Run metrics_test.py with the dsfront binary and tokenizer.json for exact rank-zero metrics checks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import socket
import sys
import threading
import time
import urllib.parse
from uuid import uuid4

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("key_file")
parser.add_argument("log", nargs="?")
parser.add_argument("--admin-socket")
args = parser.parse_args()
if not __debug__:
    parser.error("run the smoke test without Python optimization")
BASE = urllib.parse.urlparse(args.url)
KEY = open(args.key_file).read().strip()
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def conn():
    return http.client.HTTPConnection(BASE.hostname, BASE.port, timeout=60)


def metrics():
    c = conn()
    try:
        c.request("GET", "/metrics")
        response = c.getresponse()
        assert response.status == 200, response.status
        assert response.getheader("Content-Type") == "text/plain; version=0.0.4"
        lines = response.read().decode().splitlines()
        types = set(line for line in lines if line.startswith("# TYPE "))
        assert types == {
            "# TYPE tensorfold:requests_running gauge",
            "# TYPE tensorfold:requests_waiting gauge",
            "# TYPE tensorfold:generation_tokens_total counter",
            "# TYPE tensorfold:prompt_tokens_total counter",
            "# TYPE tensorfold:prompt_tokens_cached_total counter",
            "# TYPE tensorfold:time_to_first_token_seconds summary",
            "# TYPE tensorfold:request_prefill_seconds summary",
        }, f"unexpected Prometheus types: {types}"
        return {name.removeprefix("tensorfold:"): float(value)
                for line in lines if line.startswith("tensorfold:")
                for name, value in [line.split()]}
    finally:
        c.close()


def check_metrics(before, usage, case, *, all_tokens_visible=False):
    after = metrics()
    delta = {name: value - before[name] for name, value in after.items()}
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    check(delta["prompt_tokens_total"] == usage["prompt_tokens"]
          and delta["prompt_tokens_cached_total"] == cached
          and delta["request_prefill_seconds_count"] == delta["time_to_first_token_seconds_count"] == 1
          and 0 < delta["request_prefill_seconds_sum"] <= delta["time_to_first_token_seconds_sum"],
          f"{case} contributes one completed-request sample")
    generated = delta["generation_tokens_total"]
    visible = usage["completion_tokens"]
    check(generated == visible if all_tokens_visible else generated >= visible,
          f"{case} counts generated tokens")


def wait_for_metrics(predicate, description):
    deadline = time.monotonic() + 5
    while True:
        sample = metrics()
        if predicate(sample):
            return sample
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {description}: {sample}")
        time.sleep(0.01)


def request(method, path, body=None, key=KEY):
    c = conn()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    c.request(method, path, body=body if isinstance(body, (bytes, str)) or body is None else json.dumps(body), headers=headers)
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, data


def stream(body, read_events=None, observe=None):
    """Streamed chat: (events, raw lines). Stops after read_events events if given (and drops the connection)."""
    c = conn()
    c.request("POST", "/v1/chat/completions", body=json.dumps(dict(body, stream=True)),
              headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    r = c.getresponse()
    assert r.status == 200, (r.status, r.read())
    events = []
    while True:
        line = r.fp.readline()
        if not line:
            break
        line = line.decode().rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            events.append("[DONE]")
            break
        events.append(json.loads(data))
        if observe:
            observe(events)
        if read_events and len(events) >= read_events:
            c.close()
            return events
    c.close()
    return events


def check_held_metrics(path):
    def admin(command):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
            c.settimeout(5)
            c.connect(path)
            c.sendall((command + "\n").encode())
            with c.makefile() as reply:
                return reply.readline().strip()

    first_reasoning = []

    def record_first_reasoning(events):
        choices = events[-1].get("choices", [])
        if not first_reasoning and choices and choices[0]["delta"].get("reasoning_content"):
            first_reasoning.append(time.monotonic())

    # The SSE [DONE] event can arrive while the previous request still owns its gate pass.
    before = wait_for_metrics(lambda m: m["requests_running"] == m["requests_waiting"] == 0, "an idle gate")
    body = {"messages": [{"role": "user", "content": "MOCK_LONG 100"}], "thinking": {"type": "enabled"}}
    hold = admin("hold 0")
    if hold != "ok":
        raise AssertionError(hold)
    with ThreadPoolExecutor(max_workers=1) as executor:
        started = time.monotonic()
        reply = executor.submit(stream, body, observe=record_first_reasoning)
        try:
            held = wait_for_metrics(lambda m: m["requests_waiting"] > 0, "a held request")
            check((held["requests_running"], held["requests_waiting"]) == (0, 1),
                  "metrics distinguish the held request from a running request")
            wait_started = time.monotonic()
            time.sleep(0.35)
            deliberate_wait = time.monotonic() - wait_started
        finally:
            release = admin("release")
            if release != "ok":
                raise AssertionError(release)
        check(reply.result()[-1] == "[DONE]", "held request completes after release")
    after = metrics()
    ttft = after["time_to_first_token_seconds_sum"] - before["time_to_first_token_seconds_sum"]
    check(first_reasoning and deliberate_wait <= ttft <= first_reasoning[0] - started,
          "frontend TTFT includes the gate hold and ends before the first reasoning delta")
    prefill = after["request_prefill_seconds_sum"] - before["request_prefill_seconds_sum"]
    check(0 < prefill <= ttft - deliberate_wait, "engine prefill excludes the frontend gate hold")


def collect(events):
    out = {"reasoning": "", "content": "", "calls": {}, "finish": None, "usage": None}
    for e in events:
        if e == "[DONE]" or "error" in e:
            continue
        if e.get("usage"):
            out["usage"] = e["usage"]
        for ch in e["choices"]:
            d = ch["delta"]
            out["reasoning"] += d.get("reasoning_content") or ""
            out["content"] += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                slot = out["calls"].setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
                slot["id"] = slot["id"] or tc.get("id")
                slot["name"] += tc["function"].get("name") or ""
                slot["arguments"] += tc["function"].get("arguments") or ""
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
    return out


TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
SYSTEM = {"role": "system", "content": "You are a coding agent. " + "Be careful. " * 50}


def main():
    s, b = request("GET", "/v1/models")
    check(s == 200 and b["data"][0]["id"], "models lists the served model")
    s, b = request("GET", "/v1/models", key="wrong")
    check(s == 401, "a wrong key is rejected")
    s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, key=None)
    check(s == 401, "a missing key is rejected")

    before = metrics()
    check(before["requests_running"] == before["requests_waiting"] == 0, "idle gate gauges are zero")
    cold_system = {"role": "system", "content": f"{uuid4().hex} {SYSTEM['content']}"}
    s, b = request("POST", "/v1/chat/completions", {"messages": [cold_system, {"role": "user", "content": "Hello there"}]})
    check(s == 200 and b["choices"][0]["message"]["content"] == "Mock reply to: Hello there", "non-streaming chat answer")
    check(b["choices"][0]["finish_reason"] == "stop" and b["usage"]["completion_tokens"] > 0, "finish_reason stop and usage")
    check("reasoning_content" not in b["choices"][0]["message"], "no reasoning_content when thinking is off (the default)")
    check(b["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, "the fresh session has no cached tokens")
    check_metrics(before, b["usage"], "fresh session", all_tokens_visible=True)

    s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello there"}],
                   "reasoning_effort": "none", "chat_template_kwargs": {"thinking": True}})
    check(s == 200 and b["choices"][0]["message"]["content"] == "Mock reply to: Hello there"
          and not b["choices"][0]["message"].get("reasoning_content"), "none disables thinking despite a conflicting template flag")

    # an explicit off-switch wins over an effort, and a template effort alone does not switch thinking on
    for extra, what in (({"reasoning_effort": "high", "chat_template_kwargs": {"thinking": False}}, "template thinking false wins over a top-level effort"),
                        ({"chat_template_kwargs": {"reasoning_effort": 40}}, "a template effort alone leaves thinking off")):
        s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello there"}], **extra})
        check(s == 200 and b["choices"][0]["message"]["content"] == "Mock reply to: Hello there"
              and not b["choices"][0]["message"].get("reasoning_content"), what)

    # the request's own thinking switch
    hi = [{"role": "user", "content": "Hello there"}]
    s, b = request("POST", "/v1/chat/completions", {"messages": hi, "thinking": {"type": "enabled"}})
    check(s == 200 and bool(b["choices"][0]["message"].get("reasoning_content")), "thinking: {type: enabled} switches thinking on")
    s, b = request("POST", "/v1/chat/completions", {"messages": hi, "thinking": {"type": "disabled"}, "reasoning_effort": "high"})
    check(s == 200 and b["choices"][0]["message"]["content"] == "Mock reply to: Hello there"
          and not b["choices"][0]["message"].get("reasoning_content"), "thinking: {type: disabled} switches thinking off")
    s, b = request("POST", "/v1/chat/completions", {"messages": hi, "thinking": {"type": "banana"}})
    check(s == 400 and "enabled, disabled" in b.get("error", {}).get("message", ""), "an unknown thinking type returns 400")

    allowed = "none, minimal, low, medium, high, xhigh, ultra, max"
    for streaming in (False, True):
        s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                       "reasoning_effort": "banana", "stream": streaming})
        check(s == 400 and allowed in b.get("error", {}).get("message", ""),
              f"invalid effort returns 400 and lists allowed values (stream={streaming})")
    for effort in allowed.split(", ")[1:]:
        s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                       "reasoning_effort": effort})
        check(s == 200 and bool(b["choices"][0]["message"].get("reasoning_content")), f"{effort} enables thinking")

    for effort in (40, "40", "HIGH", 0.5):
        s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                       "reasoning_effort": effort})
        check(s == 200 and bool(b["choices"][0]["message"].get("reasoning_content")), f"effort {effort!r} is still accepted")
    for effort in (0, 101, True):
        s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                       "reasoning_effort": effort})
        check(s == 400 and allowed in b.get("error", {}).get("message", ""), f"effort {effort!r} returns 400")

    # thinking, streamed, pi-style kwargs
    body = {"messages": [SYSTEM, {"role": "user", "content": "Think first"}], "max_completion_tokens": 200,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}}
    ev = stream(body)
    o = collect(ev)
    check(o["reasoning"].startswith("Mock reasoning") and o["content"] == "Mock reply to: Think first", "streamed reasoning and content")
    check(o["finish"] == "stop" and o["usage"] and o["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0, "stream usage with reasoning tokens")
    check(ev[-1] == "[DONE]" and ev[0]["choices"][0]["delta"].get("role") == "assistant", "stream starts with the role and ends with [DONE]")

    # an agent turn: tool call, then the tool result, reusing the prefix
    msgs = [SYSTEM, {"role": "user", "content": 'Please read it. MOCK_TOOL read {"path": "notes.txt"}'}]
    body = {"messages": msgs, "tools": TOOLS, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": True}}
    before = metrics()
    o = collect(stream(body))
    call = o["calls"].get(0)
    check(o["finish"] == "tool_calls" and call and call["name"] == "read" and json.loads(call["arguments"]) == {"path": "notes.txt"},
          "streamed tool call with finish_reason tool_calls")
    check(call and call["id"].startswith("call_"), "tool call has an id")
    check_metrics(before, o["usage"], "tool call")
    first_prompt = o["usage"]["prompt_tokens"]
    first_total = first_prompt + o["usage"]["completion_tokens"]
    msgs2 = msgs + [{"role": "assistant", "content": o["content"] or None, "reasoning_content": o["reasoning"],
                     "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": "read", "arguments": call["arguments"]}}]},
                    {"role": "tool", "tool_call_id": call["id"], "content": "line one\nline two"}]
    before = metrics()
    o2 = collect(stream(dict(body, messages=msgs2)))
    check(o2["content"] == "The tool returned: line one line two" and o2["finish"] == "stop", "answer after the tool result")
    cached = o2["usage"]["prompt_tokens_details"]["cached_tokens"]
    check(first_total - 1 <= cached < o2["usage"]["prompt_tokens"], f"tool-result turn reuses the whole previous turn (cached {cached} of {o2['usage']['prompt_tokens']}, previous turn {first_total})")
    check_metrics(before, o2["usage"], "tool result", all_tokens_visible=True)

    # stop strings, max_tokens, loop guard
    before = metrics()
    s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello stopword here"}], "stop": ["stopword"]})
    check(b["choices"][0]["message"]["content"] == "Mock reply to: Hello " and b["choices"][0]["finish_reason"] == "stop", "stop string cuts the output")
    check_metrics(before, b["usage"], "stop string")
    before = metrics()
    s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "MOCK_LONG 500"}], "max_tokens": 5})
    check(b["choices"][0]["finish_reason"] == "length" and b["usage"]["completion_tokens"] == 5, "max_tokens ends with length")
    check_metrics(before, b["usage"], "length limit", all_tokens_visible=True)
    before = metrics()
    t0 = time.time()
    s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "MOCK_LOOP"}], "max_tokens": 5000})
    n = b["usage"]["completion_tokens"]
    check(b["choices"][0]["finish_reason"] == "stop" and n < 400, f"loop guard stops a repeating output ({n} tokens, {time.time() - t0:.1f} s)")
    check_metrics(before, b["usage"], "loop guard")

    # bad requests
    for what, body in [("invalid JSON", b"{"), ("image input", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}),
                       ("n = 2", {"messages": [{"role": "user", "content": "x"}], "n": 2}), ("no messages", {"messages": []}),
                       ("unknown role", {"messages": [{"role": "robot", "content": "x"}]})]:
        s, b = request("POST", "/v1/chat/completions", body)
        check(s == 400 and "error" in b, f"400 for {what}")

    # a client that goes away mid-stream, then the server still answers
    before = metrics()

    def observe_active(events):
        if len(events) != 20:
            return
        active = metrics()
        check((active["requests_running"], active["requests_waiting"]) == (1, 0),
              "metrics count the frontend request while its stream is open")
        check(active["generation_tokens_total"] > before["generation_tokens_total"],
              "generation advances before the request completes")

    ev = stream({"messages": [{"role": "user", "content": "MOCK_LONG 3000"}]}, read_events=20, observe=observe_active)
    check(len(ev) == 20, "read part of a long stream, then disconnect")
    after = wait_for_metrics(lambda m: m["requests_running"] == 0, "the abandoned stream to release its gate pass")
    complete = ("prompt_tokens_total", "prompt_tokens_cached_total",
                "request_prefill_seconds_count", "time_to_first_token_seconds_count",
                "request_prefill_seconds_sum", "time_to_first_token_seconds_sum")
    check(all(after[name] == before[name] for name in complete),
          "an abandoned stream leaves completed request counters unchanged")
    s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "still alive?"}]})
    check(s == 200 and b["choices"][0]["message"]["content"] == "Mock reply to: still alive?", "server answers after a disconnect")

    # where a request starts: two pi-like conversations whose system text differs only in its last
    # line (the time) share all before it; dsfront's log (the third argument) says so, in counts only
    sysp = "You are an expert coding assistant. Read files, run commands and edit code.\n" * 40

    def pi(when, msg):
        return {"messages": [{"role": "system", "content": f"{sysp}Current date and time: Monday, October 12, 2026 at {when}"},
                             {"role": "user", "content": msg}], "max_tokens": 8, "reasoning_effort": "none"}

    s1, b1 = request("POST", "/v1/chat/completions", pi("09:14:08 AM", "fix the build"))
    s2, b2 = request("POST", "/v1/chat/completions", pi("09:15:31 AM", "write the tests"))
    u = b2.get("usage") or {}
    c2, p2 = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), u.get("prompt_tokens", 0)
    check(s1 == 200 and s2 == 200 and 0.9 * p2 < c2 < p2,
          f"a conversation with another's system text but a later time reuses what comes before the time (cached {c2} of {p2})")
    if args.log:
        time.sleep(0.2)
        plans = [l for l in open(args.log).read().splitlines() if " plan: " in l]
        check(len(plans) >= 2 and ", new: " in plans[-2] and ", a fork of " in plans[-1] and " ids in common" in plans[-1],
              f"the log says where each started: {plans[-2:]}")

    # concurrency
    results = [None] * 8

    def worker(i):
        results[i] = collect(stream({"messages": [{"role": "user", "content": f"parallel {i} MOCK_LONG 100"}], "max_tokens": 1000}))

    th = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    check(all(r and r["finish"] == "stop" and r["content"].count("word") == 100 for r in results), f"8 concurrent streams ({time.time() - t0:.1f} s)")

    if args.admin_socket:
        check_held_metrics(args.admin_socket)

    print(f"{len(FAILS)} failures")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
