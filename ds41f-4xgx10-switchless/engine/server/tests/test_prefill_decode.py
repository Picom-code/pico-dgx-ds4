#!/usr/bin/env python3
"""Compile real serve.cpp scheduling methods with a synthetic rank/model boundary."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def section(source, first, last):
    start = source.index(first)
    return source[start:source.index(last, start)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serve", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "serve.cpp")
    args = parser.parse_args()
    source = args.serve.read_text()
    helpers = ""
    if "static bool prefill_decode_enabled()" in source:
        helpers = section(source, "  static bool prefill_decode_enabled()", "  // Flight f's prompt run")
    tail = section(source, "    // ---- prefill [q, n)", "  // Vision: rq")
    decode = section(source, "  static constexpr uint32_t kPaused", "  // A run: the requests")
    decode += section(source, "  void rejoin(", "  // The flights' run together")
    with tempfile.TemporaryDirectory(prefix="pico-prefill-decode-") as temp:
        folder = Path(temp)
        (folder / "schedule.inc").write_text(
            helpers
            + "  bool prefill(Flight& f, bool resume, int s, int q, bool ced) {\n"
            + "    Session* S = find(f.rq.h.session_id);\n"
            + "    const auto& h = f.rq.h;\n"
            + "    const auto& prompt = f.rq.prompt;\n"
            + "    const int n = (int)prompt.size();\n"
            + "    Turn& tu = f.tu;\n"
            + "    const double t0 = now_s();\n"
            + tail
            + decode
        )
        fixture = Path(__file__).with_suffix(".cpp").resolve()
        binary = folder / "test_prefill_decode"
        subprocess.run(
            [
                "nice", "-n", "19", os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                "-Werror", "-fsanitize=undefined", "-I", temp, str(fixture), "-o", str(binary),
            ],
            check=True,
        )
        env = dict(os.environ, DSGB10_ORDINARY_PREFILL_DECODE="1")
        subprocess.run(["nice", "-n", "19", str(binary), "enabled"], check=True, env=env)
        env.pop("DSGB10_ORDINARY_PREFILL_DECODE")
        subprocess.run(["nice", "-n", "19", str(binary), "disabled"], check=True, env=env)


if __name__ == "__main__":
    main()
