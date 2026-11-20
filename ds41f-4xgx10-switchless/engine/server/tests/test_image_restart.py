#!/usr/bin/env python3
"""Run the server's slot and restart selection with synthetic CPU-only state."""
import os
from pathlib import Path
import subprocess
import tempfile

from test_image_spill import extract


def main():
    engine = Path(__file__).resolve().parents[2]
    source = (engine / "server/serve.cpp").read_text()
    start = source.index("    // ---- the slot, where the state comes from")
    end = source.index("    NVTX_RANGE", start)
    with tempfile.TemporaryDirectory(prefix="pico-image-restart-") as temp:
        folder = Path(temp)
        (folder / "restart.inc").write_text(source[start:end])
        (folder / "planner.inc").write_text(extract(source, "  Plan choose("))
        (folder / "session.inc").write_text(
            extract(source, "struct Session") + ";\n"
            + source[source.index("constexpr int kReplay"):source.index("// The window snapshots")]
            + source[source.index("enum { kCold"):source.index("int lcp(")]
            + extract(source, "int lcp(")
        )
        binary = folder / "test_image_restart"
        subprocess.run(
            [
                "nice", "-n", "19", os.environ.get("CXX", "c++"), "-std=c++17", "-Wall",
                "-Wextra", "-Werror", "-fsanitize=undefined",
                "-I", str(engine), "-I", temp, str(Path(__file__).with_suffix(".cpp")),
                "-o", str(binary),
            ],
            check=True,
        )
        subprocess.run(["nice", "-n", "19", str(binary)], check=True)


if __name__ == "__main__":
    main()
