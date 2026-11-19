#!/usr/bin/env python3
"""Exercise allocator functions with a synthetic pool and spill sink."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def extract(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    for match in re.finditer(r"[{}]", source[opening + 1 :]):
        depth += 1 if match.group() == "{" else -1
        if not depth:
            return source[start : opening + 2 + match.start()]
    raise ValueError(f"unclosed function {signature}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pico_root", type=Path, nargs="?", default=Path(__file__).resolve().parents[4])
    parser.add_argument("--expect-upstream-baseline", action="store_true")
    args = parser.parse_args()
    engine = args.pico_root / "ds41f-4xgx10-switchless" / "engine"
    source = (engine / "model/state.cu").read_text()
    signatures = [
        "uint8_t* Model::Impl::store_alloc(",
        "void Model::Impl::state_spill(",
        "void Model::Impl::state_copy_snaps(",
    ]
    if not args.expect_upstream_baseline:
        signatures.append("void Model::state_mark_ram_only(")
        cmd = (engine / "server/cmd.cpp").read_text()
        serve = (engine / "server/serve.cpp").read_text()
        assert "case kStMarkRamOnly: m.state_mark_ram_only(o.h);" in cmd
        assert serve.index("v.push_back({kStMarkRamOnly") < serve.index("v.push_back({kStCopySnaps")
    with tempfile.TemporaryDirectory(prefix="pico-image-pressure-") as temp:
        folder = Path(temp)
        (folder / "allocator.inc").write_text("\n\n".join(extract(source, item) for item in signatures))
        fixture = Path(__file__).with_suffix(".cpp").resolve()
        binary = folder / "test_image_spill"
        flags = ["-DEXPECT_UPSTREAM_BASELINE"] if args.expect_upstream_baseline else []
        subprocess.run(
            [
                "nice", "-n", "19", os.environ.get("CXX", "c++"), "-std=c++17", "-Wall",
                "-Wextra", "-Werror", "-fsanitize=undefined", *flags, "-I", temp, str(fixture),
                "-o", str(binary),
            ],
            check=True,
        )
        subprocess.run(["nice", "-n", "19", str(binary)], check=True)


if __name__ == "__main__":
    main()
