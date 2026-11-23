#!/usr/bin/env python3
"""Compile the Pico restore function and allocators with synthetic host I/O."""
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
    raise ValueError(f"unclosed definition {signature}")


def run(root, patched):
    model = root / "engine/model"
    source = (model / "state.cu").read_text()
    header = (model / "state.h").read_text()
    declarations = [
        line for line in header.splitlines()
        if line.startswith(("constexpr int kPagePos ", "constexpr size_t kPoolGrain "))
    ]
    if len(declarations) != 2:
        raise ValueError("missing source page position or pool grain constant")
    declarations.extend(
        extract(header, name) + ";"
        for name in ("struct Pool {", "struct Store {")
    )
    unspill = "void Model::Impl::state_unspill(int h" + (
        ", int window_limit, int preferred_snap, bool window_first)" if patched else ")"
    )
    signatures = [
        "size_t Pool::take_off(", "uint8_t* Pool::take(", "void Pool::give(",
        "bool Pool::take_at(", "size_t Pool::largest(", "void Pool::give_off(",
        "uint8_t* Model::Impl::store_alloc(", "void Model::Impl::store_release(",
        "void disk_release(", "void Model::Impl::state_free(",
        "size_t Model::Impl::disk_take(", "void Model::Impl::state_spill(", unspill,
    ]
    with tempfile.TemporaryDirectory(prefix="pico-state-restore-") as temp:
        folder = Path(temp)
        (folder / "store.inc").write_text("\n\n".join(declarations))
        (folder / "restore.inc").write_text(
            "\n\n".join(extract(source, signature) for signature in signatures)
        )
        fixture = Path(__file__).with_suffix(".cpp").resolve()
        binary = folder / "test_restore"
        flags = ["-DPATCHED_RESTORE"] if patched else []
        subprocess.run(
            [
                "nice", "-n", "19", os.environ.get("CXX", "c++"), "-std=c++17",
                "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                "-fno-sanitize-recover=undefined", *flags, "-I", temp,
                str(fixture), "-o", str(binary),
            ],
            check=True,
        )
        subprocess.run(["nice", "-n", "19", str(binary)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="original tree containing engine/model")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[3], help="patched tree containing engine/model")
    args = parser.parse_args()
    if args.baseline:
        run(args.baseline, patched=False)
    run(args.source, patched=True)


if __name__ == "__main__":
    main()
