"""Compile selected Pico serving code with synthetic token state and rank votes."""
import argparse
import hashlib
from pathlib import Path
import resource
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "serve.cpp")
parser.add_argument("--baseline", action="store_true")
args = parser.parse_args()
source_bytes = args.source.read_bytes()
source = source_bytes.decode()


def part(start, end):
    offset = source.index(start)
    return source[offset:source.index(end, offset)]


root = Path(__file__).parent
cases = (root / "text_existing_handle_cases.cpp").read_text()
parts = {
    "SESSION": part("struct Session {", "// The server's index"),
    "PLAN_TYPE": part("enum { kCold", "// A run's time split"),
    "PERSIST": part("  void persist_ops(", "  // Slot s's session"),
    "SELECTORS": part("  Plan choose(", "  static bool ordinary_text_alternates_enabled()")
    + part("  struct TextChoice {", "  bool text_spilled_alternate_ready("),
    "SELECTION": part("    bool src_live = src &&", "    const int q = pl.q;"),
    "DESTINATION": part("    std::vector<StateOp> v;\n    if (slot_[s]", "    const bool ced = o_.ced && n - q"),
}
for label, extracted in parts.items():
    marker = f"// {label}"
    assert cases.count(marker) == 1, marker
    cases = cases.replace(marker, extracted)

if not args.baseline:
    assert "const bool same_chat_lookup" in parts["SELECTION"]
    assert "*S = Session{}" in parts["DESTINATION"]
print("source sha256", hashlib.sha256(source_bytes).hexdigest(), flush=True)
with tempfile.TemporaryDirectory(prefix="pico-fresh-handle-") as tmp:
    fixture = Path(tmp) / "witness.cpp"
    binary = Path(tmp) / "witness"
    fixture.write_text(cases)
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    subprocess.run(["nice", "-n", "19", "c++", "-std=c++20", "-Wall", "-Wextra",
                    "-Werror", str(fixture), "-o", str(binary)], check=True)
    subprocess.run([str(binary), "baseline" if args.baseline else "candidate"], check=True)
