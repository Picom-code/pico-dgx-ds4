#!/usr/bin/env python3
"""Compile the Pico planner and reservation code against synthetic host fixtures."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def section(source, start, end):
    left = source.index(start)
    return source[left:source.index(end, left)]


def harness(source, patched):
    scheduler = (source / "engine/server/serve.cpp").read_text()
    main = (source / "engine/server/main.cpp").read_text()
    planner = section(scheduler, "  Plan choose(", "  void tokens(")
    retention = section(scheduler, "bool keep_snap(", "enum { kCold")
    reserve = section(main, "    const uint64_t mine[3] = {dfree, avail,", "  }\n  ring.barrier();")
    predicate = section(scheduler, "      for (int sn = (pos / o_.snap_every", "      wd_.started")
    return (
        f"#define PATCHED {int(patched)}\n"
        + r"""
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

constexpr size_t MiB = 1u << 20;
constexpr double gib = 1024 * MiB;
constexpr int kReplay = 128, kMaxSnaps = 8;
size_t free_after, host_after;
int cudaMemGetInfo(size_t* free, size_t*) { *free = free_after; return 0; }
size_t mem_available() { return host_after; }
void die(const char*, ...) { throw false; }
#define CUDA_CHECK(op) assert((op) == 0)

struct Ring {
    uint64_t before[4][3] = {{1024*MiB, 16384*MiB, 0}, {8192*MiB, 16384*MiB, 0},
                            {512*MiB, 16384*MiB, 0}, {4096*MiB, 16384*MiB, 0}};
    uint64_t after[4][3] = {};
    bool called = false;
    void allgather_host(const uint64_t* local, uint64_t dest[4][3], size_t size) {
        assert(size == sizeof before[0]);
        if (!called) assert(local[0] == before[0][0] && local[2] == before[0][2]);
        else assert(local[1] == free_after && local[2] == host_after);
        memcpy(dest, called ? after : before, sizeof before);
        called = true;
    }
};
namespace model {
struct Model {
    struct StateInfo { int saved = 0, end_pos = -1; std::vector<int> snaps; } info;
    size_t requested = 0;
    void state_reserve(size_t amount) { requested = amount; }
    size_t state_capacity() const { return requested; }
    StateInfo state_info(int) const { return info; }
};
}
bool reserve_pool(Ring& ring, model::Model& m, size_t minimum_mib) {
    struct Args { int rank; double session_gib; size_t min_session_mib; };
    const Args a{0, 12, minimum_mib};
    [[maybe_unused]] size_t dtotal = 0;
    size_t dfree = ring.before[0][0], avail = ring.before[0][1];
    const size_t margin = 2048*MiB, cache = 4096*MiB;
    try {
""" + reserve + r"""
    } catch (bool) { return false; }
    return true;
}
""" + retention + r"""
enum { kCold, kLive, kEnd, kSnap };
struct Plan { int q = 0, how = kCold, snap = -1; };
struct Session {
    int h = 1, lo = 0, clo = 0, w = 0, dec_lo = 0;
    int end_lo = 0, end_clo = 0, end_w = 0, end_dec_lo = 0;
    std::vector<int> hist, enc_snaps;
};
struct Scheduler {
    struct Options {
        bool ced = true, snap_thin = true, keep_early_snapshot = PATCHED;
        int snap_every = 1024;
    } o_;
    model::Model m_;
""" + planner + r"""
    std::vector<int> capture(bool images, int n) {
        struct { struct { std::vector<int> spans; } rq; } f;
        if (images) f.rq.spans.push_back(1);
        std::vector<int> out;
        for (int pos = 0; pos < n; pos += 8192) {
            int R = std::min(n - pos, 8192);
            struct { int n_snaps = 0, snaps[kMaxSnaps]; } c;
""" + predicate + r"""
            out.insert(out.end(), c.snaps, c.snaps + c.n_snaps);
        }
        return out;
    }
};
int main() {
    model::Model m;
    Ring constrained;
    assert(reserve_pool(constrained, m, 0) && m.requested == 0);
    Ring gated;
    assert(reserve_pool(gated, m, 1024) == !PATCHED && m.requested == 0);
    Ring fit;
    for (int r = 0; r < 4; ++r) {
        fit.before[r][0] = (r == 2 ? 3072 : 4096)*MiB;
        fit.before[r][2] = fit.before[r][0] - 2048*MiB;
        fit.after[r][0] = 1024*MiB;
        fit.after[r][1] = fit.before[r][2] + 1024*MiB;
        fit.after[r][2] = 24*1024*MiB;
    }
    free_after = fit.after[0][1], host_after = fit.after[0][2];
    auto device_veto = fit, host_veto = fit, capacity_veto = fit;
    device_veto.after[2][1] -= MiB;
    host_veto.after[2][2] = 6144*MiB - 1;
    capacity_veto.after[2][0] -= MiB;
    assert(reserve_pool(fit, m, 1024) && m.requested == 1024*MiB);
    assert(reserve_pool(device_veto, m, 1024) == !PATCHED);
    assert(reserve_pool(host_veto, m, 1024) == !PATCHED);
    assert(reserve_pool(capacity_veto, m, 1024) == !PATCHED);

    constexpr int n = 384*1024+3, anchor = 4096;
    Scheduler scheduler;
    Session session;
    session.hist.resize(n);
    session.w = n;
    const auto text = scheduler.capture(false, n);
    const auto images = scheduler.capture(true, n);
    const auto has_anchor = [=](const auto& snapshots) {
        return std::find(snapshots.begin(), snapshots.end(), anchor) != snapshots.end();
    };
    assert(has_anchor(text) == PATCHED && !has_anchor(images));
    assert(text.size() == images.size() + PATCHED);
    scheduler.o_.keep_early_snapshot = false;
    assert(scheduler.capture(false, n) == images);
#if PATCHED
    assert(keep_snap(anchor, n + 16384, 1024, true));
#endif
    assert(!keep_snap(anchor, n + 16384, 1024));
    scheduler.m_.info.snaps = text;
    session.enc_snaps = text;
    if (PATCHED) {
        assert(scheduler.choose(session, true, 7168, n+8192).q == anchor);
        assert(scheduler.choose(session, true, anchor - 1, n+8192).q == 0);
        assert(scheduler.choose(session, true, anchor, anchor + 127).q == 0);
        assert(scheduler.choose(session, true, anchor, anchor + 128).q == anchor);
        assert(scheduler.choose(session, false, 7168, n+8192).q == 0);
        scheduler.m_.info.saved = anchor;
        assert(scheduler.choose(session, false, 7168, n+8192).q == anchor);
    } else {
        assert(scheduler.choose(session, true, 7168, n+8192).q == 0);
    }
    assert(scheduler.choose(session, true, n - 40, n + 1024).q == n - 41);
    // Use a long synthetic history and diverge
    // behind the live window, then supply the windows the capture code selects.
    constexpr int long_n = 512*1024+181, shared = long_n-300;
    session.hist.resize(long_n);
    session.w = long_n;
    scheduler.m_.info = {};
    session.enc_snaps.clear();
    assert(scheduler.choose(session, true, shared, long_n).q == 0);
    scheduler.m_.info.snaps = scheduler.capture(false, long_n);
    session.enc_snaps = scheduler.m_.info.snaps;
    assert(scheduler.choose(session, true, shared, long_n).q == (shared / 1024) * 1024);
#if PATCHED
    // The end wins on disk, but the restore may only have space for the smaller window.
    scheduler.m_.info.saved = 10000;
    scheduler.m_.info.end_pos = 10000;
    scheduler.m_.info.snaps = {8192, 9984};
    session.end_w = 10000;
    session.enc_snaps.clear();
    Plan window;
    assert(scheduler.choose(session, false, 10000, 10001, &window).how == kEnd);
    assert(window.how == kSnap && window.q == 9984);
#endif
    puts(PATCHED ? "Patched fixture passed" : "Baseline reproduced");
}
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pico-checkpoint-") as directory:
        directory = Path(directory)
        for source, patched in ([(args.baseline, False)] if args.baseline else []) + [(args.source, True)]:
            cpp, executable = directory / "test.cpp", directory / "test"
            cpp.write_text(harness(source, patched))
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror", cpp, "-o", executable],
                check=True,
            )
            subprocess.run([executable], check=True)


if __name__ == "__main__":
    main()
