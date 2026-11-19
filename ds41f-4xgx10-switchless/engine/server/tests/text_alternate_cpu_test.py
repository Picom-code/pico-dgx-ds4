"""Exercise the ordinary-text selector with synthetic CPU state and rank votes.

Run with python3. A C++17 host compiler is required. CUDA is not used.
"""
import os
from pathlib import Path
import subprocess
import tempfile

server = Path(__file__).resolve().parent.parent
serve = (server / 'serve.cpp').read_text()
cmd = (server / 'cmd.h').read_text()
protocol = (server.parent / 'common/frontend_protocol.h').read_text()

def part(source, start, stop):
    offset = source.index(start)
    return source[offset:source.index(stop, offset)]

program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>
constexpr int kReplay = 128;
namespace comm {
struct Ring {
    bool peer_ok = true;
    int votes = 0;
    void allgather_host(const int32_t* mine, int32_t* all, size_t bytes) {
        assert(bytes == sizeof(*mine));
        all[0] = *mine;
        all[1] = all[2] = 1;
        all[3] = peer_ok;
        ++votes;
    }
};
}
namespace model {
struct Model {
    struct StateInfo {
        int saved = 0, end_pos = -1;
        std::vector<int> snaps{};
        bool on_disk = false;
    };
    std::map<int, StateInfo> states;
    StateInfo state_info(int h) const {
        const auto i = states.find(h);
        return i == states.end() ? StateInfo{} : i->second;
    }
};
}
struct Conn { bool dead = false; };
using ConnP = std::shared_ptr<Conn>;
namespace fe { constexpr uint32_t ''' + part(protocol, 'kFlagImages =', ',') + r'''; }
constexpr int ''' + part(cmd, '  kCmdTextCheckpoint =', '\n};').split('//', 1)[0].strip().rstrip(',') + r''';
struct Cmd { int op = 0, handle = -1, pos0 = 0, next = 0, rows = 0, n = 0; };
''' + part(cmd, 'inline bool local_text_checkpoint_ready', 'model::Ctl to_ctl') + r'''
struct Session {
    int h = 0, slot = -1;
    std::vector<int32_t> hist{};
    uint64_t used = 0;
    int lo = 0, clo = 0, w = 0, dec_lo = 0;
    int end_lo = 0, end_clo = 0, end_w = 0, end_dec_lo = 0;
    std::vector<int> enc_snaps{};
    bool ordinary_text = false;
    std::weak_ptr<Conn> text_owner{};
};
''' + part(serve, 'enum { kCold', '// A run\'s time split') + r'''
struct Server {
    model::Model m_;
    comm::Ring ring_;
    struct { bool ced = true; } o_{};
    std::map<uint64_t, Session> sess_;
    std::set<uint64_t> flying_;
    std::vector<uint64_t> slot_{0, 0};
    int commands = 0;
    bool fits(int slot, int need) const { return slot == 0 ? need <= 2048 : need <= 1024; }
    void issue(const Cmd& c) { assert(c.op == kCmdTextCheckpoint && c.handle > 0); ++commands; }
    Session* find(uint64_t id) {
        const auto i = sess_.find(id);
        return i == sess_.end() ? nullptr : &i->second;
    }
''' + part(serve, '  Plan choose(', '  void tokens(') + r'''
    TextChoice adopt(const ConnP& conn, const std::vector<int32_t>& prompt, bool resume,
                     bool image_flag, Session* S, uint64_t src_id, Plan pl,
                     bool has_spans = false, bool has_rows = false) {
      struct { uint64_t base_session_id; uint32_t flags; } h{src_id, image_flag ? fe::kFlagImages : 0u};
      const int n = static_cast<int>(prompt.size());
      struct Request {
          ConnP conn;
          std::vector<int> spans;
          std::vector<uint16_t> img;
      };
      struct { Request rq; } f{{conn, {}, {}}};
      if (has_spans) f.rq.spans.push_back(1);
      if (has_rows) f.rq.img.push_back(1);
      Session* src = S ? S : find(src_id);
      int p = 0, s = 0, need = 1400;
      bool src_live = false;
      std::set<uint64_t> keep;
''' + part(serve, '    const bool image_request =', '    bool src_live =') + part(serve, '    if (!resume && ordinary_text && !S && h.base_session_id', '    const int q = pl.q;') + r'''
      assert(!S || src == S);
      return {src_id, p, src_live ? s : -1, pl};
    }
};
int main() {
    Server engine;
    auto conn = std::make_shared<Conn>();
    auto other = std::make_shared<Conn>();
    auto add = [&](uint64_t id, int length, int common, int saved, std::vector<int> snaps) -> Session& {
        auto& session = engine.sess_[id];
        session.h = static_cast<int>(id);
        session.hist.assign(length, 7);
        if (common < length) session.hist[common] = 8;
        session.ordinary_text = true;
        session.text_owner = conn;
        session.w = session.end_w = length;
        engine.m_.states[session.h] = {saved, length, std::move(snaps), false};
        return session;
    };
    const std::vector<int32_t> prompt(1400, 7);
    auto& named = add(1, 1600, 901, 1600, {1024});
    auto& shorter = add(2, 1200, 900, 1200, {512, 1024});
    auto& tied = add(3, 1300, 901, 1300, {512, 1024});
    auto& image = add(4, 1500, 1000, 1500, {768});
    auto& wrong_connection = add(5, 1500, 1000, 1500, {768});
    auto& kept_after_restart = add(6, 1500, 1000, 1500, {768});
    auto& spilled = add(7, 1500, 1000, 1500, {768});
    add(8, 1500, 1000, 511, {768}); // saved pages end too soon
    add(9, 1500, 1000, 1500, {768}); // flying
    image.ordinary_text = engine.ordinary_text_lineage(true, nullptr);
    wrong_connection.text_owner = other;
    kept_after_restart.ordinary_text = Session{}.ordinary_text;
    kept_after_restart.text_owner.reset();
    auto& derived = add(13, 1500, 1000, 1500, {768});
    derived.ordinary_text = engine.ordinary_text_lineage(false, &image);
    assert(!derived.ordinary_text);
    assert(!engine.ordinary_text_lineage(false, &kept_after_restart));
    engine.m_.states[spilled.h].on_disk = true;
    engine.flying_.insert(9);

    const Plan baseline = engine.choose(named, false, lcp(named.hist, prompt), prompt.size());
    assert(baseline.q == 0);
    if (!engine.ordinary_text_alternates_enabled()) {
        assert(engine.adopt(conn, prompt, false, false, nullptr, 1, baseline).id == 1);
        assert(engine.commands == 0);
        std::puts("disabled feature preserves baseline");
        return 0;
    }
    auto select = [&](int floor = 0) { return engine.text_alternate(conn, 1, prompt, 1400, {}, floor); };
    auto hit = select();
    assert(hit.id == 2 && hit.common == 900 && hit.restart.q == 512 && hit.restart.how == kSnap);
    tied.used = 1;
    assert(select().id == 3);
    tied.used = 0;
    assert(select(512).id == 0);
    conn->dead = true;
    assert(select().id == 0);
    conn->dead = false;

    // Rank 0 sees the winner. A peer without pages/the checkpoint rejects it.
    assert(engine.adopt(conn, prompt, false, false, nullptr, 1, baseline).id == 2);
    assert(engine.commands == 1 && engine.ring_.votes == 1);
    engine.ring_.peer_ok = false;
    assert(engine.adopt(conn, prompt, false, false, nullptr, 1, baseline).id == 1);
    assert(engine.commands == 2 && engine.ring_.votes == 2);
    engine.ring_.peer_ok = true;
    const Cmd probe{kCmdTextCheckpoint, 2, 512, 512, -1, 1};
    assert(local_text_checkpoint_ready(engine.m_.state_info(2), probe));
    assert(!local_text_checkpoint_ready({511, -1, {512}, false}, probe));
    assert(!local_text_checkpoint_ready({512, -1, {514}, false}, probe));
    assert(!local_text_checkpoint_ready({512, -1, {512}, true}, probe));
    const Cmd end_probe{kCmdTextCheckpoint, 2, 512, -1, 514, 1};
    assert(local_text_checkpoint_ready({512, 514, {}, false}, end_probe));
    assert(!local_text_checkpoint_ready({512, 512, {}, false}, end_probe));

    assert(engine.adopt(conn, prompt, true, false, nullptr, 1, baseline).id == 1);
    assert(engine.adopt(conn, prompt, false, true, nullptr, 1, baseline).id == 1);
    assert(engine.adopt(conn, prompt, false, false, nullptr, 1, baseline, true).id == 1);
    assert(engine.adopt(conn, prompt, false, false, nullptr, 1, baseline, false, true).id == 1);
    assert(engine.adopt(conn, prompt, false, false, nullptr, 4, baseline).id == 4);
    assert(engine.adopt(conn, prompt, false, false, nullptr, 6, baseline).id == 6);
    assert(engine.adopt(conn, prompt, false, false, nullptr, 13, baseline).id == 13);
    assert(engine.adopt(conn, prompt, false, false, &named, 1, baseline).id == 1);
    assert(engine.adopt(conn, prompt, false, false, nullptr, 99, {}).id == 2);
    const int before_no_scan = engine.commands;
    assert(engine.adopt(conn, prompt, false, false, nullptr, 0, {}).id == 0);
    assert(engine.adopt(conn, prompt, false, false, nullptr, 1, {1399, kLive, -1}).id == 1);
    assert(engine.commands == before_no_scan);

    auto& live = add(10, 1024, 950, 0, {});
    live.slot = 0;
    engine.slot_[0] = 10;
    hit = select();
    assert(hit.id == 10 && hit.slot == 0 && hit.restart.how == kLive && hit.restart.q == 950);
    assert(engine.text_alternate(conn, 1, prompt, 1400, {10}, 0).id == 2);
    live.hist[950] = 7;
    live.hist[512] = 8;
    live.w = 600; // usable live ring ends at a position that can serve the same restart
    live.hist.resize(600);
    assert(select().id == 10 && select().restart.q == 512); // live beats the equal stored candidate
    live.slot = 1;
    engine.slot_[0] = 0;
    engine.slot_[1] = 10;
    assert(select().id == 2); // its slot cannot fit the request

    auto& old_end = add(12, 1500, 850, 1500, {});
    old_end.slot = 0;
    old_end.end_w = engine.m_.states[old_end.h].end_pos = 900;
    engine.slot_[0] = 12;
    const Plan unsafe = engine.choose(old_end, true, 850, prompt.size());
    assert(unsafe.q == 850 && unsafe.how == kEnd);
    assert(select().id == 2); // persisting the borrowed slot would replace the older end
    old_end.slot = -1;
    engine.slot_[0] = 0;
    assert(select().id == 12); // the same end is usable when no slot overwrites it
    engine.sess_.erase(12);

    auto& tail = add(11, 1400, 1400, 1400, {1024, 1344, 1400});
    tail.enc_snaps = {1344};
    engine.m_.states[tail.h].end_pos = -1;
    hit = select();
    assert(hit.id == 11 && hit.restart.q == 1024); // decoder replay, and leave at least one input token
    tail.enc_snaps.clear();
    assert(select().restart.q == 1344);
    assert(shorter.hist.size() == 1200 && named.hist.size() == 1600);
    const std::vector<int32_t> long_prompt(1024 * 1024, 7);
    for (uint64_t id = 20; id < 25; ++id) add(id, long_prompt.size(), long_prompt.size(), long_prompt.size(), {65536});
    assert(engine.adopt(conn, long_prompt, false, false, nullptr, 1, baseline).id == 1);
    engine.sess_.erase(24);
    for (uint64_t id = 20; id < 24; ++id) engine.m_.states[id].on_disk = true;
    assert(engine.adopt(conn, long_prompt, false, false, nullptr, 1, baseline).id == 11);
    for (uint64_t id = 100; id < 357; ++id) engine.sess_[id] = Session{};
    assert(engine.adopt(conn, prompt, false, false, nullptr, 1, baseline).id == 1);
    std::puts("alternate selection, ownership, limits, checkpoint vote, and fallback passed");
}
'''
with tempfile.TemporaryDirectory(prefix='text-alternate-cpu-') as run_dir:
    fixture = Path(run_dir) / 'check.cpp'
    binary = Path(run_dir) / 'check'
    fixture.write_text(program)
    subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', str(fixture), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], env={**os.environ, 'DSGB10_ORDINARY_TEXT_ALTERNATES': '0'}, check=True)
    subprocess.run([str(binary)], env={**os.environ, 'DSGB10_ORDINARY_TEXT_ALTERNATES': '1'}, check=True)
