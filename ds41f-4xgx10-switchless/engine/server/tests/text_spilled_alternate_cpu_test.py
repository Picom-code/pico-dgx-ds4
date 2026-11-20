"""Exercise the exact spilled selector, staging vote, and request adoption on CPU."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
serve = (root / "server/serve.cpp").read_text()
cmd = (root / "server/cmd.h").read_text()


def between(text, first, last):
    start = text.index(first)
    return text[start:text.index(last, start)]

adopt = between(serve, '    if (!resume && ordinary_text && !S && h.base_session_id && (h.flags & fe::kFlagKeep)',
                '    // ---- prefill [q, n)')
before_token, token = adopt.split('      c.op = kCmdTokens;', 1)
assert 'issue(' not in before_token and 'state({' not in before_token
assert '{kStLoad, src->h, q, 0}' in before_token and '{kStRestore, src->h, pl.snap, 0}' in before_token
assert 'spilled_source_h = alt->h;' in before_token
assert token.index('c.handle = spilled_source_h;') < token.index('issue(c);')
assert token.index('issue(c);') < token.index('m_.set_tokens(') < token.index('{kStFree, disk_source.h, 0, 0}')
apply_ops = between((root / 'server/cmd.cpp').read_text(), 'void apply_ops(', 'Cmd bcast(')
pin_state = apply_ops.index('m.state_pin(pin);')
assert apply_ops.index('pin.push_back(c.ops[i].h);') < pin_state
assert apply_ops.index('pin.push_back(c.handle);') < pin_state < apply_ops.index('apply_state(m, c.ops[i]);')


program = r'''
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>
#include <cstdint>
constexpr int kReplay = 128, kCmdStageTextCheckpoint = 27;
namespace fe { constexpr int kFlagKeep = 2; }
enum { kCold, kLive, kEnd, kSnap };
struct Plan { int q = 0, how = kCold, snap = -1; };
struct Conn { bool dead = false; };
using ConnP = std::shared_ptr<Conn>;
struct Session {
    int h = 0, slot = -1;
    std::vector<int32_t> hist;
    bool ordinary_text = true;
    std::weak_ptr<Conn> text_owner;
    uint64_t used = 0;
    std::vector<int> enc_snaps;
};
struct Cmd { int op = 0, handle = 0, next = 0, pos0 = 0; };
struct Ring {
    bool reject_after_load = false;
    int votes = 0;
    void allgather_host(const int32_t* mine, int32_t* all, size_t size) {
        assert(size == sizeof(*mine));
        all[0] = *mine;
        all[1] = all[2] = 1;
        all[3] = !(reject_after_load && votes == 1);
        ++votes;
    }
};
struct Model {
    struct StateInfo {
        bool on_disk = false;
        int saved = 0;
        std::vector<int> snaps;
    };
    std::map<int, StateInfo> states;
    int reads = 0;
    StateInfo state_info(int h) const {
        auto it = states.find(h);
        return it == states.end() ? StateInfo{} : it->second;
    }
    std::vector<int> state_sessions() {
        std::vector<int> ids;
        for (auto& [id, state] : states) ids.push_back(id);
        return ids;
    }
    bool state_stage_spilled_snapshot(int src, int dest, int snap) {
        assert(states.at(src).on_disk);
        ++reads;
        return states.emplace(dest, StateInfo{false, snap & ~1, {snap}}).second;
    }
    void state_free(int h) { states.erase(h); }
};
namespace srv {
''' + between(cmd, 'template<class Ring>\nbool text_checkpoint_vote', 'inline bool text_checkpoint_ready') + between(
    cmd, 'template<class Model, class Ring>\nbool stage_text_checkpoint', 'model::Ctl to_ctl') + r'''
struct Fixture {
    Model m_;
    Ring ring_;
    struct { bool ced = true; } o_;
    std::map<uint64_t, Session> sess_;
    std::set<uint64_t> flying_;
    int next_h_ = 100, issued = 0;
    Cmd issued_command;
    void issue(const Cmd& c) { ++issued; issued_command = c; }
    Session* find(uint64_t id) {
        auto it = sess_.find(id);
        return it == sess_.end() ? nullptr : &it->second;
    }
''' + between(serve, '  static bool ordinary_text_spilled_alternates_enabled()', '  // Alternate sources') + between(
    serve, '  // Cheap disk metadata', '  // Validate only the winner.'
) + r'''
    int adopt(const ConnP& conn, const std::vector<int32_t>& prompt, bool resume,
              bool image_request, bool in_place, int flags, int named, int floor, bool named_text = true) {
        struct { struct { ConnP conn; } rq; } f{{conn}};
        struct { int base_session_id, flags; } h{named, flags};
        Session named_ram;
        named_ram.ordinary_text = named_text;
        const bool ordinary_text = ordinary_text_lineage(image_request, &named_ram);
        Session* S = in_place ? &named_ram : nullptr;
        Session* src = &named_ram;
        Session disk_source;
        int spilled_source_h = 0;
        int src_id = named, p = floor;
        bool src_live = true;
        Plan pl{floor, kLive, -1};
''' + between(serve, '    if (!resume && ordinary_text && !S && h.base_session_id && (h.flags & fe::kFlagKeep)',
              '    const int q = pl.q;') + r'''
        if (disk_source.h) {
            assert(src == &disk_source && src_id != named && !src_live);
            assert(spilled_source_h == find(src_id)->h && m_.state_info(spilled_source_h).on_disk);
            assert(p >= pl.q && m_.state_info(src->h).saved == pl.q);
            m_.state_free(disk_source.h);
        } else {
            assert(src == &named_ram && src_id == named && p == floor && src_live && pl.q == floor);
            assert(!spilled_source_h);
        }
        return pl.q;
    }
};
}
int main(int argc, char** argv) {
    using namespace srv;
    const bool enabled = argc == 2 && argv[1][0] == '1';
    Fixture t;
    auto owner = std::make_shared<Conn>();
    auto outsider = std::make_shared<Conn>();
    std::vector<int32_t> prompt(20000, 42);
    auto add = [&](int id, bool disk, int saved, std::vector<int> snaps) {
        auto& s = t.sess_[id];
        s = Session{id, -1, prompt, true, owner, 1, {}};
        t.m_.states[id] = Model::StateInfo{disk, saved, snaps};
        return &s;
    };
    add(1, false, 10000, {8192});
    add(2, true, 17000, {16000, 17000});
    add(3, true, 19950, {19950})->enc_snaps = {19950};
    add(4, true, 19900, {19800})->text_owner = outsider;
    add(5, true, 19900, {19800})->ordinary_text = false;
    add(6, true, 19900, {19800});
    t.flying_.insert(6);
    add(7, true, 19900, {19800})->slot = 0;
    add(8, true, 16500, {19800});
    add(9, true, 19900, {19800})->hist[15000] = 1;
    assert(t.text_spilled_alternate(owner, 1, prompt, 8808).id == 2); // exactly 8192 gain
    assert(!t.text_spilled_alternate(owner, 1, prompt, 8809).id);
    assert(!t.text_spilled_alternate(owner, 2, prompt, 8808).id);
    owner->dead = true;
    assert(!t.text_spilled_alternate(owner, 1, prompt, 8808).id);
    owner->dead = false;
    t.o_.ced = false;
    assert(t.text_spilled_alternate(owner, 1, prompt, 8808).id == 2);
    t.o_.ced = true;
    assert(t.adopt(owner, prompt, false, false, false, fe::kFlagKeep, 1, 8808) == (enabled ? 17000 : 8808));
    assert(t.issued == enabled && t.m_.reads == enabled);
    if (enabled) {
        assert(t.issued_command.op == kCmdStageTextCheckpoint && t.issued_command.handle == 2 &&
               t.issued_command.next == 100 && t.issued_command.pos0 == 17000);
    }
    t.ring_.reject_after_load = true;
    t.ring_.votes = 0;
    assert(t.adopt(owner, prompt, false, false, false, fe::kFlagKeep, 1, 8808) == 8808);
    assert(t.m_.reads == 2 * enabled && !t.m_.states.count(101) && t.m_.states.at(2).on_disk);
    assert(t.adopt(owner, prompt, false, false, true, fe::kFlagKeep, 1, 8808) == 8808);
    assert(t.adopt(owner, prompt, true, false, false, fe::kFlagKeep, 1, 8808) == 8808);
    assert(t.adopt(owner, prompt, false, true, false, fe::kFlagKeep, 1, 8808) == 8808);
    assert(t.adopt(owner, prompt, false, false, false, fe::kFlagKeep, 0, 8808) == 8808);
    assert(t.adopt(owner, prompt, false, false, false, 0, 1, 8808) == 8808);
    assert(t.adopt(owner, prompt, false, false, false, fe::kFlagKeep, 1, 8808, false) == 8808);
    assert(t.issued == 2 * enabled);
    for (int id = 10; id < 258; ++id) add(id, true, 19000, {18000});
    t.sess_.erase(257);
    assert(t.sess_.size() <= 256 && !t.text_spilled_alternate(owner, 1, prompt, 8808).id);
    add(257, true, 19000, {18000});
    add(258, false, 19000, {18000});
    assert(t.sess_.size() > 256 && !t.text_spilled_alternate(owner, 1, prompt, 8808).id);
}
'''

with tempfile.TemporaryDirectory(prefix="pico-spilled-cpu-") as tmp:
    work = Path(tmp)
    source, binary = work / "test.cpp", work / "test"
    source.write_text(program)
    subprocess.run(["g++", "-std=c++20", "-O1", "-I", root, source, "-o", binary], check=True)
    for flag in ("0", "1"):
        subprocess.run([binary, flag], check=True,
                       env={**os.environ, "DSGB10_ORDINARY_TEXT_SPILLED_ALTERNATES": flag})
print("spilled selector, gate, rank refusal, and bounded lookup tests passed")
