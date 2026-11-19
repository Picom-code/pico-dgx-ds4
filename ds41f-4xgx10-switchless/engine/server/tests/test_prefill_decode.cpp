#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

void check(bool ok, const char* message) {
  if (!ok) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

double now_s() {
  static double t = 0;
  return ++t;
}

namespace fe {
enum : uint32_t { kStop = 1, kLength, kContext, kCancelled };
struct SubmitHeader {
  uint64_t session_id = 0, req_id = 0;
  uint32_t max_new_tokens = 20;
};
struct ImageSpan { int start, rows; };
}

namespace model {
namespace ops { constexpr int kDrafts = 5; }
struct Sampling { uint64_t seed = 0; };
struct SampleOut {
  struct Row { int token = 9; } row[8];
};
struct Model {
  bool pending = false;
  uint32_t ctl = 0;
  void drain() { pending = false; }
  SampleOut wait(uint64_t) { return {}; }
  bool spec() const { return true; }
  int slot_ctx(int) const { return 4096; }
  double last_gpu_ms() const { return 0; }
  double mark_ms(int, int) const { return 0; }
  void set_images(int, const void*, const void*, const void*) {}
  void set_ctl(uint32_t v) { ctl = v; }
};
}

enum { kCmdPrefill, kCmdRows, kCmdImages, kCmdState, kCmdDecode };
enum { kPfSample = 1, kPfCed = 2, kStDropSnap = 1, kStSnap = 2 };
enum : uint32_t { kEndStop, kEndLength, kEndCancel, kEndContext };
constexpr int kReplay = 128, kMaxSnaps = 8;

struct Cmd {
  int op = 0, slot = -1, pos0 = 0, rows = 0, flags = 0, next = 0, handle = -1;
  model::Sampling sp;
  int n_snaps = 0, snaps[kMaxSnaps]{}, tokens[8]{}, n = 0, n_ops = 0;
  struct Op { int kind, handle, pos, unused; } ops[2];
  int max_new = 0, limit = 0, n_stop = 0, stop[8]{}, ahead = 0, snap_every = 0;
  int spec = 0, spec_drafts = 0, spec_gate = 0;
  float spec_thresh = 0, spec_cost[3]{}, spec_row_ms[6]{};
  double spec_tok = 0, spec_ms = 0;
};

struct Watchdog { double started = 0; };
uint64_t run_prefill(model::Model& m, const Cmd& c) {
  m.pending = true;
  return c.flags & kPfSample ? 0 : ~0ull;
}
uint64_t run_rows(model::Model&, const Cmd&) { return 0; }
bool keep_snap(int, int, int) { return true; }

struct DecodeEnd {
  uint32_t why = kEndLength;
  int sampled = 1, fed = 2, written = 0;
  double spec_tok = 1, spec_ms = 0;
};

template <class OnSample>
DecodeEnd decode_loop(model::Model& m, const Cmd& c, Watchdog&, OnSample on) {
  check(!m.pending, "decode overtook prefill GPU work");
  check(c.max_new == 1, "interleaved decode exceeded its token allowance");
  check(c.spec && c.sp.seed == (uint64_t)c.handle, "decode lost peer draft or sampler settings");
  model::SampleOut sample;
  sample.row[0].token = 100 + c.pos0;
  on(0, sample, 0);
  return {kEndLength, 1, 2, c.pos0 + 2};
}

void add(DecodeEnd&, const DecodeEnd&) {}
[[noreturn]] void die(const char*, uint64_t, size_t, int) { std::abort(); }

struct Request {
  fe::SubmitHeader h;
  std::vector<int32_t> prompt, stop;
  std::vector<fe::ImageSpan> spans;
  bool cancelled = false;
};
struct Io {
  static bool cancelled(const Request& rq) { return rq.cancelled; }
  bool submit_waiting() const { return true; }
};
struct Turn {
  double rows_ms = 0, ops_gpu_ms = 0, tail_ms = 0, dec1_ms = -1;
  int chunks = 0;
  bool pending = false;
  void add_split(const model::Model& m) const { check(!m.pending, "prefill timing collected before drain"); }
};
struct Flight {
  Request rq;
  model::Sampling sp;
  std::vector<int32_t> out;
  Turn tu;
  double resume_s = 0, t_first = 0, paused_s = 0, t_paused = 0, dec_s = 0;
  double spec_tok = 1, spec_ms = 0;
  int decoded = 0, slices = 0;
  DecodeEnd de;
};
struct Session {
  int h = 0, slot = -1, w = 0, dec_lo = 0, pe_snap = -1;
  std::vector<int32_t> hist, enc_snaps;
};
struct Options {
  int prefill_rows = 128, small_rows = 8, snap_every = 128, slice = 64;
  bool snap_thin = true, spec = true;
  int spec_drafts = 5, spec_gate = 0;
  float spec_thresh = 0, spec_cost[3]{}, spec_row_ms[6]{};
};

struct Fixture {
  model::Model m_;
  Options o_;
  Watchdog wd_;
  Io io_;
  int next_prefill_peer_slot_ = 0;
  std::deque<Flight> rot_;
  std::map<uint64_t, Session> sessions;
  std::vector<uint64_t> slot_{1, 2, 3, 0};
  std::vector<Cmd> commands;
  std::vector<std::tuple<uint64_t, int>> sent;
  std::vector<std::tuple<uint64_t, uint32_t, size_t>> completed;
  int pending_lookahead_slot = -1;
  uint64_t cancel_after_first_chunk = 0;
  Flight* prefill_flight = nullptr;

  Session* find(uint64_t id) {
    const auto it = sessions.find(id);
    return it == sessions.end() ? nullptr : &it->second;
  }
  static bool stop_id(const Flight& f, int t) {
    return std::count(f.rq.stop.begin(), f.rq.stop.end(), t) > 0;
  }
  void issue(const Cmd& c) {
    commands.push_back(c);
    if (c.op == kCmdPrefill) {
      check(pending_lookahead_slot == -1 || pending_lookahead_slot == c.slot, "lookahead crossed slots");
      pending_lookahead_slot = c.next ? c.slot : -1;
      if (commands.size() == 1 && cancel_after_first_chunk) {
        if (cancel_after_first_chunk == prefill_flight->rq.h.session_id) prefill_flight->rq.cancelled = true;
        else rot_.front().rq.cancelled = true;
      }
    }
    if (c.op == kCmdDecode) check(pending_lookahead_slot == -1, "decode crossed pending Engram staging");
  }
  void tokens(const Request& rq, const model::SampleOut& o, int row) {
    sent.emplace_back(rq.h.session_id, o.row[row].token);
  }
  void finish(Flight& f, uint32_t why) {
    check(!m_.pending, "completion ran with prefill GPU work outstanding");
    completed.emplace_back(f.rq.h.session_id, why, commands.size());
  }
  bool ended(Flight& f, uint32_t why) { finish(f, why); return false; }
  bool admissible() const { return true; }

  Flight request(uint64_t id, int slot, int n, int q) {
    Flight f;
    f.rq.h.session_id = f.rq.h.req_id = f.sp.seed = id;
    f.rq.prompt.assign(n, (int)id);
    auto& s = sessions[id];
    s.h = (int)id;
    s.slot = slot;
    s.hist.assign(q, (int)id);
    return f;
  }
  void queue_peer(uint64_t id, int slot) {
    Flight f = request(id, slot, 16, 16);
    f.out = {42};
    rot_.push_back(std::move(f));
  }

#include "schedule.inc"
};

int main(int argc, char** argv) {
  check(argc == 2, "pass enabled or disabled");
  const bool enabled = std::strcmp(argv[1], "enabled") == 0;
  {
    Fixture x;
    Flight f = x.request(1, 0, 640, 128);
    x.queue_peer(2, 1);
    x.queue_peer(3, 2);
    x.queue_peer(99, -1);
    check(x.prefill(f, false, 0, 128, true), "prefill did not start decode");
    check(x.find(1)->hist == f.rq.prompt && x.find(1)->dec_lo == 512, "prefill history or CED replay boundary changed");
    check(x.commands[0].op == kCmdPrefill && x.commands[0].pos0 == 128, "reused prefix was recomputed");
    check(x.rot_[0].rq.h.session_id == 2 && x.rot_[1].rq.h.session_id == 3, "decode queue order changed");
    if (enabled) {
      check(x.sent == decltype(x.sent){{2, 116}, {3, 116}, {2, 117}, {1, 9}},
            "expected round-robin resident decode progress before the prompt's first token");
      check(x.find(2)->hist.size() == 19 && x.find(2)->hist[17] == 116 && x.find(2)->hist[18] == 117,
            "queued decode history did not resume from the emitted token");
      check(x.find(99)->slot == -1 && x.commands[0].next == 0, "displaced peer resumed or Engram ran ahead");
      for (size_t i = 0; i < x.commands.size(); ++i) {
        const auto& c = x.commands[i];
        const bool prefill = i % 2 == 0;
        check(c.op == (prefill ? kCmdPrefill : kCmdDecode), "rank command ordering changed");
        if (prefill) check(c.slot == 0 && c.pos0 == 128 + (int)i / 2 * 128,
                           "prefill resumed in the wrong slot or at the wrong position");
      }
    } else {
      check(x.sent == decltype(x.sent){{1, 9}} && x.commands[0].next == 128, "disabled path changed");
    }
  }
  {
    Fixture x;
    Flight f = x.request(1, 0, 384, 0);
    x.queue_peer(99, -1);
    check(x.prefill(f, false, 0, 0, true), "resident-free prefill failed");
    check(x.sent == decltype(x.sent){{1, 9}} && x.commands[0].next == 128, "resident-free path changed");
  }
  {
    Fixture x;
    Flight f = x.request(1, 0, 384, 0);
    x.queue_peer(2, 1);
    x.rot_.front().rq.spans.push_back({0, 3});
    check(x.prefill(f, false, 0, 0, true), "text prefill with image peer failed");
    check(x.sent == decltype(x.sent){{1, 9}} && x.commands[0].next == 128, "image peer was interleaved");
  }
  {
    for (int mode = 0; mode < 4; ++mode) {
      Fixture x;
      const bool short_prompt = mode == 0;
      Flight f = x.request(1, 0, short_prompt ? 16 : 384, 0);
      x.queue_peer(2, 1);
      if (mode == 1) f.rq.spans.push_back({0, 3});
      if (mode == 2) x.rot_.front().rq.cancelled = true;
      const bool resume = mode == 3;
      check(x.prefill(f, resume, 0, 0, !short_prompt), "excluded prefill failed");
      check(x.sent == decltype(x.sent){{1, 9}}, "excluded prefill yielded to a peer");
      check(x.commands[0].next == (short_prompt ? 0 : 128), "excluded prefill lost lookahead");
    }
  }
  {
    Fixture x;
    Flight f = x.request(1, 0, 384, 0);
    x.queue_peer(2, 1);
    x.rot_.front().rq.stop.push_back(116);
    check(x.prefill(f, false, 0, 0, true), "prefill with stopping peer failed");
    if (enabled) {
      check(x.completed == decltype(x.completed){{2, fe::kStop, 2}} && x.rot_.empty(), "stopping peer was not finalized");
      check(x.sent == decltype(x.sent){{2, 116}, {1, 9}}, "stopping peer emitted extra tokens");
      check(x.commands[2].next == 128, "lookahead did not resume after peer completed");
    }
  }
  {
    Fixture x;
    Flight f = x.request(1, 0, 384, 0);
    x.prefill_flight = &f;
    x.cancel_after_first_chunk = 1;
    x.queue_peer(2, 1);
    check(!x.prefill(f, false, 0, 0, true), "cancelled prompt still started decode");
    check(x.sent.empty() && x.commands.size() == 1, "cancelled prompt yielded to a peer");
  }
  {
    Fixture x;
    Flight f = x.request(1, 0, 384, 0);
    x.prefill_flight = &f;
    x.cancel_after_first_chunk = 2;
    x.queue_peer(2, 1);
    check(x.prefill(f, false, 0, 0, true), "prefill with cancelled peer failed");
    check(x.sent == decltype(x.sent){{1, 9}}, "cancelled peer emitted tokens");
    if (enabled) check(x.completed == decltype(x.completed){{2, fe::kCancelled, 1}} && x.rot_.empty(),
                       "cancelled peer was not finalized");
  }
}
