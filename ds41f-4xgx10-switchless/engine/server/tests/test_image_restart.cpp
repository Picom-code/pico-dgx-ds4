// Synthetic connection, slots, and state metadata around actual serve.cpp selection.
#include "server/image_prefix.h"

#include <cassert>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

namespace fe {
constexpr int kImageDim = 2;
struct ImageSpan {
  uint32_t start, rows;
};
struct SubmitHeader {
  uint64_t session_id = 99, base_session_id = 0, req_id = 1;
};
}  // namespace fe

struct Conn {
  bool dead = false;
};

namespace model {
struct Model {
  struct StateInfo {
    int saved = 0, end_pos = -1;
    std::vector<int> snaps;
  };
  std::map<int, StateInfo> sessions;
  StateInfo state_info(int h) const { return sessions.at(h); }
};
}  // namespace model

using namespace srv;
#include "session.inc"

struct Request {
  fe::SubmitHeader h;
  std::shared_ptr<Conn> conn = std::make_shared<Conn>();
  std::vector<fe::ImageSpan> spans{{128, 3}};
  std::vector<uint16_t> img{1, 2, 3, 4, 5, 6};
  std::vector<int32_t> prompt = std::vector<int32_t>(1000, 42);
};
struct Flight {
  Request rq;
};

double now_s() { return 1000; }
[[noreturn]] void die(const char*, unsigned long long) { std::abort(); }

struct Server {
  std::map<uint64_t, Session> sess_;
  std::set<uint64_t> flying_;
  model::Model m_;
  struct {
    bool ced = false;
  } o_;
  std::vector<int> slot_sizes{2048, 2048};
  bool enabled = true;
  int persisted = 0;
  bool save_has_room = false;

  Session* find(uint64_t id) {
    auto it = sess_.find(id);
    return it == sess_.end() ? nullptr : &it->second;
  }
  bool image_cache_enabled(const Request& rq) const { return enabled && !rq.spans.empty(); }
  int need_clamped(const Request&) const { return 1008; }
  bool fits(int s, int need) const { return slot_sizes.at(s) >= need; }
  void persist_slot(int slot, bool free) {
    assert(!free);
    ++persisted;
    if (!save_has_room) return;
    for (auto& [id, session] : sess_) {
      if (session.slot != slot) continue;
      const int end = (int)session.hist.size();
      m_.sessions.at(id).saved = m_.sessions.at(id).end_pos = end;
      session.end_lo = session.lo;
      session.end_clo = session.clo;
      session.end_w = session.w;
      session.end_dec_lo = session.dec_lo;
    }
  }
  void forget_evicted() {}

  int pick_slot(const std::set<uint64_t>&, int) const { return 1; }

#include "planner.inc"

  struct Result {
    uint64_t id;
    Plan plan;
    int slot, match;
    bool source_live;
  };

  Result restart(Flight& f, const std::set<uint64_t>& keep = {}) {
    const auto& h = f.rq.h;
    const auto& prompt = f.rq.prompt;
    const int n = (int)prompt.size();
    const bool resume = false;
#include "restart.inc"
    return {src ? src_id : 0, pl, s, p, src_live};
  }

  Session& add(int id, const Request& rq, size_t match, model::Model::StateInfo info) {
    auto& s = sess_[id];
    s.h = id;
    s.image = true;
    s.image_owner = rq.conn;
    s.image_used = now_s() - 1;
    s.hist = rq.prompt;
    s.hist[match] = 43;
    s.image_spans = rq.spans;
    s.image_rows = rq.img;
    m_.sessions[id] = std::move(info);
    return s;
  }
};

int main() {
  Flight f;
  Server server;
  auto& longer = server.add(1, f.rq, 901, {1000, -1, {}});
  auto& shorter = server.add(2, f.rq, 900, {1000, -1, {512}});
  auto result = server.restart(f);
  assert(result.id == 2 && result.plan.q == 512 && result.plan.how == kSnap);
  assert(result.match == 900 && !result.source_live);
  assert(longer.image_used == 999 && shorter.image_used == 1000);

  // At the same feasible position and cost, prefer recency, then the lower ID.
  server.m_.sessions[1].snaps = {512};
  shorter.used = 10;
  assert(server.restart(f).id == 2);
  longer.used = shorter.used;
  assert(server.restart(f).id == 1);

  // Exact image identity and live connection ownership still gate reuse.
  longer.image_rows[0] ^= 1;
  assert(server.restart(f).id == 2);
  const auto owner = std::make_shared<Conn>();
  shorter.image_owner = owner;
  assert(server.restart(f).plan.q == 0);
  shorter.image_owner = f.rq.conn;
  shorter.image_used = now_s() - 300;
  assert(server.restart(f).plan.q == 0);
  shorter.image_used = now_s();
  f.rq.conn->dead = true;
  assert(server.restart(f).plan.q == 0);
  f.rq.conn->dead = false;
  server.enabled = false;
  assert(server.restart(f).plan.q == 0);
  server.enabled = true;

  // A takeable live ring wins, but an incompatible or kept slot cannot use it.
  shorter.slot = 0;
  shorter.w = 1000;
  result = server.restart(f);
  assert(result.id == 2 && result.plan.q == 900 && result.plan.how == kLive && result.slot == 0);
  assert(result.source_live);
  result = server.restart(f, {2});
  assert(result.id == 2 && result.plan.q == 512 && result.plan.how == kSnap && result.slot == 1);
  server.slot_sizes[0] = 512;
  result = server.restart(f);
  assert(result.id == 2 && result.plan.q == 512 && result.plan.how == kSnap && result.slot == 1);
  assert(server.persisted == 2);  // The projected end save ran out of room in both cases.

  // After a failed save, reevaluate other sources with real retained checkpoints.
  longer.image_rows = f.rq.img;
  server.m_.sessions[1].snaps = {768};
  result = server.restart(f);
  assert(result.id == 1 && result.plan.q == 768 && result.plan.how == kSnap);
  longer.image_rows[0] ^= 1;

  server.save_has_room = true;
  result = server.restart(f);
  assert(result.id == 2 && result.plan.q == 900 && result.plan.how == kEnd && result.slot == 1);
  assert(server.persisted == 4);

  // For an equal position, prefer a live ring over reloading a stored end.
  longer.image_rows = f.rq.img;
  longer.slot = 1;
  longer.w = 1000;
  longer.hist[900] = 43;
  shorter.used = longer.used + 1;
  result = server.restart(f);
  assert(result.id == 1 && result.plan.q == 900 && result.plan.how == kLive);
  longer.image_rows[0] ^= 1;
  longer.slot = -1;
  shorter.slot = -1;

  // Saved pages and CED replay gate what the snapshot can actually restore.
  server.m_.sessions[2].saved = 511;
  assert(server.restart(f).plan.q == 0);
  server.m_.sessions[2] = {1000, -1, {512, 896}};
  shorter.enc_snaps = {896};
  server.o_.ced = true;
  assert(server.restart(f).plan.q == 512);
}
