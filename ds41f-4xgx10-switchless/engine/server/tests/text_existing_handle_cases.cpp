// text_existing_handle_cpu_test.py inserts the serving code at the marked extraction points.
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string_view>
#include <utility>
#include <tuple>
#include <vector>

constexpr int kReplay = 128;
constexpr int kNever = 1 << 30;
namespace fe {
struct ImageSpan {};
enum { kFlagKeep = 1 };
}
struct Conn { bool dead = false; };
using ConnP = std::shared_ptr<Conn>;

// SESSION
// PLAN_TYPE

namespace model {
struct Model {
  struct StateInfo {
    int saved = 0, end_pos = -1;
    std::vector<int> snaps;
    size_t bytes = 0;
    bool on_disk = false;
  };
  struct Stored {
    std::vector<int32_t> pages, end;
    std::map<int, std::vector<int32_t>> snaps;
    bool on_disk = false;
    bool operator==(const Stored&) const = default;
  };
  std::map<int, Stored> entries;
  StateInfo state_info(int handle) const {
    const auto entry = entries.find(handle);
    StateInfo info;
    if (entry == entries.end()) return info;
    const auto& value = entry->second;
    info.saved = value.pages.size();
    info.end_pos = value.end.empty() ? -1 : static_cast<int>(value.end.size());
    info.bytes = 1;
    info.on_disk = value.on_disk;
    for (const auto& [pos, snap] : value.snaps) info.snaps.push_back(pos);
    return info;
  }
};
}

enum { kStLoad, kStRestore, kStRestoreEnd, kStCopySnaps, kStDrop, kStFree,
       kStSave, kStSnapEnd, kStMarkRamOnly };
struct StateOp { int kind, handle, a, b; };

struct Server {
  model::Model m_;
  struct { bool ced = true; int snap_every = 1024; } o_;
  std::map<uint64_t, Session> sess_;
  std::vector<uint64_t> slot_{1, 0, 0, 0};
  std::vector<int32_t> live;
  std::set<uint64_t> flying_;
  int next_h_ = 100;
  uint64_t clock_ = 0;
  bool ram_vote = true, disk_vote = true;
  bool existing_enabled = true;
  bool ram_enabled = true, disk_enabled = true;
  int ram_votes = 0, disk_votes = 0;
  bool fits(int, int) const { return true; }
  double unix_now() const { return 1.0; }
  bool ordinary_existing_text_alternates_enabled() const { return existing_enabled; }
  bool ordinary_text_alternates_enabled() const { return ram_enabled; }
  bool ordinary_text_spilled_alternates_enabled() const { return disk_enabled; }
  Session* find(uint64_t id) {
    const auto found = sess_.find(id);
    return found == sess_.end() ? nullptr : &found->second;
  }
  // SELECTORS
  bool text_alternate_ready(const Session&, const TextChoice&) {
    ++ram_votes;
    return ram_vote;
  }
  bool text_spilled_alternate_ready(const Session& source, const TextChoice& hit, int staged) {
    ++disk_votes;
    if (!disk_vote) return false;
    const auto& from = m_.entries.at(source.h);
    auto& to = m_.entries[staged];
    to.pages.assign(from.pages.begin(), from.pages.begin() + hit.restart.q);
    to.snaps.emplace(hit.restart.snap, from.snaps.at(hit.restart.snap));
    return true;
  }
  // PERSIST

  void apply(const std::vector<StateOp>& ops) {
    for (const auto& op : ops) {
      // Copy transfers snapshots only. Saved pages and end remain separate.
      switch (op.kind) {
        case kStLoad: {
          const auto& pages = m_.entries.at(op.handle).pages;
          assert(op.a <= static_cast<int>(pages.size()));
          live.assign(pages.begin(), pages.begin() + op.a);
          break;
        }
        case kStRestore: {
          const auto& snapshot = m_.entries.at(op.handle).snaps.at(op.a);
          assert(snapshot.size() <= live.size());
          assert(std::equal(snapshot.begin(), snapshot.end(), live.begin()));
          break;
        }
        case kStRestoreEnd:
          assert(live == m_.entries.at(op.handle).end);
          break;
        case kStCopySnaps: {
          const auto& from = m_.entries.at(op.handle);
          for (const auto& [position, snapshot] : from.snaps) {
            if (position <= op.b) m_.entries[op.a].snaps[position] = snapshot;
          }
          break;
        }
        case kStDrop: {
          auto& entry = m_.entries.at(op.handle);
          if (static_cast<int>(entry.pages.size()) > op.a) entry.pages.resize(op.a);
          if (static_cast<int>(entry.end.size()) > op.a) entry.end.clear();
          entry.snaps.erase(entry.snaps.upper_bound(op.a), entry.snaps.end());
          break;
        }
        case kStFree:
          assert(m_.entries.erase(op.handle) == 1);
          break;
        case kStSave: {
          auto& pages = m_.entries[op.handle].pages;
          assert(static_cast<int>(pages.size()) == op.a && op.b <= static_cast<int>(live.size()));
          pages.insert(pages.end(), live.begin() + op.a, live.begin() + op.b);
          break;
        }
        case kStSnapEnd:
          assert(op.a == static_cast<int>(live.size()));
          m_.entries[op.handle].end = live;
          break;
        default:
          assert(false);
      }
    }
  }

  int submit(const ConnP& owner, const std::vector<int32_t>& prompt) {
    const struct { uint64_t session_id, base_session_id; int flags; } h{1, 0, fe::kFlagKeep};
    const struct { struct { ConnP conn; std::vector<fe::ImageSpan> spans; } rq; } f{{owner, {}}};
    const bool resume = false;
    const int n = prompt.size(), need = n;
    const std::set<uint64_t> keep;
    Session* S = find(h.session_id);
    assert(S != nullptr);
    uint64_t src_id = h.session_id;
    Session* src = S;
    int s = S->slot >= 0 ? S->slot : 0;
    const bool ordinary_text = S->ordinary_text;
    // SELECTION
    (void)spilled_source_h;  // The model-free fixture does not run a rank command.
    const int q = pl.q;
    // DESTINATION
    apply(v);
    if (disk_source.h) apply({{kStFree, disk_source.h, 0, 0}});

    // Model prefill as extending the restored token content. No model logits.
    assert(q <= static_cast<int>(live.size()));
    live.resize(q);
    live.insert(live.end(), prompt.begin() + q, prompt.end());
    assert(live == prompt);
    S->hist = prompt;
    S->w = n;
    S->dec_lo = n - 127;
    auto& entry = m_.entries[S->h];
    entry.snaps[4096] = {live.begin(), live.begin() + 4096};
    return q;
  }

  void persist() {
    auto& own = sess_.at(1);
    std::vector<StateOp> ops;
    persist_ops(own, ops);
    apply(ops);
    const auto& saved = m_.entries.at(own.h);
    assert(saved.pages == live && saved.end == live);
    // Model releasing the active slot, forcing the next request through storage.
    own.slot = -1;
    slot_[0] = 0;
    live.clear();
  }
};

constexpr int kShallow = 4096, kDeep = 103424, kEdited = 104240;

std::vector<int32_t> history(int lineage, int length, int suffix) {
  std::vector<int32_t> tokens(length, 7);
  tokens[16384] = lineage;
  std::fill(tokens.begin() + 103600, tokens.end(), suffix);
  return tokens;
}

void seed(Server& server, const ConnP& owner, int id,
          const std::vector<int32_t>& tokens, int checkpoint, bool disk = false) {
  auto& session = server.sess_[id];
  session.h = id;
  session.hist = tokens;
  session.text_owner = owner;
  session.ordinary_text = true;
  session.w = session.end_w = tokens.size();
  auto& entry = server.m_.entries[id];
  entry.pages = entry.end = tokens;
  entry.snaps[kShallow] = {tokens.begin(), tokens.begin() + kShallow};
  entry.snaps[checkpoint] = {tokens.begin(), tokens.begin() + checkpoint};
  entry.on_disk = disk;
}

Server setup(const ConnP& owner, bool disk) {
  Server server;
  const auto old = history(10, 108000, 20);
  seed(server, owner, 1, old, kShallow);
  seed(server, owner, 2, history(11, 108100, 21), kDeep, disk);
  seed(server, owner, 3, history(10, 108200, 22), kDeep, disk);
  auto& own = server.sess_.at(1);
  own.slot = 0;
  own.pe_snap = 8500;
  own.enc_snaps = {90000};
  server.m_.entries.at(1).snaps[90000] = {old.begin(), old.begin() + 90000};
  server.live = old;
  return server;
}

void accepted(bool disk, bool baseline, const ConnP& owner) {
  auto server = setup(owner, disk);
  const auto sibling_a = server.m_.entries.at(2);
  const auto sibling_b = server.m_.entries.at(3);
  const auto* destination = &server.sess_.at(1);
  for (int edit = 0; edit < 4; ++edit) {
    const int old_handle = destination->h;
    const auto prompt = history(11 - edit % 2, kEdited - edit * 12, 40 + edit);
    const int q = server.submit(owner, prompt);
    assert(q == (baseline ? kShallow : kDeep));
    assert(&server.sess_.at(1) == destination);
    assert(destination->slot == 0 && server.slot_ == std::vector<uint64_t>({1, 0, 0, 0}));
    if (!baseline) {
      assert(destination->h != old_handle && !server.m_.entries.count(old_handle));
      assert(destination->pe_snap == -1);
      assert(destination->enc_snaps.empty());
      const auto& fresh = server.m_.entries.at(destination->h);
      assert(fresh.end.empty() && fresh.pages.empty() && !fresh.snaps.count(90000));
      assert(fresh.snaps.count(kDeep));
    } else {
      assert(destination->h == old_handle);
    }
    server.persist();
    auto exact = prompt;
    exact.insert(exact.end(), 4, 90);
    const int exact_handle = destination->h;
    assert(server.submit(owner, exact) == static_cast<int>(prompt.size()));
    assert(destination->h == exact_handle);
    server.persist();
    assert(server.m_.entries.size() == 3);
    assert(server.m_.entries.at(2) == sibling_a && server.m_.entries.at(3) == sibling_b);
  }
  printf("%s repeated edits: 4 edits at %d, 4 exact stored follow-ups, slot and siblings intact\n",
         disk ? "disk" : "RAM", baseline ? kShallow : kDeep);
}

void refused(bool disk, bool baseline, const ConnP& owner) {
  auto server = setup(owner, disk);
  const auto sibling = server.m_.entries.at(2);
  server.ram_vote = server.disk_vote = false;
  const auto prompt = history(11, kEdited, 40);
  assert(server.submit(owner, prompt) == kShallow);
  const auto* own = server.find(1);
  assert(own->h == 1 && own->slot == 0);
  assert(server.m_.entries.at(2) == sibling && server.m_.entries.size() == 3);
  assert(server.ram_votes == (!disk && !baseline));
  assert(server.disk_votes == (disk && !baseline));
  server.persist();
  auto next = prompt;
  next.push_back(91);
  assert(server.submit(owner, next) == static_cast<int>(prompt.size()));
  assert(server.find(1)->h == 1 && server.m_.entries.at(2) == sibling);
  server.persist();
  printf("%s vote refusal: own restart %d, original handle and sibling intact\n",
         disk ? "disk" : "RAM", kShallow);
}


void partial_append(const ConnP& owner, bool disk, bool baseline) {
  auto server = setup(owner, disk);
  auto original = history(11, 104000, 20);
  // Longest match is the original entire prompt. The competing sibling
  // diverges at 103600, but has more saved pages and a later snapshot.
  auto& own = server.sess_.at(1);
  own.hist = original;
  own.slot = -1;
  own.enc_snaps.clear();
  server.slot_[0] = 0;
  server.live.clear();
  auto& partial = server.m_.entries.at(1);
  partial = {};
  partial.pages.assign(original.begin(), original.begin() + 8192);
  partial.snaps[kShallow] = {original.begin(), original.begin() + kShallow};
  const auto sibling = server.m_.entries.at(2);
  auto prompt = original;
  prompt.resize(kEdited, 77);
  assert(lcp(original, prompt) == 104000 && lcp(server.sess_.at(2).hist, prompt) == 103600);
  assert(server.submit(owner, prompt) == (baseline ? kShallow : kDeep));
  server.persist();
  prompt.push_back(92);
  assert(server.submit(owner, prompt) == kEdited);
  assert(server.m_.entries.at(2) == sibling);
  server.persist();
  printf("%s partial append: %d positions, exact follow-up %d\n",
         disk ? "disk" : "RAM", baseline ? kShallow : kDeep, kEdited);
}

void mixed_sources(const ConnP& owner, bool baseline) {
  for (bool accept_disk : {false, true}) {
    auto server = setup(owner, false);
    constexpr int intermediate = 81920;
    seed(server, owner, 4, history(11, 108120, 30), intermediate);
    server.m_.entries.at(2).on_disk = true;
    const auto disk_before = server.m_.entries.at(2);
    const auto ram_before = server.m_.entries.at(4);
    server.disk_vote = accept_disk;
    const auto prompt = history(11, kEdited, 41);
    const int selected = server.submit(owner, prompt);
    assert(selected == (baseline ? kShallow : accept_disk ? kDeep : intermediate));
    assert(server.ram_votes == !baseline && server.disk_votes == !baseline);
    server.persist();
    assert(server.m_.entries.size() == 4);
    assert(server.m_.entries.at(2) == disk_before && server.m_.entries.at(4) == ram_before);
    printf("mixed: disk vote %d, selected %d\n", accept_disk, selected);
  }
}

void rollback_flag(const ConnP& owner) {
  auto server = setup(owner, false);
  server.existing_enabled = false;
  assert(server.submit(owner, history(11, kEdited, 40)) == kShallow);
  assert(server.find(1)->h == 1 && server.ram_votes == 0 && server.disk_votes == 0);
  puts("opt-in disabled: own handle and original restart retained");
}

void ineligible(const ConnP& owner) {
  const auto prompt = history(11, kEdited, 40);
  for (int test = 0; test < 6; ++test) {
    auto server = setup(owner, false);
    const auto other = std::make_shared<Conn>();
    auto& source = server.sess_.at(2);
    switch (test) {
      case 0: server.sess_.at(1).text_owner = other; break;
      case 1: source.text_owner = other; break;
      case 2: server.sess_.at(1).ordinary_text = false; break;
      case 3: server.flying_.insert(2); break;
      case 4: source.slot = 1; server.slot_[1] = 2; break;
      case 5: server.next_h_ = INT_MAX - 1; break;
    }
    assert(server.submit(owner, prompt) == kShallow);
    assert(server.find(1)->h == 1 && server.ram_votes == 0 && server.disk_votes == 0);
  }
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const bool baseline = std::string_view(argv[1]) == "baseline";
  const auto owner = std::make_shared<Conn>();
  for (const bool disk : {false, true}) {
    accepted(disk, baseline, owner);
    refused(disk, baseline, owner);
    partial_append(owner, disk, baseline);
  }
  mixed_sources(owner, baseline);
  rollback_flag(owner);
  ineligible(owner);
}
