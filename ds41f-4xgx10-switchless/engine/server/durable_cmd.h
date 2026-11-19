// All-rank durable ordinary checkpoints. No request content or error text is logged.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace srv::durable {
constexpr int32_t save = 22, load = 23, drop = 24;
template<class Ring> bool vote(Ring& ring, bool yes) {
  const int32_t mine = yes ? 1 : 0;
  int32_t all[4] = {};
  ring.allgather_host(&mine, all, sizeof mine);
  return std::all_of(all, all + 4, [](int32_t v) { return v == 1; });
}
template<class Model> bool restartable(Model& m, int h, int snap) {
  if (h <= 0 || (snap & ~1) <= 0) return false;
  const auto info = m.state_info(h);
  return info.saved >= (snap & ~1) && std::find(info.snaps.begin(), info.snaps.end(), snap) != info.snaps.end();
}
// Save false owns no new file; save true owns only its newly published file.
// Load h must be absent on every rank. Never free a preexisting handle on refusal.
template<class Model, class Ring>
bool run(Model& m, Ring& ring, int32_t op, int h, uint64_t generation, int snap, bool encoder_only) {
  if (!vote(ring, generation != 0 && (op == save || op == load || op == drop))) return false;
  if (op == drop) { m.state_durable_drop(generation); return vote(ring, true); }
  if (op == save) {
    if (!vote(ring, restartable(m, h, snap))) return false;
    const bool created = m.state_durable_save(h, generation, snap, encoder_only);
    const bool complete = vote(ring, created);
    if (!complete && created) m.state_durable_drop(generation);
    // The second vote is a cleanup barrier before rank0 can publish/fallback.
    return vote(ring, complete);
  }
  const auto handles = m.state_sessions();
  const bool fresh = h > 0 && (snap & ~1) > 0 &&
      std::find(handles.begin(), handles.end(), h) == handles.end();
  if (!vote(ring, fresh)) return false;
  const bool loaded = m.state_durable_load(h, generation, snap, encoder_only);
  const bool complete = vote(ring, loaded && restartable(m, h, snap));
  if (!complete) m.state_free(h);
  return vote(ring, complete);
}
template<class Model, class Ring>
bool prune(Model& m, Ring& ring, const std::vector<uint64_t>& committed) {
  auto unique = committed;
  std::sort(unique.begin(), unique.end());
  const bool valid = unique.size() <= 128 && (unique.empty() || unique.front() != 0) &&
      std::adjacent_find(unique.begin(), unique.end()) == unique.end();
  if (!vote(ring, valid)) return false;
  return vote(ring, m.state_durable_prune(committed));
}
}  // namespace srv::durable
