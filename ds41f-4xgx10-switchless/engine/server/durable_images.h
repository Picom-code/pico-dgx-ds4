// Ordinary image restore is a second tier below the connection-local RAM cache.
// The head index is the commit record. Each rank owns its own immutable payload.
#pragma once

#include "server/serve.h"
#include "server/durable_image_index.h"

namespace srv {
class DurableImages {
 public:
  bool open(const ServeOptions& o, model::Model& m, comm::Ring& ring) {
    if (!o.durable_image_bytes || o.durable_image_dir.empty()) return false;
    if (!index_.open(o.durable_image_dir, o.durable_image_namespace,
                     durable_image::Limits{128, o.durable_image_bytes, UINT64_MAX})) return false;
    // Only a successfully read head index is authoritative for orphan cleanup.
    durable_prune(m, ring, index_.ids());
    return true;
  }
  bool enabled() const { return index_.enabled(); }

  struct Hit {
    int h, pos, snap;
    bool encoder_only;
  };

  std::optional<Hit> load(model::Model& m, comm::Ring& ring, int& next_handle,
                          const std::vector<int32_t>& tokens, const std::vector<fe::ImageSpan>& spans,
                          const std::vector<uint16_t>& rows, uint64_t now, bool ced) {
    auto images = durable_image::identify(tokens, spans, rows);
    if (!images) return {};
    std::set<uint64_t> tried;
    // A transient refusal must not destroy a good checkpoint. Try at most
    // one shorter branch, and allow a later completion to repair the entry.
    for (int attempt = 0; attempt < 2; ++attempt) {
      const auto* e = index_.find(tokens, *images, now, ced, tried);
      dispose_evicted(m, ring);
      if (!e || next_handle == INT_MAX) break;
      Hit hit{next_handle++, static_cast<int>(e->tokens.size()), e->snap, e->encoder_only};
      if (durable_state(m, ring, kCmdDurableLoad, hit.h, e->id, hit.snap, hit.encoder_only)) {
        failed_.erase(e->id);
        return hit;
      }
      failed_.insert(e->id);
      tried.insert(e->id);
    }
    return {};
  }

  void save(model::Model& m, comm::Ring& ring, int handle, uint64_t id,
            const std::vector<int32_t>& tokens, const std::vector<fe::ImageSpan>& spans,
            const std::vector<uint16_t>& rows, int snap, bool encoder_only, uint64_t now,
            size_t budget) {
    if (!id || index_.contains(id)) return;
    const auto images = durable_image::identify(tokens, spans, rows);
    if (!images) return;
    const size_t pos = snap & ~1;
    const auto* same = index_.find(tokens, *images, now, true, failed_);
    dispose_evicted(m, ring);
    if (same && same->tokens.size() >= pos) return;
    // Upper bound from the rounded device allocation including snapshots. Actual
    // disk allocation is independently enforced by each rank at write time.
    const size_t bytes = m.state_info(handle).bytes + 4096;
    if (!bytes || bytes > budget) return;
    while (index_.size() && (index_.size() >= 128 || index_.bytes() > budget - bytes)) {
      auto before = index_.size();
      index_.evict_oldest();
      if (index_.size() == before) return;
    }
    dispose_evicted(m, ring);
    if (!durable_state(m, ring, kCmdDurableSave, handle, id, snap, encoder_only)) return;
    if (!index_.publish(id, tokens, *images, pos, bytes, now, snap, encoder_only))
      durable_state(m, ring, kCmdDurableDrop, 0, id);
    dispose_evicted(m, ring);
  }

 private:
  void dispose_evicted(model::Model& m, comm::Ring& ring) {
    for (auto old : index_.take_evicted()) {
      failed_.erase(old);
      durable_state(m, ring, kCmdDurableDrop, 0, old);
    }
  }
  durable_image::Index index_;
  std::set<uint64_t> failed_;
};
}  // namespace srv
