#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace srv::image_prefix {

constexpr uint32_t kFlagCache = 32;  // matches frontend/src/protocol.rs (16 is fe::kFlagPrediction)
constexpr uint32_t kFlagDurable = 64;  // explicit ordinary frontend disk opt-in
constexpr size_t kMaxEntries = 4;
constexpr size_t kMaxRowBytes = 128u << 20;
constexpr double kIdleSeconds = 300;

// Both span tables have already been validated against their own prompt and
// row buffer. Divergent image positions, lengths, or row bytes cap reuse
// before the first divergent image. In particular, a hash is never identity.
template <class Spans, class Rows>
size_t common(size_t same_tokens, size_t dim, const Spans& old_spans, const Rows& old_rows,
              const Spans& new_spans, const Rows& new_rows) {
  size_t offset = 0;
  for (size_t i = 0; i < std::max(old_spans.size(), new_spans.size()); ++i) {
    const size_t old_start = i < old_spans.size() ? old_spans[i].start : same_tokens;
    const size_t new_start = i < new_spans.size() ? new_spans[i].start : same_tokens;
    const size_t start = std::min(old_start, new_start);
    if (start >= same_tokens) break;
    if (old_start != new_start || old_spans[i].rows != new_spans[i].rows) return start;
    const size_t words = size_t(old_spans[i].rows) * dim;
    if (std::memcmp(old_rows.data() + offset, new_rows.data() + offset, words * sizeof(uint16_t)) != 0)
      return start;
    offset += words;
  }
  return same_tokens;
}

// Complete reused spans need no rank transfer. Send the whole first remaining
// span if the engine's restart point falls within it.
template <class Spans>
size_t suffix(const Spans& spans, size_t restart, size_t& skipped_rows) {
  size_t first = 0;
  skipped_rows = 0;
  while (first < spans.size() && size_t(spans[first].start) + spans[first].rows <= restart)
    skipped_rows += spans[first++].rows;
  return first;
}

inline bool within_budget(size_t entries, size_t bytes) {
  return entries <= kMaxEntries && bytes <= kMaxRowBytes;
}

inline bool expired(double now, double last_used) {
  return now - last_used >= kIdleSeconds;
}

}  // namespace srv::image_prefix
