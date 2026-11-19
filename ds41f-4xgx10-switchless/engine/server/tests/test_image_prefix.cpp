// Compile against the actual production header from the applied candidate patch.
#include "server/image_prefix.h"

#include <cassert>
#include <vector>

struct Span {
  uint32_t start;
  uint32_t rows;
  uint64_t hash;
};

int main() {
  namespace image = srv::image_prefix;
  const std::vector<Span> before{{8, 3, 1}, {18, 4, 2}};
  const std::vector<uint16_t> rows{11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28};
  constexpr size_t dim = 2;

  assert(image::common(25, dim, before, rows, before, rows) == 25);
  assert(image::common(20, dim, before, rows, before, rows) == 20);
  assert(image::common(7, dim, before, rows, before, rows) == 7);

  auto altered = rows;
  altered[5] ^= 1;  // exactly one BF16 word differs, with identical spans and hashes
  assert(image::common(25, dim, before, rows, before, altered) == 8);
  altered = rows;
  altered.back() ^= 1;
  assert(image::common(25, dim, before, rows, before, altered) == 18);
  assert(image::common(15, dim, before, rows, before, altered) == 15);
  // Divergence later in an image still retracts a prefix ending partway through it.
  assert(image::common(20, dim, before, rows, before, altered) == 18);

  auto spans = before;
  spans.front().hash ^= 1;
  assert(image::common(25, dim, before, rows, spans, rows) == 25);
  spans.front().start++;
  assert(image::common(25, dim, before, rows, spans, rows) == 8);
  spans = before;
  spans.back().rows++;
  assert(image::common(25, dim, before, rows, spans, rows) == 18);
  spans = before;
  spans.pop_back();
  assert(image::common(25, dim, before, rows, spans, rows) == 18);
  assert(image::common(25, dim, spans, rows, before, rows) == 18);

  size_t skipped = 999;
  assert(image::suffix(before, 0, skipped) == 0 && skipped == 0);
  assert(image::suffix(before, 10, skipped) == 0 && skipped == 0);
  assert(image::suffix(before, 11, skipped) == 1 && skipped == 3);
  assert(rows[skipped * dim] == 21);
  assert(image::suffix(before, 20, skipped) == 1 && skipped == 3);
  assert(image::suffix(before, 22, skipped) == 2 && skipped == 7);
  assert(skipped * dim == rows.size());
  assert(image::suffix(before, 23, skipped) == 2 && skipped == 7);

  assert(image::within_budget(image::kMaxEntries, image::kMaxRowBytes));
  assert(!image::within_budget(image::kMaxEntries + 1, 0));
  assert(!image::within_budget(1, image::kMaxRowBytes + 1));
  assert(!image::expired(310, 11));
  assert(image::expired(311, 11));
}
