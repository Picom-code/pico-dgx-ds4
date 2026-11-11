// No GPU/model required. Exercise the exact helper used by sample.cu, not a
// reimplementation of the fix. Build: c++ -std=c++20 -O3 -Iengine this.cpp.
#include "model/sample_rng.h"
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

static void check(bool ok, const char* what) {
  if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
static float legacy(uint64_t h) {
  return ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
}
static float gumbel(float u) { return -std::log(-std::log(u)); }
static uint64_t mix64(uint64_t x) {
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 29; x *= 0x94D049BB133111EBull;
  return x ^ (x >> 32);
}
int main() {
  int old_closed = 0, changed = 0;
  for (uint64_t bin = 0; bin < (1u << 24); ++bin) {
    const uint64_t h = bin << 40;
    const float before = legacy(h), after = model::ops::sample_unit(h);
    old_closed += !(before > 0.f && before < 1.f);
    changed += std::bit_cast<uint32_t>(before) != std::bit_cast<uint32_t>(after);
    check(after > 0.f && after < 1.f, "fixed uniform outside open interval");
    // Lower 40 hash bits never affect the result, including both endpoints.
    check(after == model::ops::sample_unit(h | ((1ull << 40) - 1)), "lower hash bits changed uniform");
  }
  check(old_closed == 1, "legacy endpoint reproduction");
  check(changed == 1, "unaffected bins must remain bit-identical");
  const uint64_t h = 0xffffff0000000000ull;
  check(legacy(h) == 1.f, "legacy float32 midpoint rounds to one");
  check(std::isinf(gumbel(legacy(h))), "legacy Gumbel must reproduce infinity");
  check(std::isfinite(gumbel(model::ops::sample_unit(h))), "fixed Gumbel must be finite");
  const float unlikely_logit = -1000.f, likely_score = 10.f + gumbel(0.5f);
  check(unlikely_logit + gumbel(legacy(h)) > likely_score, "legacy tail token wins regardless of logit");
  check(unlikely_logit + gumbel(model::ops::sample_unit(h)) < likely_score, "fixed tail token cannot force a win");
  // Deterministic seed/position/id witness from the same hash as sample.cu.
  // The exact inverse-constructed seed is calculated/checked in the evidence harness.
  const uint64_t seed = 129;
  int infinite_candidates = 0;
  for (int id = 0; id < 129280; ++id) {
    const uint64_t r = mix64(mix64(seed) + (uint64_t)id * 0xD1B54A32D192ED03ull);
    infinite_candidates += !std::isfinite(gumbel(legacy(r)));
    check(std::isfinite(gumbel(model::ops::sample_unit(r))), "fixed full-vocabulary Gumbel is nonfinite");
  }
  std::printf("PASS: all 16777216 bins open; exactly 1 legacy closed bin and 1 changed bin; lower bits invariant; legacy impossible-token win reproduced; fixed rejected; full vocabulary finite (seed129 legacy infinities=%d)\n", infinite_candidates);
}
