#pragma once
#include <cstdint>

namespace model::ops {
#ifdef __CUDACC__
#define DS_SAMPLE_INLINE __host__ __device__ __forceinline__
#else
#define DS_SAMPLE_INLINE inline
#endif

// Gumbel-max needs an OPEN unit interval. The 24-bit midpoint expression
// rounds its largest bin to 1 in float32 (16777215 + 0.5 is a tie rounded up).
// log(-log(1)) then produces -infinity: an arbitrary vocabulary token wins,
// regardless of its logit. Clamp ONLY that endpoint; preserve every other
// bin bit-for-bit, including the minimum, and leave greedy sampling alone.
DS_SAMPLE_INLINE float sample_unit(uint64_t h) {
  const float u = ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
  return u < 1.0f ? u : 0x1.fffffep-1f;
}
#undef DS_SAMPLE_INLINE
}  // namespace model::ops
