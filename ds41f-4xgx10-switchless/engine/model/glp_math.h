// Spec-v1 Owen arithmetic, shared with the host regression (not an inference emulator).
#pragma once
#include <cmath>
#ifdef __CUDACC__
#define GLP_HD __host__ __device__ __forceinline__
#else
#define GLP_HD inline
#endif
namespace model { namespace detail {
constexpr int kGlpThreads = 320, kGlpVec = 8, kGlpPer = 16;
GLP_HD int glp_column(int tid, int j) {
  return (j / kGlpVec * kGlpThreads + tid) * kGlpVec + j % kGlpVec;
}
GLP_HD float glp_mul(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fmul_rn(a, b);
#else
  volatile float v = a * b; return v;
#endif
}
GLP_HD float glp_add(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fadd_rn(a, b);
#else
  volatile float v = a + b; return v;
#endif
}
GLP_HD float glp_fma(float a, float b, float c) {
#ifdef __CUDA_ARCH__
  return __fmaf_rn(a, b, c);
#else
  return std::fma(a, b, c);
#endif
}
// Owen tilelang/mhc.py post_streams: MUL then four FMAs, before BF16 storage.
GLP_HD float glp_fold(float write, const float* res, float post, const float* comb, int n) {
  float v = glp_mul(comb[n], res[0]);
  v = glp_fma(post, write, v);
  for (int m = 1; m < 4; ++m) v = glp_fma(comb[m * 4 + n], res[m], v);
  return v;
}
} }
#undef GLP_HD
