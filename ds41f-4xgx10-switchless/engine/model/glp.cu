// The GLP steering kernel (glp.h): a layer's hc streams projected off one unit direction.
//
// The recorded Owen v1 image uses BF16 post_streams (MUL then four FMAs), then
// 320 threads per stream with two contiguous groups of 8 columns per thread.
// The active fold is reconstructed from the reduced BF16 write and original
// residual, not Pico's differently associated baseline fold. OFF/alpha0 retain
// the baseline path. See glp_math.h and tests/glp_math_test.cpp.
#include <cuda_runtime.h>

#include "attn/attn_common.cuh"
#include "common/check.h"
#include "model/glp.h"
#include "model/glp_math.h"

namespace model {
namespace ops {
namespace {

constexpr int kThreads = detail::kGlpThreads;
constexpr int kPer = detail::kGlpPer;
static_assert(kThreads * kPer == model::kDim, "Owen v1 projection layout");
constexpr int kWarps = kThreads / 32;

__global__ void __launch_bounds__(kThreads) k_glp_steer(uint16_t* streams, int rows, const float* __restrict__ dir,
                                                        const float* __restrict__ alpha, const float* __restrict__ pre,
                                                        uint16_t* __restrict__ yraw, const uint16_t* write,
                                                        const uint16_t* residual, const float* post, const float* comb) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint16_t* row = streams + (size_t)r * model::kHcDim;
  float h[model::kHc][kPer];
  float d[kPer];
  float acc[model::kHc] = {};
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int c = detail::glp_column(tid, j);
    d[j] = dir[c];
#pragma unroll
    for (int n = 0; n < model::kHc; ++n) {
      if (residual) {
        float rv[model::kHc];
#pragma unroll
        for (int m = 0; m < model::kHc; ++m)
          rv[m] = attn::bf2f(residual[((size_t)r * model::kHc + m) * model::kDim + c]);
        const float folded = detail::glp_fold(attn::bf2f(write[(size_t)r * model::kDim + c]), rv,
                                              post[r * model::kHc + n], comb + r * model::kHc * model::kHc, n);
        h[n][j] = attn::bf2f(attn::f2bf(folded));
      } else {
        h[n][j] = attn::bf2f(row[(size_t)n * model::kDim + c]);
      }
      acc[n] = __fmaf_rn(h[n][j], d[j], acc[n]);
    }
  }
  __shared__ float red[model::kHc][kWarps];
  __shared__ float scale[model::kHc];
#pragma unroll
  for (int n = 0; n < model::kHc; ++n) {
#pragma unroll
    for (int off = 16; off; off >>= 1) acc[n] = __fadd_rn(acc[n], __shfl_xor_sync(0xffffffffu, acc[n], off));
    if (lane == 0) red[n][warp] = acc[n];
  }
  __syncthreads();
  if (tid < model::kHc) {
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) tot = __fadd_rn(tot, red[tid][w]);
    scale[tid] = __fmul_rn(alpha[0], tot);  // alpha * (h . d)
  }
  __syncthreads();
  const float* pr = pre ? pre + (size_t)r * model::kHc : nullptr;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int c = detail::glp_column(tid, j);
    uint16_t y[model::kHc];
#pragma unroll
    for (int n = 0; n < model::kHc; ++n) {
      // the stream the next sublayer reads: bf16(h - alpha (h . d) d)
      y[n] = attn::f2bf(__fsub_rn(h[n][j], __fmul_rn(scale[n], d[j])));
      row[(size_t)n * model::kDim + c] = y[n];
    }
    if (pr) {  // attn::hc_pre's collapse of the projected streams, read back at their stored bf16 values
      float v = __fadd_rn(__fmul_rn(pr[0], attn::bf2f(y[0])), __fmul_rn(pr[1], attn::bf2f(y[1])));
      v = __fadd_rn(v, __fmul_rn(pr[2], attn::bf2f(y[2])));
      v = __fadd_rn(v, __fmul_rn(pr[3], attn::bf2f(y[3])));
      yraw[(size_t)r * model::kDim + c] = attn::f2bf(v);
    }
  }
}

}  // namespace

void glp_steer(uint16_t* streams, int rows, const float* dir, const float* alpha, const float* pre, uint16_t* yraw,
               cudaStream_t s) {
  if (rows < 1) return;
  k_glp_steer<<<rows, kThreads, 0, s>>>(streams, rows, dir, alpha, pre, yraw, nullptr, nullptr, nullptr, nullptr);
  CUDA_CHECK(cudaGetLastError());
}

void glp_fold_steer(uint16_t* streams, int rows, const float* dir, const float* alpha,
                     const uint16_t* write, const uint16_t* residual, const float* post, const float* comb,
                     const float* pre, uint16_t* yraw, cudaStream_t s) {
  if (rows < 1) return;
  k_glp_steer<<<rows, kThreads, 0, s>>>(streams, rows, dir, alpha, pre, yraw, write, residual, post, comb);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ops
}  // namespace model
