// The GLP steering kernel (glp.h): a layer's hc streams projected off one unit direction.
//
// Owen's tilelang kernel folds the layer's write into the row's streams and projects them in one block
// per row (glp.py's glp_steer); the fold's result is what this kernel reads, because the decode and
// prefill paths here materialise the streams at the boundary (ops::allreduce_hc, dsk::hc_boundary_p).
// One CTA per row: 4 streams x 5120 columns, each thread 20 columns of every stream, the four dots
// reduced in a fixed order (each thread's columns in sequence, then the warp's lanes, then the warps), so
// a row's result never depends on the batch.
#include <cuda_runtime.h>

#include "attn/attn_common.cuh"
#include "common/check.h"
#include "model/glp.h"

namespace model {
namespace ops {
namespace {

constexpr int kThreads = 256;                     // 5120 / 256 = 20 columns per thread
constexpr int kPer = model::kDim / kThreads;
constexpr int kWarps = kThreads / 32;

__global__ void __launch_bounds__(kThreads) k_glp_steer(uint16_t* streams, int rows, const float* __restrict__ dir,
                                                        const float* __restrict__ alpha, const float* __restrict__ pre,
                                                        uint16_t* __restrict__ yraw) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint16_t* row = streams + (size_t)r * model::kHcDim;
  float h[model::kHc][kPer];
  float d[kPer];
  float acc[model::kHc] = {};
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int c = tid + kThreads * j;
    d[j] = dir[c];
#pragma unroll
    for (int n = 0; n < model::kHc; ++n) {
      h[n][j] = attn::bf2f(row[(size_t)n * model::kDim + c]);
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
    const int c = tid + kThreads * j;
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
  k_glp_steer<<<rows, kThreads, 0, s>>>(streams, rows, dir, alpha, pre, yraw);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ops
}  // namespace model
