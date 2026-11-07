// GLP steering: spec v1, hook residual_stream_post_layer (the hook Owen's engine applies).
//
// A GLP file holds one unit direction per decoder layer. At the end of a layer, after its write has
// been folded into the hc streams, every stream is projected off that direction:
//     h[n] <- bf16(h[n] - alpha * (h[n] . d) * d),   n = 0 .. hc_mult - 1
// which is what models/deepseek_v4_1/glp.py's steer_post_layer does (read_glp -> GLPSteering ->
// tilelang_glp.glp_project: fold the layer's write with post/comb into the row's streams, project every
// stream, keep the result with comb the identity and post zero). Off unless WEIGHTLESS_STEER_PATH is
// set; a file the reader cannot apply exactly is refused at startup instead of serving unsteered.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "model/config.h"

namespace model {
namespace detail {

// The file's contents, validated: unit directions by layer (index kDim * L), the layers it steers and
// its alpha. spec_v1 only: rank-1 projective removal, one scalar alpha, hook residual_stream_post_layer.
struct GlpFile {
  std::vector<float> dir;     // [kLayers][kDim], zero rows for the layers the file does not steer
  std::vector<int> layers;    // the steered layers, ascending
  float alpha = 0.f;
  std::string hook;
};

// Owen's read_glp, fail-closed: another mode, spec version, rank or hook; a direction of another width,
// not finite or not unit norm; a tensor that is not direction.<layer>; a layer list or content digest
// that disagrees with the tensors.
GlpFile glp_read(const std::string& path, int num_layers, int hidden);

// What the model runs: the reader's rows and the env overrides (WEIGHTLESS_STEER_ALPHA,
// WEIGHTLESS_STEER_LAYERS). on == false when WEIGHTLESS_STEER_PATH is unset: no buffer, no kernel.
struct GlpCfg {
  bool on = false;
  float alpha = 0.f;
  std::vector<float> dir;  // [kLayers][kDim]
  bool layer[kLayers] = {};
};
GlpCfg glp_from_env();

// Ideal algebraic projection reference for reader tests (double dot accumulator).
// Not a bit-level CUDA oracle; tests/glp_math_test.cpp covers the Owen FP32 order.
void glp_project_row(const float* dir, float alpha, const uint16_t* in, int streams, int hidden, uint16_t* out);

}  // namespace detail
}  // namespace model

namespace model {
namespace ops {
// streams [rows][kHc][kDim] bf16 in place; pre [rows][4] optional: then yraw [rows][kDim] is rewritten as
// attn::hc_pre's collapse of the projected streams, bf16(((p0 y0 + p1 y1) + p2 y2) + p3 y3).
void glp_steer(uint16_t* streams, int rows, const float* dir, const float* alpha, const float* pre, uint16_t* yraw,
               cudaStream_t s);
// Active-only Owen fold from write [rows][kDim] BF16 and residual [rows][4][kDim]
// BF16 with post/comb FP32; then the same projection/collapse. Buffers must not alias streams.
void glp_fold_steer(uint16_t* streams, int rows, const float* dir, const float* alpha,
                     const uint16_t* write, const uint16_t* residual, const float* post, const float* comb,
                     const float* pre, uint16_t* yraw, cudaStream_t s);
}  // namespace ops
}  // namespace model
