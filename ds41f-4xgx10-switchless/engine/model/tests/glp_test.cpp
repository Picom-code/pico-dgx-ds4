// GLP reader and projection: host only, no GPU. With a file argument it checks the real GLP-39 export;
// without one it checks synthetic files (a valid one and one refusal per contract) and the arithmetic.
//
//   glp_test [path.gguf]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/glp.h"

using model::detail::GlpFile;
using model::detail::glp_project_row;
using model::detail::glp_read;

namespace {

int failures = 0;
#define CHECK(cond, ...)                               \
  do {                                                 \
    if (!(cond)) {                                     \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                        \
      std::printf("\n");                               \
      ++failures;                                      \
    }                                                  \
  } while (0)

float bf2f(uint16_t v) {
  const uint32_t u = (uint32_t)v << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// ---- a minimal GGUF v3 writer, for the contract's refusals --------------------------------------
struct Writer {
  std::vector<uint8_t> d;
  void u32(uint32_t v) { d.insert(d.end(), (uint8_t*)&v, (uint8_t*)&v + 4); }
  void u64(uint64_t v) { d.insert(d.end(), (uint8_t*)&v, (uint8_t*)&v + 8); }
  void raw(const void* p, size_t n) { d.insert(d.end(), (const uint8_t*)p, (const uint8_t*)p + n); }
  void str(const std::string& s) {
    u64(s.size());
    raw(s.data(), s.size());
  }
  void kv_str(const std::string& k, const std::string& v) { str(k), u32(8), str(v); }
  void kv_i32(const std::string& k, int32_t v) { str(k), u32(5), raw(&v, 4); }
  void kv_f32(const std::string& k, float v) { str(k), u32(6), raw(&v, 4); }
};

struct Synth {
  int layers = 3, hidden = 4;
  float alpha = 0.75f;
  int spec = 1;
  float norm2 = 1.f;  // direction.2's norm
  std::string mode = "project", hook = "residual_stream_post_layer";
  bool digest = false, extra_tensor = false, layer_list = true;
  const char* path = "/tmp/glp_synth.gguf";

  void build() const {
    Writer w;
    w.u32(0x46554747u), w.u32(3);  // "GGUF", version 3
    const int tensors = layers + (extra_tensor ? 1 : 0);
    const int keys = 4 + (digest ? 1 : 0) + (layer_list ? 1 : 0);
    w.u64(tensors), w.u64(keys);
    w.kv_str("glp.mode", mode);
    w.kv_i32("glp.spec_version", spec);
    w.kv_str("glp.hook_point", hook);
    w.kv_f32("glp.alpha_default", alpha);
    if (layer_list) {
      std::string ids;
      for (int L = 1; L <= layers; ++L) ids += (L > 1 ? "," : "") + std::to_string(L + (extra_tensor ? 1 : 0));
      w.kv_str("glp.layer_ids_zero_based", ids);
    }
    if (digest) w.kv_str("glp.content_sha256", std::string(64, '0'));  // never the tensors' digest
    struct T {
      std::string name;
      std::vector<float> v;
    };
    std::vector<T> ts;
    for (int L = 1; L <= layers; ++L) {
      std::vector<float> v(hidden, 0.f);
      v[L % hidden] = L == 2 ? norm2 : 1.f;
      ts.push_back({"direction." + std::to_string(L + (extra_tensor ? 1 : 0)), v});
    }
    if (extra_tensor) ts.push_back({"direction.0", std::vector<float>(hidden, 0.f)});
    // each tensor's data starts at a multiple of general.alignment (32), as GGUF requires
    std::vector<uint64_t> offs;
    uint64_t off = 0;
    for (const T& t : ts) {
      w.str(t.name), w.u32(1), w.u64(t.v.size()), w.u32(0), w.u64(off);  // GGML_TYPE_F32 == 0
      offs.push_back(off);
      off += 4 * t.v.size();
      off = (off + 31) / 32 * 32;
    }
    while (w.d.size() % 32) w.d.push_back(0);
    const size_t base = w.d.size();
    for (size_t i = 0; i < ts.size(); ++i) {
      while (w.d.size() < base + offs[i]) w.d.push_back(0);
      w.raw(ts[i].v.data(), 4 * ts[i].v.size());
    }
    std::FILE* f = std::fopen(path, "wb");
    if (!f) throw std::runtime_error("cannot write the synthetic file");
    std::fwrite(w.d.data(), 1, w.d.size(), f);
    std::fclose(f);
  }
};

bool refuses(const Synth& s, const char* what) {
  s.build();
  try {
    const GlpFile f = glp_read(s.path, s.layers + 1, s.hidden);
    std::printf("FAIL: %s was accepted (%zu layers)\n", what, f.layers.size());
    ++failures;
    return false;
  } catch (const std::exception&) {
    return true;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <glp.gguf>\n", argv[0]);
    return 2;
  }
  // ---- a synthetic file's contract -------------------------------------------------------------
  Synth ok;
  ok.path = "/tmp/glp_ok.gguf";
  ok.build();
  {
    const GlpFile f = glp_read(ok.path, ok.layers + 1, ok.hidden);
    CHECK(f.layers.size() == 3, "synthetic layers %zu", f.layers.size());
    CHECK(std::fabs(f.alpha - 0.75f) < 1e-6f, "synthetic alpha %g", f.alpha);
    CHECK(f.dir[(size_t)1 * ok.hidden + 1] == 1.f, "direction.1 is not the unit row");
    CHECK(f.dir[0] == 0.f, "layer 0 of the table is not zero");
  }
  {
    Synth s;
    s.path = "/tmp/glp_bad_spec.gguf", s.spec = 3;
    refuses(s, "spec_version 3");
    s = Synth(), s.path = "/tmp/glp_bad_hook.gguf", s.hook = "hc_fold_pre_engram";
    refuses(s, "the spec-v3 hook");
    s = Synth(), s.path = "/tmp/glp_bad_mode.gguf", s.mode = "add";
    refuses(s, "mode add");
    s = Synth(), s.path = "/tmp/glp_bad_norm.gguf", s.norm2 = 2.f;
    refuses(s, "a direction of norm 2");
    s = Synth(), s.path = "/tmp/glp_bad_layer.gguf", s.extra_tensor = true;
    refuses(s, "direction.0");
    s = Synth(), s.path = "/tmp/glp_bad_digest.gguf", s.digest = true;
    refuses(s, "a digest that disagrees with the tensors");
  }

  // ---- the real export -------------------------------------------------------------------------
  GlpFile f = glp_read(argv[1], 40, 5120);
  CHECK(f.layers.size() == 39, "layers %zu", f.layers.size());
  CHECK(!f.layers.empty() && f.layers.front() == 1 && f.layers.back() == 39, "layers %d..%d", f.layers.front(),
        f.layers.back());
  CHECK(std::fabs(f.alpha - 0.75f) < 1e-6f, "alpha %g", f.alpha);
  CHECK(f.hook == "residual_stream_post_layer", "hook %s", f.hook.c_str());
  double worst = 0;
  for (int L : f.layers) {
    double n = 0;
    for (int i = 0; i < 5120; ++i) n += (double)f.dir[(size_t)L * 5120 + i] * f.dir[(size_t)L * 5120 + i];
    worst = std::max(worst, std::fabs(std::sqrt(n) - 1.0));
  }
  CHECK(worst < 1e-4, "worst |norm - 1| %g", worst);
  CHECK(f.dir[0] == 0.f, "layer 0 has no direction");

  // alpha 0 is the identity, bit for bit; a projection leaves the full 5120-wide row orthogonal to d.
  std::vector<uint16_t> row((size_t)4 * 5120), out((size_t)4 * 5120);
  uint32_t x = 12345u;
  for (size_t i = 0; i < row.size(); ++i) {  // finite bf16 values in +-1..2 with random signs and mantissas
    x = x * 1664525u + 1013904223u;
    row[i] = (uint16_t)(((x >> 31) & 1u) << 15 | (0x7fu << 7) | ((x >> 16) & 0x7fu));
  }
  const std::vector<uint16_t> in = row;
  glp_project_row(f.dir.data() + 5120, 0.f, row.data(), 4, 5120, out.data());
  CHECK(std::memcmp(in.data(), out.data(), in.size() * 2) == 0, "alpha 0 changed the row");
  glp_project_row(f.dir.data() + 5120, 0.75f, row.data(), 4, 5120, out.data());
  const double before0 = [&] {
    double d0 = 0;
    for (int i = 0; i < 5120; ++i) d0 += (double)bf2f(in[i]) * f.dir[5120 + i];
    return d0;
  }();
  for (int n = 0; n < 4; ++n) {
    double d0 = 0, sq = 0;
    for (int i = 0; i < 5120; ++i) {
      d0 += (double)bf2f(out[(size_t)n * 5120 + i]) * f.dir[5120 + i];
      sq += (double)bf2f(out[(size_t)n * 5120 + i]) * bf2f(out[(size_t)n * 5120 + i]);
    }
    // the residue is bf16 rounding of the projected row, so |d . row| is bounded by the row's scale
    CHECK(std::fabs(d0) < 1e-2 * std::sqrt(sq) + 1e-6, "stream %d still holds %.3g of d (|row| %.3g, before %.3g)",
          n, d0, std::sqrt(sq), before0);
  }

  // ---- the env contract: off when unset, and the overrides refused when malformed -----------------
  using model::detail::glp_from_env;
  {
    ::unsetenv("WEIGHTLESS_STEER_PATH"), ::unsetenv("WEIGHTLESS_STEER_ALPHA"), ::unsetenv("WEIGHTLESS_STEER_LAYERS");
    const auto off = glp_from_env();
    CHECK(!off.on && off.dir.empty() && off.alpha == 0.f, "an unset WEIGHTLESS_STEER_PATH must be off");
    int marked = 0;
    for (int L = 0; L < 40; ++L) marked += off.layer[L];
    CHECK(marked == 0, "off but %d layers marked", marked);

    ::setenv("WEIGHTLESS_STEER_PATH", argv[1], 1);
    const auto on = glp_from_env();
    int n = 0;
    for (int L = 0; L < 40; ++L) n += on.layer[L];
    CHECK(on.on && n == 39 && !on.layer[0] && on.layer[1] && on.layer[39], "path: on %d, %d layers", (int)on.on, n);
    CHECK(std::fabs(on.alpha - 0.75f) < 1e-6f, "path alpha %g", on.alpha);

    ::setenv("WEIGHTLESS_STEER_ALPHA", "0", 1);
    CHECK(glp_from_env().alpha == 0.f, "WEIGHTLESS_STEER_ALPHA=0 not applied");
    ::setenv("WEIGHTLESS_STEER_ALPHA", "nought", 1);
    bool threw = false;
    try { glp_from_env(); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "a malformed WEIGHTLESS_STEER_ALPHA was accepted");
    ::unsetenv("WEIGHTLESS_STEER_ALPHA");

    ::setenv("WEIGHTLESS_STEER_LAYERS", " 1 , 2 ", 1);
    const auto few = glp_from_env();
    int m = 0;
    for (int L = 0; L < 40; ++L) m += few.layer[L];
    CHECK(m == 2 && few.layer[1] && few.layer[2], "WEIGHTLESS_STEER_LAYERS 1,2 gave %d layers", m);
    ::setenv("WEIGHTLESS_STEER_LAYERS", "1,0", 1);
    threw = false;
    try { glp_from_env(); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "a layer without a direction was accepted");
    ::setenv("WEIGHTLESS_STEER_LAYERS", "1,99", 1);
    threw = false;
    try { glp_from_env(); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "layer 99 was accepted");
    ::unsetenv("WEIGHTLESS_STEER_LAYERS");

    ::setenv("WEIGHTLESS_STEER_PATH", "/nonexistent/glp.gguf", 1);
    threw = false;
    try { glp_from_env(); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "a missing GLP file was accepted");
    ::unsetenv("WEIGHTLESS_STEER_PATH");
  }

  std::printf("real file: %zu directions, layers %d..%d, alpha %g, worst |norm-1| %.2e; 6 synthetic refusals;"
              " env contract checked\n",
              f.layers.size(), f.layers.front(), f.layers.back(), f.alpha, worst);
  std::printf(failures ? "FAILED: %d\n" : "ok\n", failures);
  return failures ? 1 : 0;
}
