// The GLP reader (glp.h): a GGUF v3 file of unit directions, checked the way Owen's engine checks it
// (vllm/models/deepseek_v4_1/glp.py's read_gguf/_read_glp) so that a file it accepts here is the file
// that engine applied, layer for layer, and a file it would refuse is refused here too.
#include "model/glp.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <map>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace model {
namespace detail {
namespace {

// ---- SHA-256 (FIPS 180-4), for glp.content_sha256 --------------------------------------------
struct Sha256 {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint64_t n = 0;
  uint8_t buf[64] = {};
  size_t fill = 0;

  static uint32_t rotr(uint32_t x, int k) { return (x >> k) | (x << (32 - k)); }

  void block(const uint8_t* p) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + mj;
      hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
  }

  void update(const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    n += len;
    while (len) {
      const size_t take = std::min(len, sizeof buf - fill);
      std::memcpy(buf + fill, p, take);
      fill += take, p += take, len -= take;
      if (fill == sizeof buf) block(buf), fill = 0;
    }
  }

  std::string hex() {
    const uint64_t bits = n * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t z = 0;
    while (fill != 56) update(&z, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = (uint8_t)(bits >> (56 - 8 * i));
    update(len, 8);
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
    return std::string(out, 64);
  }
};

// ---- a byte cursor over the file ---------------------------------------------------------------
struct Cursor {
  const std::string& path;
  const std::vector<uint8_t>& d;
  size_t o = 0;

  void need(size_t size) const {
    if (o + size > d.size())
      throw std::runtime_error(path + ": truncated GGUF at byte " + std::to_string(o));
  }
  template <class T>
  T take() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, d.data() + o, sizeof(T));
    o += sizeof(T);
    return v;  // the file is little-endian and so is every host this engine builds for
  }
  std::string str() {
    const uint64_t size = take<uint64_t>();
    need(size);
    std::string s(reinterpret_cast<const char*>(d.data() + o), size);
    o += size;
    return s;
  }
};

// GGUF metadata value types (GGUF_TYPE_*): what a key's value is. A tensor's type is a ggml_type instead,
// where F32 is 0 (ggml_type::GGML_TYPE_F32); the two numberings are different and a file's tensors are the
// former's F32 == the latter's 0.
enum : uint32_t { kU8 = 0, kI8 = 1, kU16 = 2, kI16 = 3, kU32 = 4, kI32 = 5, kF32 = 6, kBool = 7, kString = 8, kArray = 9, kU64 = 10, kI64 = 11, kF64 = 12 };
enum : uint32_t { kGgmlF32 = 0 };

struct Value {
  uint32_t kind = kU32;
  double num = 0;      // scalars
  bool b = false;
  std::string s;       // strings
  std::vector<Value> arr;
};

Value value(Cursor& c, uint32_t kind) {
  Value v;
  v.kind = kind;
  switch (kind) {
    case kU8: v.num = c.take<uint8_t>(); break;
    case kI8: v.num = c.take<int8_t>(); break;
    case kU16: v.num = c.take<uint16_t>(); break;
    case kI16: v.num = c.take<int16_t>(); break;
    case kU32: v.num = c.take<uint32_t>(); break;
    case kI32: v.num = c.take<int32_t>(); break;
    case kU64: v.num = (double)c.take<uint64_t>(); break;
    case kI64: v.num = (double)c.take<int64_t>(); break;
    case kF32: v.num = c.take<float>(); break;
    case kF64: v.num = c.take<double>(); break;
    case kBool: v.b = c.take<uint8_t>() != 0; break;
    case kString: v.s = c.str(); break;
    case kArray: {
      const uint32_t item = c.take<uint32_t>();
      const uint64_t count = c.take<uint64_t>();
      if (item == kArray) throw std::runtime_error("nested GGUF arrays are not supported");
      v.arr.reserve((size_t)count);
      for (uint64_t i = 0; i < count; ++i) v.arr.push_back(value(c, item));
      break;
    }
    default: throw std::runtime_error("unknown GGUF value type " + std::to_string(kind));
  }
  return v;
}

bool is_int(const Value& v) {
  return v.kind == kU8 || v.kind == kI8 || v.kind == kU16 || v.kind == kI16 || v.kind == kU32 || v.kind == kI32 ||
         v.kind == kU64 || v.kind == kI64;
}

std::vector<int> layer_list(const std::string& text, const std::string& path) {
  std::vector<int> out;
  size_t i = 0;
  while (i < text.size()) {
    size_t j = text.find(',', i);
    if (j == std::string::npos) j = text.size();
    std::string item = text.substr(i, j - i);
    const size_t b = item.find_first_not_of(" \t");
    const size_t e = item.find_last_not_of(" \t");
    if (b != std::string::npos) {
      item = item.substr(b, e - b + 1);
      char* end = nullptr;
      const long v = std::strtol(item.c_str(), &end, 10);
      if (!end || *end) throw std::runtime_error(path + ": layer list is not a comma list of layer ids: " + text);
      out.push_back((int)v);
    }
    i = j + 1;
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

GlpFile glp_read(const std::string& path, int num_layers, int hidden) {
  std::vector<uint8_t> d;
  {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error(path + ": cannot open the GLP file");
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size < 0) {
      std::fclose(f);
      throw std::runtime_error(path + ": cannot size the GLP file");
    }
    d.resize((size_t)size);
    const size_t got = std::fread(d.data(), 1, d.size(), f);
    std::fclose(f);
    if (got != d.size()) throw std::runtime_error(path + ": short read of the GLP file");
  }
  Cursor c{path, d};
  if (c.take<uint32_t>() != 0x46554747u) throw std::runtime_error(path + ": not a GGUF file");  // "GGUF"
  const uint32_t version = c.take<uint32_t>();
  if (version != 3) throw std::runtime_error(path + ": GGUF version " + std::to_string(version) + ", expected 3");
  const uint64_t tensor_count = c.take<uint64_t>();
  const uint64_t key_count = c.take<uint64_t>();

  std::map<std::string, Value> md;
  for (uint64_t i = 0; i < key_count; ++i) {
    const std::string key = c.str();
    const uint32_t kind = c.take<uint32_t>();
    if (md.count(key)) throw std::runtime_error(path + ": duplicate GGUF key " + key);
    md[key] = value(c, kind);
  }

  struct Info {
    std::string name;
    uint64_t count = 0;
    uint64_t offset = 0;
  };
  std::vector<Info> infos;
  infos.reserve((size_t)tensor_count);
  for (uint64_t i = 0; i < tensor_count; ++i) {
    Info in;
    in.name = c.str();
    const uint32_t ndim = c.take<uint32_t>();
    uint64_t dims[8] = {};
    if (ndim == 0 || ndim > 8) throw std::runtime_error(path + ": tensor " + in.name + " has ndim " + std::to_string(ndim));
    for (uint32_t k = 0; k < ndim; ++k) dims[k] = c.take<uint64_t>();
    const uint32_t kind = c.take<uint32_t>();
    in.offset = c.take<uint64_t>();
    if (kind != kGgmlF32)
      throw std::runtime_error(path + ": tensor " + in.name + " is GGML type " + std::to_string(kind) + ", not F32");
    if (ndim != 1) throw std::runtime_error(path + ": tensor " + in.name + " is not 1-D");
    in.count = dims[0];
    infos.push_back(in);
  }

  uint64_t alignment = 32;
  if (auto it = md.find("general.alignment"); it != md.end()) {
    if (!is_int(it->second) || it->second.num <= 0)
      throw std::runtime_error(path + ": general.alignment is not a positive integer");
    alignment = (uint64_t)it->second.num;
  }
  const uint64_t base = (c.o + alignment - 1) / alignment * alignment;

  auto meta_str = [&](const char* key) -> const std::string* {
    auto it = md.find(key);
    if (it == md.end() || it->second.kind != kString) return nullptr;
    return &it->second.s;
  };
  // The hook and spec this engine implements: residual_stream_post_layer, spec v1 (Owen's HOOK_VERSIONS).
  if (meta_str("glp.mode") == nullptr || *meta_str("glp.mode") != "project")
    throw std::runtime_error(path + ": glp.mode is not 'project'; this runtime projects and never adds a control vector");
  auto spec = md.find("glp.spec_version");
  if (spec == md.end() || !is_int(spec->second) || (int)spec->second.num != 1)
    throw std::runtime_error(path + ": glp.spec_version is not 1 (spec v1, hook residual_stream_post_layer)");
  if (auto r = md.find("glp.rank"); r != md.end() && (!is_int(r->second) || (int)r->second.num != 1))
    throw std::runtime_error(path + ": glp.rank is not 1; this reader applies rank 1");
  for (const char* key : {"glp.dir_scales", "glp.layer_scales"})
    if (md.count(key))
      throw std::runtime_error(path + ": " + key + " is not supported; this reader applies unit directions with scalar glp.alpha_default");
  if (meta_str("glp.hook_point") == nullptr || *meta_str("glp.hook_point") != "residual_stream_post_layer")
    throw std::runtime_error(path + ": glp.hook_point is not residual_stream_post_layer; spec v3 hooks are not applied by this runtime");
  if (meta_str("glp.derived_at") && *meta_str("glp.derived_at") != "residual_stream_post_layer")
    std::fprintf(stderr, "glp: warning: %s: transferred vector, derived at %s and applied at residual_stream_post_layer\n",
                 path.c_str(), meta_str("glp.derived_at")->c_str());

  float alpha = 0.f;
  bool have_alpha = false;
  if (auto it = md.find("glp.alpha_default"); it != md.end()) {
    if (it->second.kind != kF32 || !std::isfinite(it->second.num))
      throw std::runtime_error(path + ": glp.alpha_default is not a finite float");
    alpha = (float)it->second.num, have_alpha = true;
  }

  GlpFile out;
  out.dir.assign((size_t)num_layers * hidden, 0.f);
  out.hook = "residual_stream_post_layer";
  uint64_t end = base;
  std::vector<int> layers;
  for (const Info& in : infos) {
    const std::string prefix = "direction.";
    if (in.name.compare(0, prefix.size(), prefix) != 0)
      throw std::runtime_error(path + ": tensor " + in.name + " is not a direction.<layer>");
    const std::string num = in.name.substr(prefix.size());
    // direction.<layer> exactly: digits only, no sign, space or prefix (Owen's _DIRECTION.fullmatch)
    if (num.empty() || num.find_first_not_of("0123456789") != std::string::npos)
      throw std::runtime_error(path + ": tensor " + in.name + " is not a direction.<layer>");
    const long layer = std::strtol(num.c_str(), nullptr, 10);
    if (layer < 1 || layer >= num_layers)
      throw std::runtime_error(path + ": " + in.name + " is outside layers 1.." + std::to_string(num_layers - 1));
    if (in.count != (uint64_t)hidden)
      throw std::runtime_error(path + ": " + in.name + " has width " + std::to_string(in.count) + ", not " + std::to_string(hidden));
    const uint64_t start = base + in.offset;
    if (start % alignment || start + 4 * in.count > d.size())
      throw std::runtime_error(path + ": tensor " + in.name + " lies outside the file");
    const float* v = reinterpret_cast<const float*>(d.data() + start);
    double norm2 = 0;
    for (uint64_t i = 0; i < in.count; ++i) {
      if (!std::isfinite(v[i])) throw std::runtime_error(path + ": " + in.name + " is not finite");
      norm2 += (double)v[i] * v[i];
    }
    const double norm = std::sqrt(norm2);
    if (std::fabs(norm - 1.0) > 1e-3)
      throw std::runtime_error(path + ": " + in.name + " has norm " + std::to_string(norm) + ", not 1");
    for (uint64_t i = 0; i < in.count; ++i) out.dir[(size_t)layer * hidden + i] = (float)((double)v[i] / norm);
    layers.push_back((int)layer);
    end = std::max(end, start + 4 * in.count);
  }
  if (layers.empty()) throw std::runtime_error(path + ": no direction tensors");
  std::sort(layers.begin(), layers.end());
  if (std::adjacent_find(layers.begin(), layers.end()) != layers.end())
    throw std::runtime_error(path + ": duplicate direction tensor for a layer");

  if (meta_str("glp.layer_ids_zero_based")) {
    const std::vector<int> declared = layer_list(*meta_str("glp.layer_ids_zero_based"), path);
    if (declared != layers)
      throw std::runtime_error(path + ": glp.layer_ids_zero_based disagrees with the direction tensors");
  }
  if (meta_str("glp.content_sha256")) {
    const std::string& want = *meta_str("glp.content_sha256");
    if (want.empty()) throw std::runtime_error(path + ": glp.content_sha256 is empty");
    Sha256 sha;
    for (int layer : layers) {
      const Info* in = nullptr;
      for (const Info& i : infos)
        if (i.name == "direction." + std::to_string(layer)) {
          in = &i;
          break;
        }
      sha.update(d.data() + base + in->offset, (size_t)4 * in->count);
    }
    const std::string got = sha.hex();
    std::string lower = want;
    for (char& ch : lower) ch = (char)std::tolower((unsigned char)ch);
    if (got != lower) throw std::runtime_error(path + ": glp.content_sha256 disagrees with the direction tensors (" + got + ")");
  }
  if ((uint64_t)d.size() - end >= alignment)
    throw std::runtime_error(path + ": " + std::to_string((uint64_t)d.size() - end) + " unexpected bytes after the last tensor");

  out.layers = layers;
  if (!have_alpha) {
    if (const char* a = std::getenv("WEIGHTLESS_STEER_ALPHA")) {
      try {
        alpha = std::stof(a);
      } catch (const std::exception&) {
        throw std::runtime_error(std::string("WEIGHTLESS_STEER_ALPHA is ") + a);
      }
      if (!std::isfinite(alpha)) throw std::runtime_error(std::string("WEIGHTLESS_STEER_ALPHA is ") + a);
      have_alpha = true;
    }
  }
  if (!have_alpha) throw std::runtime_error(path + ": no glp.alpha_default and no WEIGHTLESS_STEER_ALPHA");
  out.alpha = alpha;
  return out;
}

GlpCfg glp_from_env() {
  GlpCfg cfg;
  const char* path = std::getenv("WEIGHTLESS_STEER_PATH");
  if (!path) return cfg;  // off: the forward is unchanged
  GlpFile f = glp_read(path, kLayers, kDim);
  if (const char* a = std::getenv("WEIGHTLESS_STEER_ALPHA")) {
    try {
      f.alpha = std::stof(a);
    } catch (const std::exception&) {
      throw std::runtime_error(std::string("WEIGHTLESS_STEER_ALPHA is ") + a);
    }
    if (!std::isfinite(f.alpha)) throw std::runtime_error(std::string("WEIGHTLESS_STEER_ALPHA is ") + a);
  }
  std::vector<int> layers = f.layers;
  if (const char* l = std::getenv("WEIGHTLESS_STEER_LAYERS")) {
    layers = layer_list(l, "WEIGHTLESS_STEER_LAYERS");
    if (layers.empty()) throw std::runtime_error(std::string("WEIGHTLESS_STEER_LAYERS is ") + l);
    for (int layer : layers)
      if (std::find(f.layers.begin(), f.layers.end(), layer) == f.layers.end())
        throw std::runtime_error(std::string("WEIGHTLESS_STEER_LAYERS names layer ") + std::to_string(layer) +
                                 " without a direction in " + path);
  }
  cfg.on = true;
  cfg.alpha = f.alpha;
  cfg.dir = std::move(f.dir);
  for (int layer : layers) cfg.layer[layer] = true;
  return cfg;
}

void glp_project_row(const float* dir, float alpha, const uint16_t* in, int streams, int hidden, uint16_t* out) {
  auto bf2f = [](uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  auto f2bf = [](float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)0x7fc0u;  // NaN stays NaN
    u = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;           // __float2bfloat16_rn: nearest, ties to even
    return (uint16_t)u;
  };
  for (int n = 0; n < streams; ++n) {
    const uint16_t* src = in + (size_t)n * hidden;
    double dot = 0;
    for (int i = 0; i < hidden; ++i) dot += (double)bf2f(src[i]) * (double)dir[i];
    const float scale = (float)((double)alpha * dot);
    for (int i = 0; i < hidden; ++i) out[(size_t)n * hidden + i] = f2bf(bf2f(src[i]) - scale * dir[i]);
  }
}

}  // namespace detail
}  // namespace model
