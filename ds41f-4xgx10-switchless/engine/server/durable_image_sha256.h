#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace srv::durable_image {

using Digest = std::array<uint8_t, 32>;

// SHA-256 compression and padding (FIPS 180-4). Keep the cache format independent of an optional
// system crypto library. This checksum detects corruption and gives image rows a content identity.
inline Digest sha256(const void* bytes, size_t size) {
  constexpr std::array<uint32_t, 64> k = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  };
  std::array<uint32_t, 8> state = {
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  };
  const auto compress = [&](const uint8_t* block) {
    uint32_t words[64];
    for (size_t i = 0; i < 16; ++i) {
      const uint8_t* p = block + i * 4;
      words[i] = uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | uint32_t(p[3]);
    }
    for (size_t i = 16; i < 64; ++i) {
      const uint32_t a = words[i - 15], b = words[i - 2];
      const uint32_t s0 = std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3);
      const uint32_t s1 = std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = state;
    for (size_t i = 0; i < 64; ++i) {
      const uint32_t s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const uint32_t choose = (e & f) ^ (~e & g);
      const uint32_t t1 = h + s1 + choose + k[i] + words[i];
      const uint32_t s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + s0 + majority;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  };

  const auto* data = static_cast<const uint8_t*>(bytes);
  const uint64_t bit_size = uint64_t(size) * 8;
  while (size >= 64) {
    compress(data);
    data += 64;
    size -= 64;
  }
  uint8_t tail[128]{};
  if (size != 0) std::memcpy(tail, data, size);
  tail[size] = 0x80;
  const size_t padded = size < 56 ? 64 : 128;
  for (size_t i = 0; i < 8; ++i) tail[padded - 1 - i] = uint8_t(bit_size >> (8 * i));
  compress(tail);
  if (padded == 128) compress(tail + 64);

  Digest digest{};
  for (size_t i = 0; i < state.size(); ++i) {
    for (size_t j = 0; j < 4; ++j) digest[4 * i + j] = uint8_t(state[i] >> (24 - 8 * j));
  }
  return digest;
}

}  // namespace srv::durable_image
