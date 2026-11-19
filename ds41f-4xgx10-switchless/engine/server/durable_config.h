// Boot-only opt-in for ordinary image checkpoints; never enables secure spill.
#pragma once
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include "common/keepfile.h"
#include "server/durable_cmd.h"

namespace srv::durable {
struct Config { std::string directory, namespace_id; size_t bytes = 0; };
struct Input { uint64_t enabled = 0, bytes = 0; char epoch[65] = {}; };
inline bool parse(Input& out, const char* ordinary, const char* enabled, const char* budget, const char* epoch) {
  out = {};
  if (!enabled || !std::strcmp(enabled, "0")) return !budget && !epoch;
  if (std::strcmp(enabled, "1") || !ordinary || std::strcmp(ordinary, "1") || !budget || !epoch || std::strlen(epoch) != 64) return false;
  for (size_t i = 0; i < 64; ++i) if (!((epoch[i] >= '0' && epoch[i] <= '9') || (epoch[i] >= 'a' && epoch[i] <= 'f'))) return false;
  uint64_t n = 0;
  for (const char* c = budget; *c; ++c) {
    if (*c < '0' || *c > '9' || n > (std::numeric_limits<size_t>::max() - (*c - '0')) / 10) return false;
    n = n * 10 + (*c - '0');
  }
  if (!n) return false;
  out.enabled = 1; out.bytes = n; std::memcpy(out.epoch, epoch, 64); return true;
}
inline void private_directory(const std::string& path) {
  if (::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) throw std::runtime_error("private durable directory unavailable");
  struct stat st{};
  if (::lstat(path.c_str(), &st) || !S_ISDIR(st.st_mode) || st.st_uid != ::geteuid() || (st.st_mode & 07777) != 0700)
    throw std::runtime_error("durable directory must be owned mode0700 without symlink");
}
// All ranks enter once, including disabled ranks, so mismatched gates refuse
// rather than entering different command/collective sequences.
template<class Model, class Ring, class Identity, class Eligible>
Config configure(Model& m, Ring& ring, int rank, const std::string& disk_dir,
                 const std::string& disk_tag, Identity identity, Eligible eligible) {
  Input mine{};
  const bool valid = parse(mine, std::getenv("DSGB10_ORDINARY_IMAGE_PREFIX_CACHE"),
      std::getenv("DSGB10_ORDINARY_IMAGE_DURABLE"), std::getenv("DSGB10_ORDINARY_IMAGE_DURABLE_BYTES"),
      std::getenv("DSGB10_ORDINARY_IMAGE_ASSETS_EPOCH"));
  if (!vote(ring, valid)) throw std::runtime_error("invalid ordinary durable opt-in");
  Input all[4]{}; ring.allgather_host(&mine, all, sizeof mine);
  for (const auto& x : all) if (x.enabled != mine.enabled || x.bytes != mine.bytes || std::strcmp(x.epoch, mine.epoch))
    throw std::runtime_error("ordinary durable configuration differs across ranks");
  if (!mine.enabled) return {};
  if (!vote(ring, !disk_dir.empty() && eligible())) throw std::runtime_error("ordinary durable path refused");
  const uint64_t local = identity(); uint64_t ids[4]{};
  ring.allgather_host(&local, ids, sizeof local);
  const uint64_t combined = keepfile::fnv(ids, sizeof ids);
  char hex[17]; std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long)combined);
  Config result; result.namespace_id = std::string(hex) + "-" + mine.epoch;
  const std::string root = disk_dir + "/ordinary-image" + disk_tag + "." + result.namespace_id;
  result.directory = root + "/head"; result.bytes = mine.bytes;
  bool ready = false;
  try {
    private_directory(disk_dir); private_directory(root);
    if (rank == 0) private_directory(result.directory);
    m.state_durable_config(root + "/rank" + std::to_string(rank), result.bytes, local);
    ready = true;
  } catch (const std::exception&) { /* all ranks refuse after the setup vote */ }
  if (!vote(ring, ready)) throw std::runtime_error("ordinary durable rank setup refused");
  return result;
}
}  // namespace srv::durable
