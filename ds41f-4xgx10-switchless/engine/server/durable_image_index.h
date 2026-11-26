#pragma once

#include "../common/frontend_protocol.h"
#include "../common/keepfile.h"
#include "../model/config.h"
#include "durable_image_sha256.h"

#include <dirent.h>
#include <sys/file.h>

#include <algorithm>
#include <charconv>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <tuple>
#include <utility>

namespace srv::durable_image {

static_assert(std::endian::native == std::endian::little);

using CheckpointId = uint64_t;

struct ImageIdentity {
  uint32_t start;
  uint32_t rows;
  Digest digest;
  bool operator==(const ImageIdentity&) const = default;
};
using Images = std::vector<ImageIdentity>;

template <class Spans>
bool valid_layout(const std::vector<int32_t>& prompt, const Spans& spans) {
  if (spans.empty() || spans.size() > fe::kMaxImages) return false;
  size_t end = 0;
  for (const auto& span : spans) {
    if (span.start < end || span.rows < 3 || span.rows > fe::kMaxImageRows) return false;
    end = size_t(span.start) + span.rows;
    if (end > prompt.size() || !std::all_of(prompt.begin() + span.start, prompt.begin() + end,
                                           [](int32_t t) { return t == int32_t(fe::kImageTokenId); })) return false;
  }
  return true;
}

// Hash the actual BF16 data sent to the model, including start/end and newline rows. The frontend
// image-source hash is a log field. The span bounds are stored and compared separately.
inline std::optional<Images> identify(const std::vector<int32_t>& prompt, const std::vector<fe::ImageSpan>& spans,
                                      const std::vector<uint16_t>& rows) {
  if (!valid_layout(prompt, spans)) return std::nullopt;
  size_t words = 0;
  for (const auto& span : spans) words += size_t(span.rows) * fe::kImageDim;
  if (words != rows.size()) return std::nullopt;

  Images images;
  images.reserve(spans.size());
  size_t offset = 0;
  for (const auto& span : spans) {
    const size_t count = size_t(span.rows) * fe::kImageDim;
    images.push_back({span.start, span.rows, sha256(rows.data() + offset, count * sizeof(uint16_t))});
    offset += count;
  }
  return images;
}

struct Entry {
  CheckpointId id;
  std::vector<int32_t> tokens;
  Images images;
  uint64_t written;
  uint64_t checkpoint_bytes;
  int snap;
  bool encoder_only;
};

struct Limits {
  size_t entries;
  uint64_t bytes;
  uint64_t max_age_seconds;
};

// Controller-owned and single-threaded. A dedicated, pre-existing 0700 directory is locked for
// this index's lifetime and bound to a deployment namespace on first use. Publish only after every
// rank confirms its immutable checkpoint; unlink metadata before deleting the rank checkpoints.
class Index {
 public:
  Index() = default;
  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;
  ~Index() { if (dir_ >= 0) close(dir_); }

  bool open(const std::string& private_dir, const std::string& cache_namespace, Limits limits) {
    if (enabled() || private_dir.empty() || cache_namespace.empty() || cache_namespace.size() > 4096 ||
        !limits.entries || limits.entries > kMaxEntries || limits.bytes <= kFixedBytes ||
        !limits.max_age_seconds) return false;
    dir_ = ::open(private_dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (dir_ < 0) return false;
    if (flock(dir_, LOCK_EX | LOCK_NB) != 0 || !private_directory()) return disable();
    limits_ = limits;
    namespace_ = sha256(cache_namespace.data(), cache_namespace.size());

    const auto names = list_ids();
    if (!names) return disable();
    struct stat st{};
    if (fstatat(dir_, kNamespaceFile, &st, AT_SYMLINK_NOFOLLOW) == 0) {
      const auto binding = read_file(kNamespaceFile, namespace_.size(), true);
      if (!binding || binding->size() != namespace_.size() ||
          !std::equal(binding->begin(), binding->end(), namespace_.begin())) return disable();
      if (!unlink_file(kPendingFile)) return disable();
    } else {
      if (errno != ENOENT || !names->empty() || !unlink_file(kPendingFile) ||
          !publish_file(kNamespaceFile, namespace_.data(), namespace_.size())) return disable();
    }

    for (CheckpointId id : *names) {
      const auto data = read_file(filename(id).c_str(), kMaxRecordBytes);
      // A failed open or read leaves the file unverified. Never direct ranks to delete its state.
      if (!data) return disable();
      auto entry = decode(id, *data);
      const auto charge = entry ? cost(entry->checkpoint_bytes, entry->tokens.size(), entry->images.size()) : std::nullopt;
      if (!charge || *charge > limits_.bytes || !load_room(*entry, *charge)) {
        if (!enabled() || !remove_file(id)) return disable();
        continue;
      }
      bytes_ += *charge;
      entries_.emplace(id, std::move(*entry));
    }
    return true;
  }

  bool enabled() const { return dir_ >= 0; }
  bool contains(CheckpointId id) const { return entries_.contains(id); }
  size_t size() const { return entries_.size(); }
  uint64_t bytes() const { return bytes_; }

  // Used for reconciliation only after a successful open of the entire private index.
  std::vector<CheckpointId> ids() const {
    std::vector<CheckpointId> ids;
    ids.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) ids.push_back(id);
    return ids;
  }

  bool room_for(uint64_t checkpoint_bytes, size_t prefix_tokens, size_t image_count) const {
    const auto charge = cost(checkpoint_bytes, prefix_tokens, image_count);
    return enabled() && charge && *charge <= limits_.bytes &&
           size() < limits_.entries && bytes_ <= limits_.bytes - *charge;
  }

  const Entry* find(const std::vector<int32_t>& prompt, const Images& images, uint64_t now, bool ced,
                    const std::set<CheckpointId>& excluded = {}) {
    prune(now);
    if (!enabled() || !valid_layout(prompt, images)) return nullptr;
    const Entry* best = nullptr;
    for (const auto& [id, entry] : entries_) {
      const size_t q = entry.tokens.size();
      if (excluded.contains(id) || q >= prompt.size() || (entry.encoder_only && (!ced || prompt.size() - q < kDecoderReplay)) ||
          (best && std::make_pair(q, age_key(entry)) <= std::make_pair(best->tokens.size(), age_key(*best)))) continue;
      const auto count = prefix_images(images, q);
      if (count && *count == entry.images.size() && std::equal(entry.images.begin(), entry.images.end(), images.begin()) &&
          std::equal(entry.tokens.begin(), entry.tokens.end(), prompt.begin())) best = &entry;
    }
    return best;
  }

  bool publish(CheckpointId id, const std::vector<int32_t>& prompt, const Images& images, size_t prefix_tokens,
               uint64_t checkpoint_bytes, uint64_t now, int snap, bool encoder_only) {
    if (!enabled() || !id || !now || contains(id) || prefix_tokens > kMaxTokens || snap < 1 ||
        size_t(snap & ~1) != prefix_tokens || size_t(snap) > prompt.size() || !valid_layout(prompt, images)) return false;
    const auto count = prefix_images(images, prefix_tokens);
    if (!count || !*count || (size_t(snap) > prefix_tokens && *count < images.size() &&
                              images[*count].start == prefix_tokens)) return false;
    const auto charge = cost(checkpoint_bytes, prefix_tokens, *count);
    if (!charge || *charge > limits_.bytes || !valid_tokens(prompt.begin(), prompt.begin() + prefix_tokens)) return false;
    prune(now);
    if (!enabled()) return false;
    while (!room_for(checkpoint_bytes, prefix_tokens, *count))
      if (!evict_oldest()) return false;

    Entry entry{id, {prompt.begin(), prompt.begin() + prefix_tokens}, {images.begin(), images.begin() + *count},
                now, checkpoint_bytes, snap, encoder_only};
    const auto data = encode(entry);
    if (!publish_file(filename(id).c_str(), data.data(), data.size())) return disable();
    entries_.emplace(id, std::move(entry));
    bytes_ += *charge;
    return true;
  }

  bool erase(CheckpointId id) {
    const auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    if (!remove_file(id)) return disable();
    bytes_ -= *cost(it->second.checkpoint_bytes, it->second.tokens.size(), it->second.images.size());
    entries_.erase(it);
    return true;
  }

  bool evict_oldest() { return !entries_.empty() && erase(oldest()->first); }

  void prune(uint64_t now) {
    if (!enabled() || !private_directory()) { disable(); return; }
    for (auto it = entries_.begin(); it != entries_.end();) {
      const Entry& entry = (it++)->second;
      if (entry.written > now || now - entry.written >= limits_.max_age_seconds) {
        if (!erase(entry.id)) return;
      }
    }
  }

  std::vector<CheckpointId> take_evicted() { return std::exchange(evicted_, {}); }

 private:
  static constexpr size_t kMaxEntries = 128;
  static constexpr size_t kMaxDirectoryEntries = 2 * kMaxEntries;
  static constexpr size_t kMaxTokens = 1u << 20;
  static constexpr size_t kDecoderReplay = 128;
  static constexpr size_t kFixedBytes = 112, kImageBytes = 40;
  static constexpr size_t kMaxRecordBytes = kFixedBytes + 4 * kMaxTokens + kImageBytes * fe::kMaxImages;
  static constexpr std::array<uint8_t, 8> kMagic = {'D', 'S', 'G', 'B', 'I', 'M', 'G', '1'};
  static constexpr const char* kNamespaceFile = "namespace.sha256";
  static constexpr const char* kPendingFile = ".pending";

  using EntryMap = std::map<CheckpointId, Entry>;
  EntryMap entries_;
  std::vector<CheckpointId> evicted_;
  Digest namespace_{};
  Limits limits_{};
  uint64_t bytes_ = 0;
  int dir_ = -1;

  bool disable() {
    if (dir_ >= 0) close(std::exchange(dir_, -1));
    entries_.clear();
    bytes_ = 0;
    return false;
  }

  bool private_directory() const {
    struct stat st{};
    return fstat(dir_, &st) == 0 && S_ISDIR(st.st_mode) && st.st_uid == geteuid() && (st.st_mode & 07777) == 0700;
  }

  static std::optional<size_t> prefix_images(const Images& images, size_t cut) {
    size_t count = 0;
    for (const auto& image : images) {
      if (image.start >= cut) break;
      if (size_t(image.start) + image.rows > cut) return std::nullopt;
      ++count;
    }
    return count;
  }

  static bool valid_tokens(std::vector<int32_t>::const_iterator begin, std::vector<int32_t>::const_iterator end) {
    return std::all_of(begin, end, [](int32_t t) { return t >= 0 && t < model::kVocab; });
  }

  static size_t record_size(size_t tokens, size_t images) { return kFixedBytes + 4 * tokens + kImageBytes * images; }

  static std::optional<uint64_t> cost(uint64_t rank_bytes, size_t tokens, size_t images) {
    if (!tokens || tokens > kMaxTokens || !images || images > fe::kMaxImages || !rank_bytes) return std::nullopt;
    const size_t metadata = record_size(tokens, images);
    if (rank_bytes > std::numeric_limits<uint64_t>::max() - metadata) return std::nullopt;
    return rank_bytes + metadata;
  }

  static std::pair<uint64_t, CheckpointId> age_key(const Entry& entry) { return {entry.written, entry.id}; }

  EntryMap::iterator oldest() {
    return std::min_element(entries_.begin(), entries_.end(), [](const auto& a, const auto& b) {
      return age_key(a.second) < age_key(b.second);
    });
  }

  bool load_room(const Entry& entry, uint64_t charge) {
    size_t count = size();
    uint64_t bytes = bytes_;
    for (const auto& [id, candidate] : entries_) {
      if (age_key(candidate) >= age_key(entry)) continue;
      --count;
      bytes -= *cost(candidate.checkpoint_bytes, candidate.tokens.size(), candidate.images.size());
    }
    if (count >= limits_.entries || bytes > limits_.bytes - charge) return false;
    while (size() >= limits_.entries || bytes_ > limits_.bytes - charge) {
      if (!evict_oldest()) return false;
    }
    return true;
  }

  static std::string filename(CheckpointId id) {
    std::string name = "i-0000000000000000.idx";
    constexpr char hex[] = "0123456789abcdef";
    for (size_t j = 0; j < 16; ++j) name[17 - j] = hex[(id >> (4 * j)) & 15];
    return name;
  }

  std::optional<std::vector<CheckpointId>> list_ids() const {
    const int fd = fcntl(dir_, F_DUPFD_CLOEXEC, 0);
    if (fd < 0) return std::nullopt;
    DIR* stream = fdopendir(fd);
    if (!stream) { close(fd); return std::nullopt; }
    std::vector<CheckpointId> ids;
    bool ok = true;
    while (true) {
      errno = 0;
      const dirent* d = readdir(stream);
      if (!d) { ok = errno == 0; break; }
      const std::string_view name(d->d_name);
      if (name == "." || name == ".." || name == kNamespaceFile || name == kPendingFile) continue;
      CheckpointId id = 0;
      if (name.size() != 22 || ids.size() >= kMaxDirectoryEntries) { ok = false; break; }
      const auto end = name.data() + 18;
      const auto result = std::from_chars(name.data() + 2, end, id, 16);
      if (result.ec != std::errc{} || result.ptr != end || !id || name != filename(id)) { ok = false; break; }
      ids.push_back(id);
    }
    closedir(stream);
    if (!ok) return std::nullopt;
    return ids;
  }

  std::optional<std::vector<uint8_t>> read_file(const char* name, size_t max, bool allow_pending_link = false) const {
    const int fd = openat(dir_, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) return std::nullopt;
    struct stat st{};
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != geteuid() || (st.st_mode & 07777) != 0600 ||
        st.st_size < 0 || uint64_t(st.st_size) > max) { close(fd); return std::nullopt; }
    if (st.st_nlink != 1) {
      struct stat pending{};
      if (!allow_pending_link || st.st_nlink != 2 || fstatat(dir_, kPendingFile, &pending, AT_SYMLINK_NOFOLLOW) != 0 ||
          pending.st_dev != st.st_dev || pending.st_ino != st.st_ino) { close(fd); return std::nullopt; }
    }
    std::vector<uint8_t> bytes(size_t(st.st_size), 0);
    size_t at = 0;
    while (at < bytes.size()) {
      const ssize_t count = read(fd, bytes.data() + at, bytes.size() - at);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) break;
      at += size_t(count);
    }
    close(fd);
    if (at != bytes.size()) return std::nullopt;
    return bytes;
  }

  bool unlink_file(const char* name, bool allow_missing = true) const {
    if (unlinkat(dir_, name, 0) != 0) return allow_missing && errno == ENOENT;
    return fsync(dir_) == 0;
  }

  bool remove_file(CheckpointId id) {
    if (!unlink_file(filename(id).c_str(), false)) return false;
    evicted_.push_back(id);
    return true;
  }

  bool publish_file(const char* name, const uint8_t* data, size_t bytes) const {
    const int fd = openat(dir_, kPendingFile, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return false;
    size_t at = 0;
    if (fchmod(fd, 0600) == 0) {
      while (at < bytes) {
        const ssize_t count = write(fd, data + at, bytes - at);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        at += size_t(count);
      }
    }
    const bool synced = at == bytes && fsync(fd) == 0;
    const bool closed = close(fd) == 0;
    // linkat is atomic and preserves an existing checkpoint ID. Startup removes any interrupted
    // pending link before opening committed metadata, so a committed file has exactly one link.
    const bool linked = synced && closed && linkat(dir_, kPendingFile, dir_, name, 0) == 0;
    const bool cleaned = unlinkat(dir_, kPendingFile, 0) == 0;
    if (linked && cleaned && fsync(dir_) == 0) return true;
    if (linked) unlink_file(name);
    return false;
  }

  std::vector<uint8_t> encode(const Entry& entry) const {
    std::vector<uint8_t> data;
    data.reserve(record_size(entry.tokens.size(), entry.images.size()));
    using keepfile::put;
    put(data, kMagic);
    put(data, namespace_);
    put(data, entry.id);
    put(data, entry.written);
    put(data, entry.checkpoint_bytes);
    put(data, uint32_t(entry.snap));
    put(data, uint32_t(entry.encoder_only));
    put(data, uint32_t(entry.tokens.size()));
    put(data, uint32_t(entry.images.size()));
    for (int32_t token : entry.tokens) put(data, token);
    for (const auto& image : entry.images) {
      put(data, image.start);
      put(data, image.rows);
      put(data, image.digest);
    }
    put(data, sha256(data.data(), data.size()));
    return data;
  }

  std::optional<Entry> decode(CheckpointId filename_id, const std::vector<uint8_t>& data) const {
    if (data.size() < kFixedBytes) return std::nullopt;
    const size_t body = data.size() - sizeof(Digest);
    keepfile::Reader r{data.data(), body};
    if (r.get<std::array<uint8_t, 8>>() != kMagic || r.get<Digest>() != namespace_) return std::nullopt;
    const auto id = r.get<CheckpointId>();
    const auto written = r.get<uint64_t>();
    const auto rank_bytes = r.get<uint64_t>();
    const auto snap = r.get<uint32_t>();
    const auto encoder_only = r.get<uint32_t>();
    const auto nt = r.get<uint32_t>(), ni = r.get<uint32_t>();
    if (!r.ok || id != filename_id || !written || (snap & ~1u) != nt || encoder_only > 1 ||
        !cost(rank_bytes, nt, ni) || data.size() != record_size(nt, ni)) return std::nullopt;
    const Digest checksum = sha256(data.data(), body);
    if (!std::equal(checksum.begin(), checksum.end(), data.begin() + body)) return std::nullopt;

    Entry entry{id, {}, {}, written, rank_bytes, int(snap), bool(encoder_only)};
    entry.tokens.reserve(nt);
    entry.images.reserve(ni);
    for (uint32_t i = 0; i < nt; ++i) entry.tokens.push_back(r.get<int32_t>());
    for (uint32_t i = 0; i < ni; ++i)
      entry.images.push_back({r.get<uint32_t>(), r.get<uint32_t>(), r.get<Digest>()});
    if (!r.ok || r.at != body || !valid_tokens(entry.tokens.begin(), entry.tokens.end()) ||
        !valid_layout(entry.tokens, entry.images)) return std::nullopt;
    return entry;
  }
};

}  // namespace srv::durable_image
