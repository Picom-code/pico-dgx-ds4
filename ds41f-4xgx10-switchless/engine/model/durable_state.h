// Ordinary checkpoint payloads are separate from the mutable session spill file.
// This layer writes one private, immutable file per rank and generation.
#pragma once

#include <dirent.h>
#include <sys/file.h>
#include <sys/statvfs.h>

#include <algorithm>
#include <charconv>
#include <string_view>

#include "common/keepfile.h"

namespace model::detail::durable {

struct Fd {
  explicit Fd(int fd) : value(fd) {}
  ~Fd() { if (value >= 0) close(value); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  int value;
};

struct File {
  uint64_t bytes = 0;
  bool regular = false;
};

// Each input used by the on-disk header is fixed-width. The payload contains
// exact (unrounded) Pico pages and the selected window block.
struct Header {
  uint64_t magic = 0x314b435049435044ull;
  uint64_t run_id = 0, generation = 0, file_bytes = 0;
  uint64_t page_bytes = 0, window_bytes = 0, checksum = 0;
  int32_t rank = -1, saved = 0, snap = 0, pages = 0, encoder_only = 0, reserved = 0;
};
static_assert(sizeof(Header) == 80);

inline uint64_t header_checksum(const Header& source) {
  Header header = source;
  header.checksum = 0;
  return keepfile::fnv(&header, sizeof header);
}

inline bool transfer(int fd, void* data, size_t n, bool writing) {
  auto* bytes = static_cast<uint8_t*>(data);
  while (n) {
    const ssize_t done = writing ? write(fd, bytes, n) : read(fd, bytes, n);
    if (done < 0 && errno == EINTR) continue;
    if (done <= 0) return false;
    n -= (size_t)done;
    bytes += done;
  }
  return true;
}

inline bool put(int fd, const void* bytes, size_t n) {
  return transfer(fd, const_cast<void*>(bytes), n, true);
}

inline uint64_t block_seed(uint64_t generation, int32_t kind, int32_t position) {
  const uint64_t n = keepfile::fnv(&generation, sizeof generation);
  return keepfile::fnv(&position, sizeof position, keepfile::fnv(&kind, sizeof kind, n));
}

inline bool put_block(int fd, const void* bytes, size_t n, uint64_t seed) {
  const uint64_t checksum = keepfile::fnv(bytes, n, seed);
  return put(fd, &checksum, sizeof checksum) && put(fd, bytes, n);
}

inline bool get_block(int fd, void* bytes, size_t n, uint64_t seed) {
  uint64_t checksum;
  return transfer(fd, &checksum, sizeof checksum, false) && transfer(fd, bytes, n, false)
      && checksum == keepfile::fnv(bytes, n, seed);
}

class Files {
 public:
  void configure(const std::string& base, size_t limit, uint64_t run_id, uint64_t free_floor = 50ull << 30) {
    dir_ = keepfile::dir_of(base);
    const size_t slash = base.rfind('/');
    prefix_ = base.substr(slash == std::string::npos ? 0 : slash + 1) + ".checkpoint.";
    limit_ = limit;
    run_id_ = run_id;
    free_floor_ = free_floor;
  }
  uint64_t run_id() const { return run_id_; }

  template <typename Write>
  bool save(uint64_t generation, size_t bytes, Write write_payload) const {
    if (!generation || !bytes || bytes > limit_) return false;
    Fd dir(open_dir());
    if (dir.value < 0) return false;
    Fd lock(lock_rank(dir.value));
    if (lock.value < 0) return false;
    const std::string name = filename(generation), temp = prefix_ + "new";
    if (unlinkat(dir.value, temp.c_str(), 0) != 0 && errno != ENOENT) return false;
    struct stat existing{};
    if (fstatat(dir.value, name.c_str(), &existing, AT_SYMLINK_NOFOLLOW) == 0 || errno != ENOENT) return false;
    struct statvfs space{};
    if (fstatvfs(dir.value, &space) != 0 || !space.f_frsize ||
        bytes > UINT64_MAX - free_floor_) return false;
    const uint64_t need = bytes + free_floor_;
    if (space.f_bavail < need / space.f_frsize + (need % space.f_frsize != 0) ||
        !fits(dir.value, bytes)) return false;
    Fd file(openat(dir.value, temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    if (file.value < 0) return false;
    const bool complete = write_payload(file.value) && file_info(file.value).bytes == bytes && fsync(file.value) == 0;
    const bool published = complete && renameat(dir.value, temp.c_str(), dir.value, name.c_str()) == 0;
    if (published && fsync(dir.value) == 0) return true;
    unlinkat(dir.value, (published ? name : temp).c_str(), 0);
    return false;
  }

  template <typename Read>
  bool load(uint64_t generation, Read read_payload) const {
    if (!generation || !limit_) return false;
    Fd dir(open_dir());
    if (dir.value < 0) return false;
    Fd file(openat(dir.value, filename(generation).c_str(), O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC));
    if (file.value < 0) return false;
    const File info = file_info(file.value);
    return info.regular && info.bytes <= limit_ && read_payload(file.value, info.bytes);
  }

  void drop(uint64_t generation) const {
    if (!generation || !limit_) return;
    Fd dir(open_dir());
    if (dir.value < 0) return;
    Fd lock(lock_rank(dir.value));
    if (lock.value < 0) return;
    if (unlinkat(dir.value, filename(generation).c_str(), 0) == 0) fsync(dir.value);
  }

  // The caller must hold the validated head catalog before supplying its complete generation set.
  bool prune(const std::vector<uint64_t>& committed) const {
    if (committed.size() > 128) return false;
    auto sorted = committed;
    std::sort(sorted.begin(), sorted.end());
    if ((!sorted.empty() && sorted.front() == 0) ||
        std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return false;
    Fd dir(open_dir());
    if (dir.value < 0) return false;
    Fd lock(lock_rank(dir.value));
    if (lock.value < 0) return false;
    std::vector<std::string> stale;
    if (!scan(dir.value, [&](const char* name) {
      const std::string_view file(name);
      if (file.substr(0, prefix_.size()) != prefix_) return true;
      const std::string_view suffix = file.substr(prefix_.size());
      const uint64_t id = generation(suffix);
      if (suffix == "new" || (id && !std::binary_search(sorted.begin(), sorted.end(), id)))
        stale.emplace_back(name);
      return true;
    })) return false;
    bool ok = true;
    for (const auto& name : stale)
      if (!remove_private(dir.value, name.c_str())) ok = false;
    return fsync(dir.value) == 0 && ok;
  }

 private:
  int open_dir() const {
    if (!run_id_ || !limit_) return -1;
    const int fd = open(dir_.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return -1;
    struct stat st{};
    if (fstat(fd, &st) == 0 && st.st_uid == getuid() && (st.st_mode & 0777) == 0700) return fd;
    close(fd);
    return -1;
  }

  int lock_rank(int dir) const {
    const int fd = openat(dir, (prefix_ + "lock").c_str(),
                          O_WRONLY | O_CREAT | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return -1;
    if (file_info(fd).regular && flock(fd, LOCK_EX | LOCK_NB) == 0) return fd;
    close(fd);
    return -1;
  }

  static File file_info(int fd) {
    struct stat st{};
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != getuid() || st.st_nlink != 1 ||
        (st.st_mode & 0777) != 0600)
      return {};
    return {(uint64_t)st.st_size, true};
  }

  std::string filename(uint64_t generation) const {
    return prefix_ + std::to_string(generation);
  }

  static uint64_t generation(std::string_view suffix) {
    if (suffix.empty() || suffix.front() == '0') return 0;
    uint64_t id = 0;
    const auto end = suffix.data() + suffix.size();
    const auto parsed = std::from_chars(suffix.data(), end, id);
    return parsed.ptr == end && parsed.ec == std::errc{} ? id : 0;
  }

  static bool remove_private(int dir, const char* name) {
    struct stat st{};
    if (fstatat(dir, name, &st, AT_SYMLINK_NOFOLLOW) != 0) return errno == ENOENT;
    if (!S_ISREG(st.st_mode) || st.st_uid != getuid() || st.st_nlink != 1 ||
        (st.st_mode & 07777) != 0600) return true;
    return unlinkat(dir, name, 0) == 0 || errno == ENOENT;
  }

  template <typename Visit>
  static bool scan(int fd, Visit visit) {
    const int scan_fd = openat(fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (scan_fd < 0) return false;
    DIR* dir = fdopendir(scan_fd);
    if (!dir) {
      close(scan_fd);
      return false;
    }
    bool ok = true;
    while (ok) {
      errno = 0;
      const dirent* entry = readdir(dir);
      if (!entry) {
        ok = errno == 0;
        break;
      }
      ok = visit(entry->d_name);
    }
    closedir(dir);
    return ok;
  }

  bool fits(int fd, size_t bytes) const {
    size_t left = limit_ - bytes;
    return scan(fd, [&](const char* name) {
      const std::string_view file(name);
      if (file.substr(0, prefix_.size()) != prefix_ || !generation(file.substr(prefix_.size()))) return true;
      struct stat st{};
      if (fstatat(fd, name, &st, AT_SYMLINK_NOFOLLOW) != 0) return false;
      if (!S_ISREG(st.st_mode)) return true;
      if ((uint64_t)st.st_size > left) return false;
      left -= (size_t)st.st_size;
      return true;
    });
  }

  std::string dir_, prefix_;
  size_t limit_ = 0;
  uint64_t run_id_ = 0, free_floor_ = 0;
};

}  // namespace model::detail::durable
