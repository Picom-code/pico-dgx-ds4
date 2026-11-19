// Compile the production restore, allocation and release code around a host spill file.
// Reduced block sizes exercise tight pools without allocating model state.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class SpillFile;
#include "store.inc"

constexpr size_t kPageBytes = 5 * kPoolGrain;
size_t snap_bytes(bool) { return 3 * kPoolGrain; }
size_t end_bytes(bool) { return 4 * kPoolGrain; }
double wall_s() { return 0; }

void sum_block(const uint8_t* data, size_t bytes, uint64_t* out, int) {
  uint64_t hash = 14695981039346656037ULL;
  for (size_t i = 0; i < Pool::round(bytes); ++i)
    hash = (hash ^ data[i]) * 1099511628211ULL;
  *out = hash;
}

class SpillFile {
 public:
  explicit SpillFile(size_t capacity) : file(capacity) {}
  void begin(int) {}
  void flush(int) {}
  void wait(uint64_t expected) const { assert(expected <= seq); }
  uint64_t* sum(size_t offset, bool read_back) {
    return &(read_back ? checked : written)[offset];
  }
  uint64_t write(const uint8_t* data, size_t bytes, size_t offset, int) {
    std::memcpy(file.data() + offset, data, Pool::round(bytes));
    return ++seq;
  }
  void read(uint8_t* data, size_t bytes, size_t offset, int) {
    std::memcpy(data, file.data() + offset, Pool::round(bytes));
    reads.push_back(offset);
  }
  void verify_reads() const {
    for (const size_t offset : reads)
      assert(checked.at(offset) == written.at(offset));
  }

  std::vector<size_t> reads;
  std::vector<uint8_t> file;
  std::map<size_t, uint64_t> written, checked;
  uint64_t seq = 0;
};

Store::~Store() = default;
void cudaStreamSynchronize(int) {}
#define CUDA_CHECK(expr) (expr)
[[noreturn]] void die(const char* message, int, int) { throw std::runtime_error(message); }

struct Model {
  struct Impl {
    Store data;
    int rank = 0;
    int st = 0;
    void* spec = nullptr;

    Store& store_of() { return data; }
    uint8_t* store_alloc(int, size_t);
    void store_release(uint8_t*, size_t);
    void state_free(int);
    size_t disk_take(int, size_t);
    void state_spill(int);
#ifdef PATCHED_RESTORE
    void state_unspill(int, int, int, bool);
#else
    void state_unspill(int);
#endif
  };
};

#include "restore.inc"

constexpr int kSource = 1;
constexpr int kPinned = 2;
constexpr int kEarly = 2 * kPagePos + 1;
constexpr int kTail = 3 * kPagePos - 63;
constexpr int kLate = 3 * kPagePos + 1;
constexpr int kBeyond = 4 * kPagePos + 1;
constexpr int kPages = 5;

uint8_t window_value(int key) {
  return 40 + key / kPagePos + (key == kTail ? 10 : 0);
}

size_t put(Model::Impl& model, size_t size, uint8_t value) {
  auto& store = model.data;
  const size_t off = model.disk_take(kSource, size);
  assert(off != Pool::kNone);
  std::vector<uint8_t> bytes(Pool::round(size), value);
  sum_block(bytes.data(), size, store.disk->sum(off, false), model.st);
  store.sess.at(kSource).dseq = store.disk->write(bytes.data(), size, off, model.st);
  return off;
}

void check_block(const uint8_t* p, size_t bytes, uint8_t value) {
  assert(p);
  assert(std::all_of(p, p + Pool::round(bytes), [=](uint8_t b) { return b == value; }));
}

struct Case {
  const char* label;
  size_t room;
  bool has_end;
  int end_position;
  int limit;
  int expected_pages;
  std::vector<int> expected_windows;
  bool expected_end;
  int preferred_snap = -1;
  bool end_first = false;
};

std::vector<size_t> run(const Case& test, int rank) {
  Model::Impl model;
  model.rank = rank;
  auto& store = model.data;
  std::vector<uint8_t> memory(kPageBytes + test.room);
  store.pool.base = memory.data();
  store.pool.size = memory.size();
  store.pool.free.emplace(0, memory.size());
  store.dpool.size = 10 * kPageBytes;
  store.dpool.free.emplace(0, store.dpool.size);
  store.disk = std::make_unique<SpillFile>(store.dpool.size);

  auto& pinned = store.sess[kPinned];
  auto* occupied = model.store_alloc(kPinned, kPageBytes);
  assert(occupied);
  pinned.pages.push_back(occupied);
  std::memset(occupied, 99, kPageBytes);
  store.pinned = {kPinned};

  auto& source = store.sess[kSource];
  source.disk = true;
  source.kept = true;
  source.saved = kPages * kPagePos;
  for (int p = 0; p < kPages; ++p)
    source.dpages.push_back(put(model, kPageBytes, 10 + p));
  for (const int key : {kEarly, kTail, kLate, kBeyond})
    source.dsnaps[key] = put(model, snap_bytes(false), window_value(key));
  if (test.has_end) {
    source.end_pos = test.end_position;
    source.dend = put(model, end_bytes(false), 66);
  }

#ifdef PATCHED_RESTORE
  model.state_unspill(kSource, test.limit, test.preferred_snap, test.preferred_snap >= 0 && !test.end_first);
#else
  model.state_unspill(kSource);
#endif
  assert(store.pinned == std::vector<int>{kPinned});
  assert(store.evicted.empty());
  assert(pinned.pages.front() == occupied);
  check_block(occupied, kPageBytes, 99);
  assert(!source.disk && !source.kept);
  assert(source.dpages.empty() && source.dsnaps.empty() && source.dend == Pool::kNone);
  assert(store.dbytes == 0 && store.dpool.largest() == store.dpool.size);
  assert(source.saved == test.expected_pages * kPagePos);
  assert(source.pages.size() == static_cast<size_t>(test.expected_pages));
  assert((source.end != nullptr) == test.expected_end);
  assert(source.end_pos == (test.expected_end ? test.end_position : -1));
  assert(source.snaps.size() == test.expected_windows.size());
  if (test.limit >= 0 && !source.end) {
    const int cap = std::min(source.saved, test.limit);
    int restart = 0;
    for (const auto& window : source.snaps) {
      const int pos = window.first & ~1;
      if (pos <= cap) restart = std::max(restart, pos);
    }
    int expected_restart = 0;
    for (const int window : test.expected_windows) {
      const int pos = window & ~1;
      if (pos <= cap) expected_restart = std::max(expected_restart, pos);
    }
    assert(restart == expected_restart);
    if (rank == 0) std::printf("covered window restart: %d positions\n", restart);
  }
  store.disk->verify_reads();

  std::vector<size_t> layout;
  for (size_t i = 0; i < source.pages.size(); ++i) {
    check_block(source.pages[i], kPageBytes, 10 + i);
    layout.push_back(source.pages[i] - memory.data());
  }
  for (const int key : test.expected_windows) {
    const auto* bytes = source.snaps.at(key);
    check_block(bytes, snap_bytes(false), window_value(key));
    layout.push_back(bytes - memory.data());
  }
  if (source.end) {
    check_block(source.end, end_bytes(false), 66);
    layout.push_back(source.end - memory.data());
  }
  assert(store.bytes == kPageBytes * (1 + source.pages.size()) +
      snap_bytes(false) * source.snaps.size() + (source.end ? end_bytes(false) : 0));
  layout.insert(layout.end(), store.disk->reads.begin(), store.disk->reads.end());
  return layout;
}

int main() {
#ifdef PATCHED_RESTORE
  const int pressure_pages = 2;
  const std::vector<int> pressure_windows{kEarly};
  const std::vector<int> cap_choice{kEarly};
  const std::vector<int> tail_choice{kTail};
  const std::vector<int> end_competing_choice{kEarly};
  const bool end_competing_restored = false;
#else
  const int pressure_pages = 3;
  const std::vector<int> pressure_windows;
  const std::vector<int> cap_choice{kBeyond};
  const std::vector<int> tail_choice{kBeyond};
  const std::vector<int> end_competing_choice;
  const bool end_competing_restored = true;
#endif
  const size_t snap = snap_bytes(false);
  const size_t end = end_bytes(false);
  const int early_limit = kEarly & ~1;
  const std::vector<Case> cases{
      {"full", kPages * kPageBytes + 4 * snap + end, true, kPages * kPagePos,
       early_limit, kPages, {kEarly, kTail, kLate, kBeyond}, true, kEarly},
      {"generic pressure", 3 * kPageBytes, false, 0, -1, 3, {}, false},
      {"covered older window under pressure", 3 * kPageBytes, false, 0, early_limit,
       pressure_pages, pressure_windows, false},
      {"latest window loses coverage when the last page is released", 3 * kPageBytes,
       false, 0, kLate & ~1, pressure_pages, pressure_windows, false},
      {"one window fits without releasing pages", 3 * kPageBytes + snap, false, 0,
       early_limit, 3, cap_choice, false},
      {"end preferred but only a smaller decoder-ready window fits", 3 * kPageBytes + snap,
       true, 3 * kPagePos, early_limit, 3, tail_choice, false, kTail, true},
      {"window beyond cap", 3 * kPageBytes, false, 0, early_limit - 1, 3, {}, false},
      {"window beyond remaining pages", 2 * kPageBytes, false, 0, early_limit, 2, {}, false},
      {"no room", 0, false, 0, early_limit, 0, {}, false},
      {"preserve restored end", 3 * kPageBytes + end, true, 3 * kPagePos, early_limit,
       3, {}, true},
      {"prefer the window eligible at the decoder tail", 3 * kPageBytes + snap, false, 0,
       3 * kPagePos, 3, tail_choice, false, kTail},
      {"prefer the scheduler window when it competes with end", 3 * kPageBytes + end, true,
       3 * kPagePos, early_limit, 3, end_competing_choice, end_competing_restored, kEarly},
  };
  for (const Case& test : cases) {
    std::printf("case: %s\n", test.label);
    std::fflush(stdout);
    const auto rank_zero = run(test, 0);
    for (int rank = 1; rank < 4; ++rank)
      assert(run(test, rank) == rank_zero);
  }
  std::printf("PASS: %zu restore cases, each repeated with four rank IDs\n", cases.size());
}
