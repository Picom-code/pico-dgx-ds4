// Synthetic host pool and spill sink. The allocator and eviction functions
// included below come from the actual source under test.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

constexpr size_t kPageBytes = 16;
size_t snap_bytes(bool) { return kPageBytes; }
size_t end_bytes(bool) { return kPageBytes; }
double wall_s() { return 0; }

struct Pool {
  static constexpr size_t kNone = SIZE_MAX;
  uint8_t block[kPageBytes]{};
  size_t size = sizeof block;
  bool taken = true;

  static size_t round(size_t bytes) { return bytes; }
  uint8_t* take(size_t bytes) {
    if (taken || bytes > size) return nullptr;
    taken = true;
    return block;
  }
};

struct Spill {
  size_t writes = 0;
  uint64_t sum_word = 0;
  void begin(int) {}
  uint64_t write(uint8_t*, size_t, size_t, int) { return ++writes; }
  uint64_t* sum(size_t, bool) { return &sum_word; }
  void flush(int) {}
};

struct Store {
  struct Sess {
    uint64_t used = 0;
    bool disk = false;
    bool ram_only = false;
    std::vector<uint8_t*> pages;
    std::map<int, uint8_t*> snaps;
    uint8_t* end = nullptr;
    uint64_t dseq = 0;
    std::vector<size_t> dpages;
    std::map<int, size_t> dsnaps;
    size_t dend = Pool::kNone;
  };
  Sess& use(int h) { return sess[h]; }
  std::map<int, Sess> sess;
  std::vector<int> pinned, evicted;
  Pool pool, dpool;
  size_t bytes = kPageBytes;
  size_t dbytes = 0;
  uint64_t clock = 0;
  Spill* disk = nullptr;
};

void sum_block(uint8_t*, size_t, uint64_t*, int) {}
void disk_release(Store&, Store::Sess&, bool) {}
constexpr int cudaMemcpyDeviceToDevice = 0;
void cudaMemcpyAsync(uint8_t* dst, uint8_t* src, size_t bytes, int, int) {
  std::memcpy(dst, src, bytes);
}
#define CUDA_CHECK(expr) (expr)
[[noreturn]] void die(const char* message, int = 0) { throw std::runtime_error(message); }

struct Model {
  struct Impl {
    Store store;
    int rank = 1;
    int st = 0;
    void* spec = nullptr;
    Store& store_of() { return store; }
    void state_spill(int);
    void state_copy_snaps(int, int, int);
    uint8_t* store_alloc(int, size_t);
    size_t disk_take(int, size_t bytes) {
      const size_t off = store.dbytes;
      store.dbytes += bytes;
      return off;
    }
    void store_release(uint8_t* p, size_t bytes) {
      if (!p) return;
      store.bytes -= bytes;
      store.pool.taken = false;
    }
    void state_free(int h) {
      auto& ss = store.sess.at(h);
      for (auto* p : ss.pages) store_release(p, kPageBytes);
      for (const auto& [position, p] : ss.snaps) store_release(p, snap_bytes(false));
      store_release(ss.end, end_bytes(false));
      store.sess.erase(h);
    }
    void state_unspill(int) { die("test unexpectedly attempted unspill"); }
    uint8_t* state_snap_buf(int, int) { return nullptr; }
  } impl;
  Impl* d_ = &impl;
  void state_mark_ram_only(int);
};

#include "allocator.inc"

int main() {
  constexpr int image = 7, admitted = 9;
  Model m;
  Spill sink;
  auto& store = m.impl.store;
  store.disk = &sink;
  auto& img = store.sess[image];
  img.pages.push_back(store.pool.block);
  img.used = 2;

#ifdef EXPECT_UPSTREAM_BASELINE
  img.ram_only = true; // original allocator has no mark or eligibility check
  assert(m.impl.store_alloc(admitted, kPageBytes) == store.pool.block);
  assert(sink.writes == 1 && store.sess.at(image).disk && store.evicted.empty());
  std::puts("upstream allocator reproduced the image spill under pressure");
#else
  constexpr int text = 8;
  m.state_mark_ram_only(image);
  store.sess[text].used = 1;
  assert(m.impl.store_alloc(admitted, kPageBytes) == store.pool.block);
  assert(sink.writes == 0 && store.dbytes == 0 && !store.sess.count(image));
  assert((store.evicted == std::vector<int>{image}));
  assert(store.sess.count(text));

  // Existing text overflow still uses the disk sink.
  store.sess[text].pages.push_back(store.pool.block);
  assert(m.impl.store_alloc(admitted, kPageBytes) == store.pool.block);
  assert(sink.writes == 1 && store.sess.at(text).disk);

  // The state-copy path independently refuses spilling an image fork into text.
  m.state_mark_ram_only(image);
  try {
    m.impl.state_copy_snaps(image, admitted, 256);
    return 1;
  } catch (const std::runtime_error& e) {
    assert(std::strcmp(e.what(), "state: RAM-only session cannot copy snapshots into a spillable session") == 0);
  }
  m.state_mark_ram_only(admitted);
  m.impl.state_copy_snaps(image, admitted, 256);

  // An explicit spill must drop the RAM-only state as well.
  store.sess[image].pages.push_back(store.pool.block);
  m.impl.state_spill(image);
  assert(sink.writes == 1 && !store.sess.count(image));
  std::puts("patched allocator drops RAM-only image state and retains ordinary text spill");
#endif
}
