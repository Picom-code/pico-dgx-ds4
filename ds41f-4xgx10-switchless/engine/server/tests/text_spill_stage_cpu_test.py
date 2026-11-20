"""Run the spill staging and optional-read bodies with synthetic CPU buffers."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2] / 'model'
state = (root / 'state.cu').read_text()
spill = (root / 'spill.cpp').read_text()


def part(source, first, last):
    start = source.index(first)
    return source[start:source.index(last, start)]


test = r'''
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <vector>

using cudaStream_t = int;
constexpr int cudaMemcpyHostToDevice = 1;
#define CUDA_CHECK(expr) assert((expr) == 0)
int copies = 0, synchronizes = 0;
int cudaEventSynchronize(int) { return 0; }
int cudaStreamSynchronize(int) { ++synchronizes; return 0; }
int cudaEventRecord(int, int) { return 0; }
int cudaMemcpyAsync(void* dst, const void* src, size_t bytes, int, int) {
    ++copies;
    std::memcpy(dst, src, bytes);
    return 0;
}
template<class... T> [[noreturn]] void die(const char*, T...) { std::abort(); }
constexpr size_t kPoolGrain = 8;
constexpr size_t kReadBuf = 64;
constexpr int kPagePos = 16;
constexpr size_t kPageBytes = 32;
constexpr int kSnapRows = 8;
size_t snap_bytes(bool) { return 16; }
struct Pool {
    static constexpr size_t kNone = SIZE_MAX;
    uint8_t* base = nullptr;
    size_t size = 0;
    std::map<size_t, size_t> free;
    static size_t round(size_t n) { return (n + kPoolGrain - 1) / kPoolGrain * kPoolGrain; }
    uint8_t* take(size_t);
    void give(uint8_t*, size_t);
    size_t take_off(size_t);
    bool take_at(size_t, size_t);
    void give_off(size_t, size_t);
    size_t largest() const;
};
''' + part(state, 'size_t Pool::take_off(', 'Store::~Store()') + r'''
uint64_t checksum(const uint8_t* p, size_t n) {
  uint64_t out = 0;
  for (size_t i = 0; i < n; ++i) out = out * 31 + p[i];
  return out;
}
void sum_block(const void* p, size_t bytes, uint64_t* sum, int) {
    *sum = checksum(static_cast<const uint8_t*>(p), Pool::round(bytes));
}
struct SpillFile {
    FILE* file = std::tmpfile();
    int fd_ = fileno(file);
    size_t size_ = 128;
    std::string path_ = "synthetic";
    uint8_t back[2][kReadBuf]{};
    uint8_t* rbuf_[2] = {back[0], back[1]};
    uint64_t sums[2][16]{};
    int rev_[2]{}, rnext_ = 0;
    bool direct_ = true;
    std::mutex mu_;
    uint64_t done_ = 76;
    bool is_written(uint64_t);
    uint64_t* sum(size_t off, bool read) { return &sums[read][off / kPoolGrain]; }
    void read(void*, size_t, size_t, cudaStream_t);
    bool try_read(void*, size_t, size_t, cudaStream_t);
    bool read_block(void*, size_t, size_t, cudaStream_t, bool);
    ~SpillFile() { std::fclose(file); }
};
''' + part(spill, 'bool io_all(', '}  // namespace') + part(
    spill, 'bool SpillFile::is_written(', 'void SpillFile::sync()') + part(
    spill, 'void SpillFile::read(', 'void SpillFile::writer()') + r'''
struct Store {
    struct Sess {
        std::vector<uint8_t*> pages{};
        int saved = 0;
        std::map<int, uint8_t*> snaps{};
        uint64_t used = 0;
        bool disk = false;
        std::vector<size_t> dpages{};
        std::map<int, size_t> dsnaps{};
        uint64_t dseq = 0;
    };
    std::map<int, Sess> sess;
    Pool pool;
    size_t bytes = 0;
    std::unique_ptr<SpillFile> disk = std::make_unique<SpillFile>();
    uint64_t clock = 99;
};
struct Model {
    struct Impl {
        uint8_t device[160]{};
        Store state;
        struct { int max_ctx = 1000; } opt;
        int* spec = nullptr;
        int st = 0;
        Store& store_of() { return state; }
        void store_release(uint8_t* ptr, size_t bytes) {
            if (ptr) {
                state.pool.give(ptr, bytes);
                state.bytes -= Pool::round(bytes);
            }
        }
        bool state_stage_spilled_snapshot(int, int, int);
    };
};
''' + part(state, 'bool Model::Impl::state_stage_spilled_snapshot(', '// Session h back from the disk tier') + r'''
int main() {
    Model::Impl model;
    auto& store = model.state;
    store.pool.base = model.device;
    store.pool.size = sizeof model.device;
    store.pool.free.emplace(0, store.pool.size);
    store.sess[9].pages.push_back(store.pool.take(kPageBytes)); // a resident fallback
    store.bytes += kPageBytes;
    auto& source = store.sess[1];
    source.disk = true;
    source.saved = 48;
    source.dpages = {0, 32, 64}; // page 3 lies beyond the needed prefix
    source.dsnaps = {{33, 96}};
    source.dseq = 77;
    source.used = 11;
    auto& disk = *store.disk;
    uint8_t bytes[128];
    for (size_t i = 0; i < sizeof bytes; ++i) bytes[i] = static_cast<uint8_t>(i + 1);
    assert(pwrite(disk.fd_, bytes, sizeof bytes, 0) == sizeof bytes);
    for (size_t pos : {0, 32, 64, 96})
        *disk.sum(pos, false) = checksum(bytes + pos, pos == 96 ? 16 : 32);

    const auto baseline_free = store.pool.free;
    auto clean = [&] {
        assert(store.pool.free == baseline_free && store.bytes == kPageBytes && store.sess.size() == 2);
        assert(source.disk && source.saved == 48 && source.dpages.size() == 3 && source.dsnaps.at(33) == 96 &&
               source.used == 11 && store.clock == 99 && store.sess.at(9).pages.size() == 1);
    };
    assert(!model.state_stage_spilled_snapshot(62, 2, 33)); // missing source
    assert(!model.state_stage_spilled_snapshot(1, 9, 33)); // never alter an incumbent handle
    assert(!model.state_stage_spilled_snapshot(1, 2, 32)); // exact odd snapshot key is required
    source.saved = 31;
    assert(!model.state_stage_spilled_snapshot(1, 2, 33));
    source.saved = 48;
    clean();
    assert(copies == 0);
    assert(disk.is_written(0));
    assert(!model.state_stage_spilled_snapshot(1, 2, 33)); // writer is still busy
    assert(copies == 0 && synchronizes == 0);
    clean();
    {
        const std::lock_guard<std::mutex> lock(disk.mu_);
        disk.done_ = source.dseq;
    }

    // Fragmented pool leaves aggregate room for all blocks, but not a window after allocating pages.
    const std::map<size_t, size_t> fragmented{{32,32},{72,32},{112,8},{128,8}};
    store.pool.free = fragmented;
    store.bytes = store.pool.size - 80;
    assert(!model.state_stage_spilled_snapshot(1, 2, 33));
    assert(store.pool.free == fragmented && store.bytes == store.pool.size - 80);
    store.pool.free = baseline_free;
    store.bytes = kPageBytes;
    clean();
    assert(copies == 0);

    // Partial pread and EOF on the window after two successful prefix copies.
    assert(ftruncate(disk.fd_, 104) == 0);
    assert(!model.state_stage_spilled_snapshot(1, 2, 33));
    assert(copies == 2 && synchronizes == 1);
    clean();

    assert(pwrite(disk.fd_, bytes, sizeof bytes, 0) == sizeof bytes);
    *disk.sum(96, false) ^= 1;
    assert(!model.state_stage_spilled_snapshot(1, 2, 33));
    assert(copies == 5 && synchronizes == 2);
    clean();
    *disk.sum(96, false) ^= 1;
    *disk.sum(64, false) ^= 1;  // irrelevant trailing page is never read

    assert(!disk.try_read(model.device, 16, SIZE_MAX, model.st));
    assert(!disk.try_read(model.device, 16, 1, model.st));
    assert(model.state_stage_spilled_snapshot(1, 2, 33));
    assert(copies == 8 && synchronizes == 3);
    auto& staged = store.sess.at(2);
    assert(staged.saved == 32 && staged.pages.size() == 2 && staged.snaps.size() == 1 &&
           !staged.disk && staged.used == 99);
    assert(std::memcmp(staged.pages[0], bytes, 32) == 0);
    assert(std::memcmp(staged.pages[1], bytes + 32, 32) == 0);
    assert(std::memcmp(staged.snaps.at(33), bytes + 96, 16) == 0);
    for (auto* page : staged.pages) model.store_release(page, kPageBytes);
    model.store_release(staged.snaps.at(33), 16);
    store.sess.erase(2);
    clean();
    std::puts("spilled snapshot staging: odd window, exact prefix pages, missing state, pending write, pool pressure, short read, and checksum refusal passed");
}
'''
with tempfile.TemporaryDirectory(prefix='pico-stage-spill-') as work:
    fixture = Path(work) / 'fixture.cpp'
    binary = Path(work) / 'fixture'
    fixture.write_text(test)
    subprocess.run(['nice', '-n', '19', 'c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    str(fixture), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
