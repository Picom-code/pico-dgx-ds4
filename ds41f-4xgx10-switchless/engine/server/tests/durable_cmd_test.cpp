// CPU-only regression of the production all-rank vote and boot config helpers.
#include "server/durable_cmd.h"
#include "server/durable_config.h"
#include <array>
#include <barrier>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <thread>
#include <vector>
using namespace srv;
struct Collective {
  std::barrier<> barrier{4};
  std::array<std::array<unsigned char, 256>, 4> input{};
};
struct Ring {
  Collective& all; int rank;
  void allgather_host(const void* mine, void* out, size_t size) {
    assert(size <= all.input[rank].size());
    std::memcpy(all.input[rank].data(), mine, size);
    all.barrier.arrive_and_wait();
    for (size_t i = 0; i < 4; ++i) std::memcpy(static_cast<unsigned char*>(out) + i * size, all.input[i].data(), size);
    all.barrier.arrive_and_wait();
  }
};
struct Model {
  struct Info { int saved = 0; std::vector<int> snaps; };
  std::map<int, Info> handles;
  std::set<uint64_t> files;
  bool fail_save = false, fail_load = false, bad_snapshot = false;
  int saves = 0, loads = 0, frees = 0, drops = 0, configured = 0, received_snap = 0;
  bool encoder_only = false; std::string base; uint64_t run_id = 0; size_t bytes = 0;
  Info state_info(int h) { const auto it = handles.find(h); return it == handles.end() ? Info{} : it->second; }
  std::vector<int> state_sessions() { std::vector<int> out; for (const auto& x : handles) out.push_back(x.first); return out; }
  bool state_durable_save(int, uint64_t generation, int snap, bool enc) {
    ++saves; received_snap = snap; encoder_only = enc;
    if (fail_save || files.count(generation)) return false;
    files.insert(generation); return true;
  }
  bool state_durable_load(int h, uint64_t generation, int snap, bool enc) {
    received_snap = snap; encoder_only = enc;
    ++loads;
    if (fail_load || !files.count(generation)) return false;
    handles[h] = {128, bad_snapshot ? std::vector<int>{126} : std::vector<int>{128}}; return true;
  }
  void state_free(int h) { ++frees; handles.erase(h); }
  void state_durable_drop(uint64_t generation) { ++drops; files.erase(generation); }
  bool state_durable_prune(const std::vector<uint64_t>& keep) {
    for (auto it = files.begin(); it != files.end();) { if (std::find(keep.begin(), keep.end(), *it) == keep.end()) it = files.erase(it); else ++it; }
    return true;
  }
  void state_durable_config(const std::string& p, size_t n, uint64_t id) { ++configured; base = p; bytes = n; run_id = id; }
};
struct Group {
  std::array<Model, 4> model;
  std::array<bool, 4> result{};
  void run(int op, int h, uint64_t generation, int snap = 128, bool encoder = false) {
    Collective all;
    std::array<std::thread, 4> threads;
    for (int rank = 0; rank < 4; ++rank) threads[rank] = std::thread([&, rank] {
      Ring ring{all, rank}; result[rank] = durable::run(model[rank], ring, op, h, generation, snap, encoder);
    });
    for (auto& thread : threads) thread.join();
  }
  void expect(bool value) { for (const bool got : result) assert(got == value); }
  void warm(int h = 7) { for (auto& m : model) m.handles[h] = {128, {128}}; }
};
static void save_votes() {
  Group g; g.warm(); g.run(durable::save, 7, 11, 128, true); g.expect(true);
  for (const auto& m : g.model) assert(m.files.count(11) && m.saves == 1 && m.received_snap == 128 && m.encoder_only);
  Group odd; odd.warm(); for (auto& m : odd.model) m.handles[7].snaps = {129};
  odd.run(durable::save, 7, 12, 129); odd.expect(true);
  Group collision; collision.warm(); collision.model[2].files.insert(19);
  collision.run(durable::save, 7, 19); collision.expect(false);
  for (int i = 0; i < 4; ++i) {
    assert(collision.model[i].handles.count(7));
    assert(collision.model[i].files.count(19) == (i == 2));
    assert(collision.model[i].drops == (i == 2 ? 0 : 1));
  }
  Group missing; missing.warm(); missing.model[1].handles[7].snaps.clear();
  missing.run(durable::save, 7, 21); missing.expect(false);
  for (const auto& m : missing.model) assert(m.saves == 0 && m.drops == 0);
}
static void load_votes() {
  Group g; for (auto& m : g.model) m.files.insert(31);
  g.run(durable::load, 9, 31, 128, true); g.expect(true);
  for (const auto& m : g.model) assert(m.handles.count(9) && m.frees == 0);
  Group partial; for (auto& m : partial.model) m.files.insert(33);
  partial.model[3].bad_snapshot = true; partial.run(durable::load, 9, 33); partial.expect(false);
  for (const auto& m : partial.model) assert(m.handles.empty() && m.frees == 1 && m.files.count(33));
  Group failed; for (auto& m : failed.model) m.files.insert(35);
  failed.model[2].fail_load = true; failed.run(durable::load, 9, 35); failed.expect(false);
  for (const auto& m : failed.model) assert(m.handles.empty() && m.frees == 1 && m.files.count(35));
  Group occupied; for (auto& m : occupied.model) m.files.insert(37);
  occupied.model[1].handles[9] = {64, {64}};
  occupied.run(durable::load, 9, 37); occupied.expect(false);
  for (int i = 0; i < 4; ++i) assert(occupied.model[i].loads == 0 && occupied.model[i].frees == 0 && occupied.model[i].handles.count(9) == (i == 1));
}
static void refusal_and_drop() {
  Group g; g.warm(); g.run(durable::save, 7, 0); g.expect(false);
  g.run(durable::save, 7, 41, 127); g.expect(false);
  for (const auto& m : g.model) assert(m.saves == 0);
  for (auto& m : g.model) m.files.insert(41);
  g.run(durable::drop, -1, 41, 0); g.expect(true);
  for (const auto& m : g.model) assert(m.files.empty() && m.handles.count(7) && m.frees == 0);
}
static void prune_votes() {
  Group g; for (auto& m : g.model) m.files = {11, 12, 13};
  const auto run = [&](const std::vector<uint64_t>& ids) {
    Collective all; std::array<std::thread, 4> threads;
    for (int rank = 0; rank < 4; ++rank) threads[rank] = std::thread([&, rank] { Ring ring{all, rank}; g.result[rank] = durable::prune(g.model[rank], ring, ids); });
    for (auto& thread : threads) thread.join();
  };
  run({11, 13}); g.expect(true); for (const auto& m : g.model) assert((m.files == std::set<uint64_t>{11, 13}));
  run({11, 11}); g.expect(false); for (const auto& m : g.model) assert(m.files.size() == 2);
  run({0}); g.expect(false); for (const auto& m : g.model) assert(m.files.size() == 2);
  run(std::vector<uint64_t>(129, 11)); g.expect(false); for (const auto& m : g.model) assert(m.files.size() == 2);
  run({}); g.expect(true); for (const auto& m : g.model) assert(m.files.empty());
}
static void config() {
  for (const char* name : {"DSGB10_ORDINARY_IMAGE_PREFIX_CACHE", "DSGB10_ORDINARY_IMAGE_DURABLE", "DSGB10_ORDINARY_IMAGE_DURABLE_BYTES", "DSGB10_ORDINARY_IMAGE_ASSETS_EPOCH"}) unsetenv(name);
  Collective disabled; std::array<Model, 4> inert; std::array<std::thread, 4> off;
  for (int rank = 0; rank < 4; ++rank) off[rank] = std::thread([&, rank] {
    Ring ring{disabled, rank}; const auto result = durable::configure(inert[rank], ring, rank, "", "", []() -> uint64_t { assert(false); return 0; }, [] { assert(false); return false; });
    assert(result.bytes == 0 && result.directory.empty() && result.namespace_id.empty());
  });
  for (auto& thread : off) thread.join();
  for (const auto& model : inert) assert(model.configured == 0);
  const char* epoch = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  durable::Input input;
  assert(durable::parse(input, nullptr, nullptr, nullptr, nullptr));
  assert(!input.enabled && !input.bytes);
  assert(!durable::parse(input, "1", nullptr, "1", epoch));
  assert(!durable::parse(input, "0", "1", "1", epoch));
  assert(!durable::parse(input, "1", "1", "18446744073709551616", epoch));
  assert(!durable::parse(input, "1", "1", "-1", epoch));
  assert(!durable::parse(input, "1", "1", "0", epoch));
  assert(!durable::parse(input, "1", "1", "100", "abc"));
  assert(durable::parse(input, "1", "1", "1024", epoch) && input.bytes == 1024);
  setenv("DSGB10_ORDINARY_IMAGE_PREFIX_CACHE", "1", 1); setenv("DSGB10_ORDINARY_IMAGE_DURABLE", "1", 1);
  setenv("DSGB10_ORDINARY_IMAGE_DURABLE_BYTES", "1024", 1); setenv("DSGB10_ORDINARY_IMAGE_ASSETS_EPOCH", epoch, 1);
  char directory[] = "/tmp/pico-durable-cmd-XXXXXX"; assert(mkdtemp(directory));
  Collective all; std::array<Model, 4> models; std::array<durable::Config, 4> output;
  std::array<std::thread, 4> threads;
  for (int rank = 0; rank < 4; ++rank) threads[rank] = std::thread([&, rank] {
    Ring ring{all, rank}; output[rank] = durable::configure(models[rank], ring, rank, directory, ".test", [rank] { return uint64_t(100 + rank); }, [] { return true; });
  });
  for (auto& thread : threads) thread.join();
  for (int rank = 0; rank < 4; ++rank) {
    assert(output[rank].namespace_id == output[0].namespace_id && output[rank].bytes == 1024);
    assert(models[rank].configured == 1 && models[rank].run_id == uint64_t(100 + rank));
    assert(models[rank].base.ends_with("/rank" + std::to_string(rank)));
  }
  struct stat st{}; assert(!lstat(output[0].directory.c_str(), &st) && (st.st_mode & 07777) == 0700);
  std::filesystem::remove_all(directory);
  unsetenv("DSGB10_ORDINARY_IMAGE_DURABLE"); unsetenv("DSGB10_ORDINARY_IMAGE_DURABLE_BYTES"); unsetenv("DSGB10_ORDINARY_IMAGE_ASSETS_EPOCH"); unsetenv("DSGB10_ORDINARY_IMAGE_PREFIX_CACHE");
}
int main() { save_votes(); load_votes(); refusal_and_drop(); prune_votes(); config(); }
