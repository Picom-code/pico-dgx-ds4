#include "model/durable_state.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>

using model::detail::durable::Files;
using model::detail::durable::Header;
using model::detail::durable::Fd;

int main() {
  namespace disk = model::detail::durable;
  char dir[] = "/tmp/pico-durable-state-test.XXXXXX";
  assert(mkdtemp(dir));
  const std::string base = std::string(dir) + "/rank0";
  const uint8_t body[] = {1, 3, 5, 7};
  constexpr size_t file_bytes = sizeof(Header) + sizeof(uint64_t) + sizeof body;
  const auto write = [&](uint64_t generation, int fd) {
    Header hd;
    hd.run_id = 93;
    hd.rank = 0;
    hd.generation = generation;
    hd.file_bytes = file_bytes;
    hd.checksum = disk::header_checksum(hd);
    return disk::put(fd, &hd, sizeof hd) && disk::put_block(fd, body, sizeof body, disk::block_seed(generation, 1, 0));
  };
  const auto check = [&](Files& files, uint64_t generation) {
    return files.load(generation, [&](int fd, uint64_t bytes) {
      Header hd;
      uint8_t data[sizeof body];
      return bytes == file_bytes && disk::transfer(fd, &hd, sizeof hd, false) &&
             hd.checksum == disk::header_checksum(hd) && hd.run_id == files.run_id() &&
             hd.generation == generation && disk::get_block(fd, data, sizeof data, disk::block_seed(generation, 1, 0)) &&
             std::memcmp(data, body, sizeof body) == 0;
    });
  };

  Files first;
  first.configure(base, file_bytes, 93, 0);
  assert(first.save(10, file_bytes, [&](int fd) { return write(10, fd); }));
  assert(!first.save(10, file_bytes, [&](int fd) { return write(11, fd); }));
  assert(!first.save(11, file_bytes, [&](int fd) { return write(11, fd); }));

  Files restarted;
  restarted.configure(base, file_bytes, 93, 0);
  assert(check(restarted, 10));
  assert(!check(restarted, 11));
  assert(!restarted.load(10, [&](int fd, uint64_t) {
    Header hd;
    uint8_t data[sizeof body];
    return disk::transfer(fd, &hd, sizeof hd, false) &&
           disk::get_block(fd, data, sizeof data, disk::block_seed(10, 1, 1));
  }));
  Files incompatible;
  incompatible.configure(base, file_bytes, 94, 0);
  assert(!check(incompatible, 10));
  Files no_capacity;
  no_capacity.configure(base, file_bytes, 93, UINT64_MAX);
  assert(!no_capacity.save(20, file_bytes, [&](int fd) { return write(20, fd); }));

  {
    Fd file(open((base + ".checkpoint.10").c_str(), O_WRONLY | O_NOFOLLOW));
    const uint8_t changed = 0x80;
    assert(file.value >= 0 && pwrite(file.value, &changed, 1, file_bytes - 1) == 1);
  }
  assert(!check(restarted, 10));
  restarted.drop(10);
  assert(!check(restarted, 10));

  assert(!restarted.save(12, file_bytes, [&](int fd) {
    return disk::put(fd, body, sizeof body) && false;
  }));
  assert(!std::filesystem::exists(base + ".checkpoint.new"));
  assert(!check(restarted, 12));
  assert(symlink("/dev/null", (base + ".checkpoint.13").c_str()) == 0);
  assert(!check(restarted, 13));
  assert(!restarted.save(13, file_bytes, [&](int fd) { return write(13, fd); }));
  assert(std::filesystem::is_symlink(base + ".checkpoint.13"));
  restarted.drop(13);

  assert(restarted.save(14, file_bytes, [&](int fd) { return write(14, fd); }));
  assert(chmod(dir, 0755) == 0);
  assert(!check(restarted, 14));
  assert(chmod(dir, 0700) == 0);
  assert(check(restarted, 14));
  Files reconcile;
  reconcile.configure(base, 8 * file_bytes, 93, 0);
  assert(reconcile.save(15, file_bytes, [&](int fd) { return write(15, fd); }));
  const auto touch = [](const std::string& name, mode_t mode) {
    Fd fd(open(name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, mode));
    assert(fd.value >= 0);
  };
  touch(base + ".checkpoint.new", 0600);
  touch(base + ".checkpoint.015", 0600);
  touch(base + ".checkpoint.18446744073709551616", 0600);
  touch(base + ".checkpoint.16", 0644);
  touch(std::string(dir) + "/rank1.checkpoint.15", 0600);
  assert(symlink("/dev/null", (base + ".checkpoint.17").c_str()) == 0);
  assert(link((base + ".checkpoint.15").c_str(), (base + ".checkpoint.18").c_str()) == 0);
  assert(!reconcile.prune({0}));
  assert(!reconcile.prune({14, 14}));
  assert(std::filesystem::exists(base + ".checkpoint.new"));
  assert(reconcile.prune({14}));
  assert(check(reconcile, 14));
  assert(std::filesystem::exists(base + ".checkpoint.15"));  // hard-linked objects stay
  assert(std::filesystem::exists(base + ".checkpoint.18"));
  assert(unlink((base + ".checkpoint.18").c_str()) == 0);
  assert(reconcile.prune({14}));
  assert(!check(reconcile, 15));
  assert(!std::filesystem::exists(base + ".checkpoint.new"));
  assert(std::filesystem::exists(base + ".checkpoint.015"));
  assert(std::filesystem::exists(base + ".checkpoint.18446744073709551616"));
  assert(std::filesystem::exists(base + ".checkpoint.16"));
  assert(std::filesystem::exists(std::string(dir) + "/rank1.checkpoint.15"));
  assert(std::filesystem::is_symlink(base + ".checkpoint.17"));
  assert(reconcile.prune({}));
  assert(!check(reconcile, 14));
  std::filesystem::remove_all(dir);
}
