// CPU-only: g++ -std=c++20 -O2 -Wall -Wextra -Werror durable_image_index_test.cpp -o /tmp/durable_image_index_test
#include "../durable_image_index.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

using namespace srv::durable_image;

namespace {

struct TmpDir {
  char path[40] = "/tmp/durable-image-index-XXXXXX";
  TmpDir() { assert(mkdtemp(path)); }
  ~TmpDir() { std::filesystem::remove_all(path); }
  std::string file(std::string_view name) const { return std::string(path) + "/" + std::string(name); }
};

std::string hex(const Digest& digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string value;
  for (uint8_t byte : digest) {
    value += digits[byte >> 4];
    value += digits[byte & 15];
  }
  return value;
}

struct Fixture {
  std::vector<int32_t> prompt = std::vector<int32_t>(140, 42);
  std::vector<fe::ImageSpan> spans = {{1, 3, 7}, {7, 3, 8}};
  std::vector<uint16_t> rows = std::vector<uint16_t>(6 * fe::kImageDim, 0x1234);
  Fixture() { for (auto span : spans) std::fill_n(prompt.begin() + span.start, span.rows, fe::kImageTokenId); }
  Images images() const { return identify(prompt, spans, rows).value(); }
};

void identity() {
  assert(hex(sha256(nullptr, 0)) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  assert(hex(sha256("abc", 3)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const std::array<uint8_t, 56> zeros{};
  assert(hex(sha256(zeros.data(), zeros.size())) == "d4817aa5497628e7c77e6b606107042bbba3130888c5f47a375e6179be789fbb");
  Fixture f;
  const auto images = f.images();
  assert(hex(images[0].digest) == "3f3c5d84fdb9963004075b91e9fad08cde277a96f03fc705958b49c5993fd9a2");
  f.spans[0].hash++;
  assert(f.images() == images);
  f.rows[0] ^= 0x0100;
  assert(f.images()[0].digest != images[0].digest && f.images()[1] == images[1]);
  f.prompt[1] = 7;
  assert(!identify(f.prompt, f.spans, f.rows));
  f.prompt[1] = fe::kImageTokenId;
  f.rows.pop_back();
  assert(!identify(f.prompt, f.spans, f.rows));
}

void roundtrip() {
  TmpDir temp;
  const Limits limits{4, 10000, 60};
  Fixture f;
  const auto images = f.images();
  {
    Index index;
    assert(index.open(temp.path, "model A", limits));
    Index second_writer;
    assert(!second_writer.open(temp.path, "model A", limits));
    assert(!index.publish(1, f.prompt, images, 2, 200, 20, 2, false)); // cuts the first image
    assert(!index.publish(1, f.prompt, images, 8, 200, 20, 8, false)); // cuts the second image
    assert(!index.publish(1, f.prompt, images, 6, 200, 20, 8, false)); // mismatched snapshot
    auto next = f;
    next.prompt[6] = fe::kImageTokenId;
    next.prompt[9] = 42;
    next.spans[1].start = 6;
    assert(!index.publish(1, next.prompt, next.images(), 6, 200, 20, 7, false)); // odd snapshot sees next image
    assert(index.publish(1, f.prompt, images, 6, 200, 20, 7, false));
    assert(!index.publish(1, f.prompt, images, 12, 400, 21, 12, false));
    assert(index.publish(2, f.prompt, images, 12, 300, 21, 12, true));
    assert(index.size() == 2 && index.bytes() == 2 * 112 + 4 * (6 + 12) + 3 * 40 + 500);
    assert(std::filesystem::file_size(temp.file("i-0000000000000001.idx")) == 112 + 6 * 4 + 40);
  }
  Index mismatched;
  assert(!mismatched.open(temp.path, "model B", limits));
  assert(mismatched.take_evicted().empty());
  Index index;
  assert(index.open(temp.path, "model A", limits));
  assert(index.ids() == (std::vector<CheckpointId>{1, 2}));
  const auto* found = index.find(f.prompt, images, 25, true);
  assert(found && found->id == 2 && found->checkpoint_bytes == 300 && found->encoder_only);
  assert(found->tokens.size() == 12 && found->snap == 12 && found->images == images);
  found = index.find(f.prompt, images, 25, true, {2});
  assert(found && found->id == 1 && index.contains(2));
  assert(!index.find(f.prompt, images, 25, true, {1, 2}));
  found = index.find(f.prompt, images, 25, false);
  assert(found && found->id == 1 && found->snap == 7 && found->tokens.size() == 6);
  f.rows.back() ^= 1;
  assert(index.find(f.prompt, f.images(), 25, true)->id == 1);
  f.rows[0] ^= 1;
  assert(!index.find(f.prompt, f.images(), 25, true));
  f.prompt[0]++;
  assert(!index.find(f.prompt, images, 25, true));
  f.prompt[0]--;
  f.prompt.resize(139);
  assert(index.find(f.prompt, images, 25, true)->id == 1);
  f.prompt.resize(6);
  assert(!index.find(f.prompt, {images[0]}, 25, true));
  index.prune(80);
  assert(!index.contains(1) && index.contains(2) && index.take_evicted() == std::vector<CheckpointId>{1});
  assert(index.ids() == std::vector<CheckpointId>{2});
  index.prune(81);
  assert(index.size() == 0 && index.take_evicted() == std::vector<CheckpointId>{2});
}

void budgets() {
  TmpDir temp;
  Fixture f;
  Index index;
  const auto images = f.images();
  assert(index.open(temp.path, "scope", {2, 800, 60}));
  assert(index.publish(5, f.prompt, images, 6, 200, 20, 6, false));
  assert(index.publish(6, f.prompt, images, 6, 200, 21, 6, false));
  assert(!index.room_for(200, 6, 1));
  assert(index.publish(7, f.prompt, images, 6, 200, 22, 6, false));
  assert(index.take_evicted() == std::vector<CheckpointId>{5});
  assert(index.evict_oldest());
  assert(index.take_evicted() == std::vector<CheckpointId>{6});
  assert(index.publish(8, f.prompt, images, 6, 501, 23, 6, false));
  assert(index.take_evicted() == std::vector<CheckpointId>{7});
  assert(!index.publish(9, f.prompt, images, 6, 800, 24, 6, false));
  assert(index.contains(8));
}

void private_and_corrupt_files() {
  TmpDir temp;
  const auto path = temp.file("i-0000000000000001.idx");
  const Limits limits{4, 10000, 60};
  Fixture f;
  assert(chmod(temp.path, 0755) == 0);
  Index index;
  assert(!index.open(temp.path, "scope", limits));
  assert(chmod(temp.path, 0700) == 0);
  {
    Index writer;
    assert(writer.open(temp.path, "scope", limits));
    assert(writer.publish(1, f.prompt, f.images(), 6, 200, 20, 6, false));
  }
  assert(chmod(path.c_str(), 0644) == 0);
  assert(!index.open(temp.path, "scope", limits));
  assert(std::filesystem::exists(path) && index.ids().empty() && index.take_evicted().empty());
  assert(chmod(path.c_str(), 0600) == 0);
  assert(link(path.c_str(), temp.file(".pending").c_str()) == 0); // interrupted after link, before cleanup
  {
    Index reader;
    assert(reader.open(temp.path, "scope", limits) && reader.contains(1));
    assert(!std::filesystem::exists(temp.file(".pending")));
  }

  std::vector<uint8_t> record;
  {
    std::ifstream input(path, std::ios::binary);
    record.assign(std::istreambuf_iterator<char>(input), {});
  }
  assert(record.size() == 176);
  std::fill_n(record.begin() + 72, 4, uint8_t{255}); // invalid token count with a valid footer
  const auto checksum = sha256(record.data(), record.size() - sizeof(Digest));
  std::copy(checksum.begin(), checksum.end(), record.end() - sizeof(Digest));
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(record.data()), record.size());
    assert(output);
  }
  assert(index.open(temp.path, "scope", limits));
  assert(!index.contains(1) && !std::filesystem::exists(path));
  assert(index.take_evicted() == std::vector<CheckpointId>{1});
}

}  // namespace

int main() {
  identity();
  roundtrip();
  budgets();
  private_and_corrupt_files();
}
