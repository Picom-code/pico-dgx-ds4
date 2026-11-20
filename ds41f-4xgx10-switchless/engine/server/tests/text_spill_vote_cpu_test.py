"""Compile the ordinary text trial's actual collective decision against four synthetic ranks."""
from pathlib import Path
import subprocess
import tempfile

engine = Path(__file__).resolve().parents[2]
cmd = (engine / "server/cmd.h").read_text()
start = cmd.index("template<class Model, class Ring>\nbool stage_text_checkpoint")
vote = cmd[cmd.index("template<class Ring>\nbool text_checkpoint_vote"):cmd.index("inline bool text_checkpoint_ready")]
body = vote + cmd[start:cmd.index("model::Ctl to_ctl", start)]
if "stage_text_checkpoint(m, ring, c);" not in (engine / "server/cmd.cpp").read_text():
    raise RuntimeError("follower command is missing")
ring = r"""
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
"""
program = """
#include <algorithm>
#include <cstdint>
#include <vector>
#include <array>
#include <barrier>
#include <cassert>
#include <cstring>
#include <map>
#include <set>
#include <thread>
""" + ring + """
struct Cmd { int handle, next, pos0; };
struct Model {
  struct Info { bool on_disk; int saved; std::vector<int> snaps; };
  int refuse = 0, attempts = 0;
  std::map<int,Info> handles{{7,{true,8192,{8193}}},{11,{false,1024,{1024}}}};
  std::vector<int> state_sessions() {
    std::vector<int> out;
    for (auto& entry : handles) out.push_back(entry.first);
    return out;
  }
  Info state_info(int h) {
    auto it = handles.find(h);
    return it == handles.end() ? Info{false,0,{}} : it->second;
  }
  bool state_stage_spilled_snapshot(int src, int out, int snap) {
    assert(src==7 && snap==8193 && !handles.contains(out));
    ++attempts;
    if (refuse==1) return false;
    handles[out] = {false, refuse==2 ? 4096 : 8192,{snap}};
    return true;
  }
  void state_free(int h) { assert(h==19 && handles.erase(h)==1); }
};
""" + body + """
int main() {
  for (int mode : {0,1,2,3,4,5}) {
    Collective all;
    std::array<std::thread,4> threads;
    for (int rank=0;rank<4;++rank) threads[rank]=std::thread([&,rank]{
      Ring ring{all,rank}; Model m;
      if (rank==2) {
        if (mode==1 || mode==2) m.refuse=mode;
        if (mode==3) m.handles[19]={false,888,{888}};
        if (mode==4) m.handles[7].snaps.clear();
        if (mode==5) m.handles[7].on_disk=false;
      }
      assert(stage_text_checkpoint(m,ring,Cmd{7,19,8193})==(mode==0));
      assert(m.state_info(11).saved==1024 && m.state_info(7).saved==8192);
      assert(m.attempts==(mode<=2 ? 1 : 0));
      if (!mode) assert(m.state_info(19).saved==8192);
      else if (mode==3 && rank==2) assert(m.state_info(19).saved==888);
      else assert(!m.handles.contains(19));
    });
    for(auto& thread : threads) thread.join();
  }
}
"""
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "vote.cpp"
    path.write_text(program)
    exe = path.with_suffix("")
    subprocess.run(["g++","-std=c++20","-O1","-pthread","-Wall","-Wextra","-Werror",
                    "-I",str(engine.resolve()),str(path),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=15)
print("All four simulated ranks agreed in six staged-snapshot cases.")
