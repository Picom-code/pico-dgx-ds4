// The decode step's L2 prefetch (model::Options::l2pf; docs/engine.md, "A paced L2 prefetch in the step's windows"):
// while a collective, or the attention, leaves DRAM idle, one warp on a side stream issues
// cp.async.bulk.prefetch.L2 of the weights the step reads next, 8 KiB chunks at a token bucket's pace
// (TensorFold's paced one-block prefetch, as sudoingX/dgx-spark-deepseek-v41's exl3_l2pf.cu runs it;
// Apache-2.0, this code is ours). It writes nothing the model reads, so no output changes.
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common/check.h"
#include "kernels/dsk.h"

namespace dsk {
namespace {

constexpr uint32_t kChunk = 8192;
constexpr int kRanges = 16;
constexpr float kGbps = 200;  // the pace (micro/l2_retention's lat test: no fill cost at ~175 in small bursts)

struct Queue {  // a layer's planned weights in the order the step reads them
  const uint8_t* p[kRanges];
  uint32_t n[kRanges];    // bytes (multiples of 16)
  uint32_t end[kRanges];  // chunks up to and including range i
  int count;
};

__device__ __forceinline__ unsigned long long ld_volatile(const unsigned long long* p) {
  return *(volatile const unsigned long long*)p;
}

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// One warp: from chunk max(first, where the layer's earlier windows ended) of the queue, up to budget chunks, one
// due every ns_chunk ns (a poll issues those due, up to 32). It ends early once the window's collective has arrived
// (the next kernel is about to start): *arrived changed since the start, or is ahead of *finished (the collectives
// run one at a time, so the one in flight has arrived, perhaps before this started). It leaves where it ended in
// *state.
__global__ void __launch_bounds__(32) k_l2pf(const __grid_constant__ Queue q, int layer, uint32_t first,
                                             uint32_t budget, uint32_t ns_chunk, const unsigned long long* arrived,
                                             const unsigned long long* finished, L2pfState* state) {
  const int lane = threadIdx.x;
  uint32_t c0 = 0;
  unsigned long long a0 = 0;
  if (lane == 0) {
    if (state->layer == layer) c0 = state->chunk;
    if (arrived) a0 = ld_volatile(arrived);
  }
  c0 = max(__shfl_sync(~0u, c0, 0), first);
  const uint32_t total = q.count ? q.end[q.count - 1] : 0, last = min(total, c0 + budget);
  const unsigned long long t0 = __shfl_sync(~0u, gtimer(), 0);
  uint32_t done = min(c0, last);
  while (done < last) {
    unsigned long long t = 0;
    int quit = 0;
    if (lane == 0) {
      t = gtimer();
      if (arrived) {
        const unsigned long long a = ld_volatile(arrived);
        quit = a != a0 || a > ld_volatile(finished);
      }
    }
    if (__shfl_sync(~0u, quit, 0)) break;
    t = __shfl_sync(~0u, t, 0);
    const unsigned long long by_now = c0 + (t - t0) / ns_chunk + 1;
    const uint32_t cap = min(last, done + 32), due = by_now < cap ? (uint32_t)by_now : cap;
    const uint32_t c = done + lane;
    if (c < due) {
      int i = 0;
      while (c >= q.end[i]) ++i;
      const uint32_t off = (c - (i ? q.end[i - 1] : 0)) * kChunk, left = q.n[i] - off;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(q.p[i] + off), "r"(min(left, kChunk))
                   : "memory");
    }
    if (due > done) done = due;
    else __nanosleep(64);
  }
  if (lane == 0) state->layer = layer, state->chunk = done;
}

// ---- DSGB10_L2PF, the plan and budgets from device memory -------------------------------
constexpr int kClasses = 4;  // 1 row, 2..8, 9..16, 17+
struct Tune {                // device memory, rewritten by l2pf_tune_poll()
  uint32_t custom[kClasses]; // 0: the stock plan and budgets for this class
  uint32_t plan[kClasses];   // bit i: target i
  float a[kClasses][4], b[kClasses][4];  // window w + 1: us = a + b (min(R, rcap) - 1)
  float w3i[kClasses];       // W3 on the indexer's layers (< 0: a and b)
  uint32_t upto[kClasses][4];  // window w + 1 prefetches no target after this one (default: the last)
  uint32_t rcap[kClasses];
  float gbps;
  uint32_t epoch;
  // pN=US: window N's warp goes on for US more AFTER its collective arrived, on the targets from qN on
  // (the kernel that waited for the collective then computes from its ring / L2: DRAM is idle again). 0: today's stop.
  float post[kClasses][4];
  uint32_t pfrom[kClasses][4];
  uint32_t prow[kClasses];  // pr=N: the post phases run in steps of at most N rows (default: every step of the class)
  // the draft pass's sites (l2pf_draft): us of prefetch at the pace; 0: nothing issued
  float dpf[kL2pfDraftSites];
};
struct Queue2 {  // every target's ranges, in the order the step reads them
  const uint8_t* p[kRanges * 2];
  uint32_t n[kRanges * 2];
  uint8_t tgt[kRanges * 2];
  int count;
};

__global__ void __launch_bounds__(32) k_l2pf2(const __grid_constant__ Queue2 q, int layer, int window, int R, int idx,
                                              const unsigned long long* arrived, const unsigned long long* finished,
                                              L2pfState* state, const Tune* tune) {
  __shared__ uint32_t s_end[kRanges * 2];
  __shared__ uint32_t s_par[4];  // first, budget, ns_chunk, where the layer's earlier windows ended
  __shared__ uint32_t s_lim;     // the chunks up to the window's last allowed target
  __shared__ uint32_t s_post[2]; // the post phase's first chunk, and its budget in chunks (0: none)
  const int lane = threadIdx.x;
  unsigned long long a0 = 0;
  if (lane == 0) {
    const int cls = R == 1 ? 0 : R <= 8 ? 1 : R <= 16 ? 2 : 3;
    const bool cu = tune->custom[cls] != 0;
    const unsigned plan = cu ? tune->plan[cls]
                             : (R == 1 ? 1u << kL2pfWqb : 0u) | 1u << kL2pfWoa | 1u << kL2pfWob | 1u << kL2pfGate;
    const int from = window == 1 ? kL2pfWqa : window == 2 ? kL2pfWqb : window == 3 ? kL2pfWoa : kL2pfGate;
    const int upto = cu ? (int)tune->upto[cls][window - 1] : kL2pfTargets - 1;
    const int pfrom = cu ? (int)tune->pfrom[cls][window - 1] : 0;
    uint32_t c = 0, first = 0, lim = 0, pfirst = 0;
    for (int i = 0; i < q.count; ++i) {
      if (plan >> q.tgt[i] & 1) c += (q.n[i] + kChunk - 1) / kChunk;
      s_end[i] = c;
      if (q.tgt[i] < from) first = c;
      if (q.tgt[i] <= upto) lim = c;
      if (q.tgt[i] < pfrom) pfirst = c;
    }
    s_lim = lim;
    s_post[0] = pfirst;
    float us;
    if (cu) {
      const int r = min(R, (int)tune->rcap[cls]);
      us = window == 3 && idx && tune->w3i[cls] >= 0 ? tune->w3i[cls]
                                                     : tune->a[cls][window - 1] + tune->b[cls][window - 1] * (r - 1);
    } else {
      const int r = min(R, 8);
      us = window == 1 ? 18 + 5.1f * (r - 1) : window == 2 ? 18 : window == 3 ? (idx ? 30 : 11) : 14 + 1.2f * (r - 1);
    }
    const float gbps = cu && tune->gbps > 0 ? tune->gbps : kGbps;
    const uint32_t ns_chunk = (uint32_t)(kChunk / gbps + 0.5f);
    s_par[0] = first, s_par[1] = (uint32_t)(us * 1000 / ns_chunk), s_par[2] = ns_chunk;
    s_par[3] = state->layer == layer ? state->chunk : 0;
    const float pus = cu && arrived && R <= (int)tune->prow[cls] ? tune->post[cls][window - 1] : 0.f;
    s_post[1] = pus > 0 ? (uint32_t)(pus * 1000 / ns_chunk) : 0u;
    if (arrived) a0 = ld_volatile(arrived);
  }
  __syncwarp();
  const uint32_t first = s_par[0], budget = s_par[1], ns_chunk = s_par[2];
  const uint32_t c0 = max(s_par[3], first);
  const uint32_t total = s_lim, last = min(total, c0 + budget);
  const unsigned long long t0 = __shfl_sync(~0u, gtimer(), 0);
  uint32_t done = min(c0, last);
  int stopped = 0;
  while (done < last) {
    unsigned long long t = 0;
    int quit = 0;
    if (lane == 0) {
      t = gtimer();
      if (arrived) {
        const unsigned long long a = ld_volatile(arrived);
        quit = a != a0 || a > ld_volatile(finished);
      }
    }
    if (__shfl_sync(~0u, quit, 0)) {
      stopped = 1;
      break;
    }
    t = __shfl_sync(~0u, t, 0);
    const unsigned long long by_now = c0 + (t - t0) / ns_chunk + 1;
    const uint32_t cap = min(last, done + 32), due = by_now < cap ? (uint32_t)by_now : cap;
    const uint32_t c = done + lane;
    if (c < due) {
      int i = 0;
      while (c >= s_end[i]) ++i;
      const uint32_t off = (c - (i ? s_end[i - 1] : 0)) * kChunk, left = q.n[i] - off;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(q.p[i] + off), "r"(min(left, kChunk))
                   : "memory");
    }
    if (due > done) done = due;
    else __nanosleep(64);
  }
  // the post phase (pN > 0, a window with a collective). The budget may have run out before the
  // collective arrived: wait for it (at most 80 us from the start), so that the phase is timed from the arrival.
  const uint32_t pbudget = s_post[1];
  if (pbudget) {
    while (!stopped) {
      int quit = 0;
      if (lane == 0) {
        const unsigned long long a = ld_volatile(arrived);
        quit = a != a0 || a > ld_volatile(finished) || gtimer() - t0 > 80000ull;
      }
      if (__shfl_sync(~0u, quit, 0)) break;
      __nanosleep(64);
    }
    const uint32_t p0 = max(done, s_post[0]), plast = min(total, p0 + pbudget);
    const unsigned long long t1 = __shfl_sync(~0u, gtimer(), 0);
    uint32_t pd = min(p0, plast);
    while (pd < plast) {
      unsigned long long t = lane == 0 ? gtimer() : 0ull;
      t = __shfl_sync(~0u, t, 0);
      const unsigned long long by_now = p0 + (t - t1) / ns_chunk + 1;
      const uint32_t cap = min(plast, pd + 32), due = by_now < cap ? (uint32_t)by_now : cap;
      const uint32_t c = pd + lane;
      if (c < due) {
        int i = 0;
        while (c >= s_end[i]) ++i;
        const uint32_t off = (c - (i ? s_end[i - 1] : 0)) * kChunk, left = q.n[i] - off;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(q.p[i] + off), "r"(min(left, kChunk))
                     : "memory");
      }
      if (due > pd) pd = due;
      else __nanosleep(64);
    }
    if (pd > p0) done = pd;  // the later windows of the layer go on after the post phase's last chunk
  }
  if (lane == 0 && done > s_par[3]) state->layer = layer, state->chunk = done;
  else if (lane == 0 && state->layer != layer) state->layer = layer, state->chunk = done;
}

// the draft pass's prefetch (l2pf_draft): one warp, the site's budget (us at the pace) from device
// memory, the ranges in the order the pass reads them; no stop (the sites' windows hold no single collective).
__global__ void __launch_bounds__(32) k_l2pf_d(const __grid_constant__ Queue q, int site, const Tune* tune) {
  __shared__ uint32_t s_par[2];
  const int lane = threadIdx.x;
  if (lane == 0) {
    const float gbps = tune->gbps > 0 ? tune->gbps : kGbps;
    const uint32_t ns_chunk = (uint32_t)(kChunk / gbps + 0.5f);
    const float us = tune->dpf[site];
    s_par[0] = us > 0 ? (uint32_t)(us * 1000 / ns_chunk) : 0u, s_par[1] = ns_chunk;
  }
  __syncwarp();
  const uint32_t ns_chunk = s_par[1];
  const uint32_t total = q.count ? q.end[q.count - 1] : 0, last = min(total, s_par[0]);
  const unsigned long long t0 = __shfl_sync(~0u, gtimer(), 0);
  uint32_t done = 0;
  while (done < last) {
    unsigned long long t = lane == 0 ? gtimer() : 0ull;
    t = __shfl_sync(~0u, t, 0);
    const unsigned long long by_now = (t - t0) / ns_chunk + 1;
    const uint32_t cap = min(last, done + 32), due = by_now < cap ? (uint32_t)by_now : cap;
    const uint32_t c = done + lane;
    if (c < due) {
      int i = 0;
      while (c >= q.end[i]) ++i;
      const uint32_t off = (c - (i ? q.end[i - 1] : 0)) * kChunk, left = q.n[i] - off;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(q.p[i] + off), "r"(min(left, kChunk))
                   : "memory");
    }
    if (due > done) done = due;
    else __nanosleep(64);
  }
}

Tune* g_tune_d = nullptr;  // device copy (null: DSGB10_L2PF unset)
Tune g_tune_h{};           // the host's copy (which classes the spec names)
std::string g_tune_file;
long g_tune_stamp = -1;
int g_tune_rank = -1;
uint32_t g_tune_epoch = 0;

Tune parse_tune(const std::string& text) {
  Tune t{};
  static const char* kNames[kL2pfTargets] = {"wqa", "wqb", "woa", "wob", "gate", "sh13", "sh2"};
  for (int c = 0; c < kClasses; ++c) {
    t.rcap[c] = 8, t.w3i[c] = 30;
    for (int w = 0; w < 4; ++w) t.upto[c][w] = kL2pfTargets - 1;
    t.pfrom[c][0] = kL2pfWqb, t.pfrom[c][1] = kL2pfWoa, t.pfrom[c][2] = kL2pfWoa, t.pfrom[c][3] = kL2pfSh13;
    t.prow[c] = 1u << 20;
    t.a[c][0] = 18, t.b[c][0] = 5.1f, t.a[c][1] = 18, t.a[c][2] = 11, t.a[c][3] = 14, t.b[c][3] = 1.2f;
    t.plan[c] = (c == 0 ? 1u << kL2pfWqb : 0u) | 1u << kL2pfWoa | 1u << kL2pfWob | 1u << kL2pfGate;
  }
  size_t i = 0;
  while (i < text.size()) {
    size_t j = text.find_first_of(";/", i);  // '/' too: a ';' does not survive the launchers' shells
    if (j == std::string::npos) j = text.size();
    std::string item = text.substr(i, j - i);
    i = j + 1;
    while (!item.empty() && (item.back() == '\n' || item.back() == ' ' || item.back() == '\r')) item.pop_back();
    while (!item.empty() && (item[0] == ' ' || item[0] == '\n')) item.erase(0, 1);
    if (item.empty() || item == "off" || item == "0" || item == "stock") continue;
    if (item.rfind("pace=", 0) == 0) {
      t.gbps = (float)atof(item.c_str() + 5);
      continue;
    }
    if (item.rfind("d:", 0) == 0) {  // the draft pass's sites: d:d0=US,da=US,db=US,dh=US
      static const char* kSites[kL2pfDraftSites] = {"d0", "da", "db", "dh"};
      size_t a = 2;
      while (a < item.size()) {
        size_t b = item.find(',', a);
        if (b == std::string::npos) b = item.size();
        const std::string kv = item.substr(a, b - a);
        a = b + 1;
        const size_t eq = kv.find('=');
        int f = -1;
        for (int q = 0; q < kL2pfDraftSites; ++q)
          if (eq != std::string::npos && kv.substr(0, eq) == kSites[q]) f = q;
        if (f < 0) die("DSGB10_L2PF: '%s' in '%s' (d0, da, db, dh = us)", kv.c_str(), item.c_str());
        t.dpf[f] = (float)atof(kv.c_str() + eq + 1);
      }
      continue;
    }
    if (item.size() < 3 || item[0] != 'c' || item[1] < '0' || item[1] >= '0' + kClasses || item[2] != ':')
      die("DSGB10_L2PF: item '%s' (off, pace=GBPS, cN:key=value,... with N 0..%d)", item.c_str(), kClasses - 1);
    const int c = item[1] - '0';
    t.custom[c] = 1;
    size_t a = 3;
    while (a < item.size()) {
      size_t b = item.find(',', a);
      if (b == std::string::npos) b = item.size();
      const std::string kv = item.substr(a, b - a);
      a = b + 1;
      const size_t eq = kv.find('=');
      if (eq == std::string::npos) die("DSGB10_L2PF: '%s' in '%s'", kv.c_str(), item.c_str());
      const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
      if (k == "plan") {
        t.plan[c] = 0;
        size_t x = 0;
        while (x < v.size()) {
          size_t y = v.find('+', x);
          if (y == std::string::npos) y = v.size();
          const std::string nm = v.substr(x, y - x);
          x = y + 1;
          if (nm == "none" || nm.empty()) continue;
          int f = -1;
          for (int q = 0; q < kL2pfTargets; ++q)
            if (nm == kNames[q]) f = q;
          if (f < 0) die("DSGB10_L2PF: target '%s' (wqa, wqb, woa, wob, gate, sh13, sh2, none)", nm.c_str());
          t.plan[c] |= 1u << f;
        }
      } else if (k.size() == 2 && k[0] == 'w' && k[1] >= '1' && k[1] <= '4') {
        const int w = k[1] - '1';
        const size_t p = v.find('+');
        t.a[c][w] = (float)atof(v.c_str());
        t.b[c][w] = p == std::string::npos ? 0.f : (float)atof(v.c_str() + p + 1);
        if (w == 2) t.w3i[c] = -1;  // an explicit w3 also sets the indexer's layers unless w3i follows
      } else if (k.size() == 2 && k[0] == 'u' && k[1] >= '1' && k[1] <= '4') {
        int f = -1;
        for (int q = 0; q < kL2pfTargets; ++q)
          if (v == kNames[q]) f = q;
        if (f < 0) die("DSGB10_L2PF: %s=%s (a target name)", k.c_str(), v.c_str());
        t.upto[c][k[1] - '1'] = (uint32_t)f;
      } else if (k.size() == 2 && k[0] == 'p' && k[1] >= '1' && k[1] <= '4') {  // us after the arrival
        t.post[c][k[1] - '1'] = (float)atof(v.c_str());
      } else if (k.size() == 2 && k[0] == 'q' && k[1] >= '1' && k[1] <= '4') {  // the post phase's first target
        int f = -1;
        for (int q = 0; q < kL2pfTargets; ++q)
          if (v == kNames[q]) f = q;
        if (f < 0) die("DSGB10_L2PF: %s=%s (a target name)", k.c_str(), v.c_str());
        t.pfrom[c][k[1] - '1'] = (uint32_t)f;
      } else if (k == "pr") {  // the post phases only in steps of at most this many rows
        t.prow[c] = (uint32_t)std::max(1, atoi(v.c_str()));
      } else if (k == "w3i") {
        t.w3i[c] = (float)atof(v.c_str());
      } else if (k == "rcap") {
        t.rcap[c] = (uint32_t)std::max(1, atoi(v.c_str()));
      } else {
        die("DSGB10_L2PF: key '%s' (plan, w1..w4, u1..u4, p1..p4, q1..q4, pr, w3i, rcap)", k.c_str());
      }
    }
  }
  return t;
}

void push_tune(const std::string& text) {
  Tune t = parse_tune(text);
  t.epoch = ++g_tune_epoch;
  g_tune_h = t;
  CUDA_CHECK(cudaMemcpy(g_tune_d, &t, sizeof t, cudaMemcpyHostToDevice));
  if (g_tune_rank == 0) {
    std::string one = text;
    for (char& ch : one)
      if (ch == '\n') ch = ' ';
    std::printf("l2pf: DSGB10_L2PF plan %u: %s\n", t.epoch, one.empty() ? "off (stock)" : one.c_str());
    std::fflush(stdout);
  }
}

bool read_tune_file(std::string& text, long& stamp) {
  struct stat sb;
  if (stat(g_tune_file.c_str(), &sb) != 0) {
    stamp = 0, text.clear();
    return true;  // no file: stock
  }
  stamp = (long)sb.st_mtim.tv_sec * 1000000L + sb.st_mtim.tv_nsec / 1000 + sb.st_size;
  FILE* f = std::fopen(g_tune_file.c_str(), "r");
  if (!f) return false;
  char buf[2048];
  const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
  std::fclose(f);
  text.assign(buf, n);
  return true;
}

}  // namespace

bool l2pf_tuned() { return g_tune_d != nullptr; }

void l2pf_tune_env(int rank) {
  const char* e = std::getenv("DSGB10_L2PF");
  if (!e || !*e || g_tune_d) return;
  g_tune_rank = rank;
  CUDA_CHECK(cudaMalloc(&g_tune_d, sizeof(Tune)));
  std::string text = e;
  if (*e == '@') {
    g_tune_file = e + 1;
    if (g_tune_file.rfind("~/", 0) == 0 && std::getenv("HOME"))
      g_tune_file = std::string(std::getenv("HOME")) + g_tune_file.substr(1);
    if (!read_tune_file(text, g_tune_stamp)) text.clear();
  }
  push_tune(text);
}

void l2pf_tune_poll() {
  if (!g_tune_d || g_tune_file.empty()) return;
  std::string text;
  long stamp = 0;
  if (!read_tune_file(text, stamp) || stamp == g_tune_stamp) return;
  g_tune_stamp = stamp;
  push_tune(text);
}

void l2pf_tune_tick() {
  static uint32_t n = 0;
  if (g_tune_d && (n++ & 7) == 0) l2pf_tune_poll();
}

// a draft-pass site's prefetch. Nothing is launched (false) unless DSGB10_L2PF is a file or names a
// budget for a draft site, so the default graph is today's.
bool l2pf_draft_on() {
  if (!g_tune_d) return false;
  if (!g_tune_file.empty()) return true;
  for (int i = 0; i < kL2pfDraftSites; ++i)
    if (g_tune_h.dpf[i] > 0) return true;
  return false;
}

bool l2pf_draft(const void* const* p, const size_t* n, int count, int site, cudaStream_t s) {
  if (!l2pf_draft_on()) return false;
  if (site < 0 || site >= kL2pfDraftSites) die("l2pf_draft: site %d", site);
  Queue q{};
  uint32_t c = 0;
  for (int i = 0; i < count; ++i) {
    const size_t b = n[i] & ~(size_t)15;
    if (!p[i] || b < 16) continue;
    if (q.count == kRanges) die("l2pf_draft: more than %d ranges", kRanges);
    if (b > 0xfffffff0ull) die("l2pf_draft: a range of %zu bytes", b);
    q.p[q.count] = static_cast<const uint8_t*>(p[i]);
    q.n[q.count] = (uint32_t)b;
    c += (uint32_t)((b + kChunk - 1) / kChunk);
    q.end[q.count++] = c;
  }
  if (!c) return false;
  k_l2pf_d<<<1, 32, 0, s>>>(q, site, g_tune_d);
  CUDA_CHECK(cudaGetLastError());
  return true;
}

void l2pf_window(const L2pfTarget (&t)[kL2pfTargets], int layer, int window, int R, const unsigned long long* arrived,
                 const unsigned long long* finished, L2pfState* state, cudaStream_t s) {
  if (!arrived != !finished) die("l2pf_window: arrived and finished go together");
  if (window < 1 || window > 4) die("l2pf_window: window %d", window);
  // DSGB10_L2PF: every target's ranges; the plan and budgets come from device memory. A fixed spec
  // (not @FILE) keeps today's kernel for the row classes it does not name.
  if (g_tune_d && (!g_tune_file.empty() || g_tune_h.custom[R == 1 ? 0 : R <= 8 ? 1 : R <= 16 ? 2 : 3])) {
    Queue2 q2{};
    for (int i = 0; i < kL2pfTargets; ++i)
      for (int j = 0; j < 4; ++j) {
        const size_t n = t[i].n[j] & ~(size_t)15;
        if (!t[i].p[j] || n < 16) continue;
        if (q2.count == kRanges * 2) die("l2pf_window: more than %d ranges", kRanges * 2);
        q2.p[q2.count] = static_cast<const uint8_t*>(t[i].p[j]);
        q2.n[q2.count] = (uint32_t)n;
        q2.tgt[q2.count++] = (uint8_t)i;
      }
    k_l2pf2<<<1, 32, 0, s>>>(q2, layer, window, R, t[kL2pfWqb].n[2] > 0, arrived, finished, state, g_tune_d);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  // the plan: wo_a, wo_b and the router gate; at one row wq_b too, whose persistent kernel gains from L2 (the
  // classic one, from two rows, does not)
  const unsigned plan = (R == 1 ? 1u << kL2pfWqb : 0u) | 1u << kL2pfWoa | 1u << kL2pfWob | 1u << kL2pfGate;
  static const int kFrom[5] = {0, kL2pfWqa, kL2pfWqb, kL2pfWoa, kL2pfGate};  // a window's first target not read
  Queue q{};
  uint32_t c = 0, first = 0;
  for (int i = 0; i < kL2pfTargets; ++i) {
    if (!(plan >> i & 1)) continue;
    for (int j = 0; j < 4; ++j) {
      const size_t n = t[i].n[j] & ~(size_t)15;
      if (!t[i].p[j] || n < 16) continue;
      if (q.count == kRanges) die("l2pf_window: more than %d ranges", kRanges);
      q.p[q.count] = static_cast<const uint8_t*>(t[i].p[j]);
      q.n[q.count] = (uint32_t)n;
      c += (uint32_t)((n + kChunk - 1) / kChunk);
      q.end[q.count++] = c;
    }
    if (i < kFrom[window]) first = c;
  }
  if (first >= c) return;
  // the budget: the window's time at the pace. The collectives' at about their 75th percentile on rank 0 of the
  // engine's 1k decode (a stop ends them earlier); the attention's at about its 5th (nothing does).
  const int r = std::min(R, 8);
  const bool idx = t[kL2pfWqb].n[2] > 0;
  const float us = window == 1   ? 18 + 5.1f * (r - 1)
                   : window == 2 ? 18
                   : window == 3 ? (idx ? 30 : 11)
                                 : 14 + 1.2f * (r - 1);
  const uint32_t ns_chunk = (uint32_t)(kChunk / kGbps + 0.5f), budget = (uint32_t)(us * 1000 / ns_chunk);
  k_l2pf<<<1, 32, 0, s>>>(q, layer, first, budget, ns_chunk, arrived, finished, state);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace dsk
