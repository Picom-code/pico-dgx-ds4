// Copy drafts' host side (docs/copy.md): when a reply repeats text that is already in the prompt (a file rewrite, an
// edit, an agent echoing code) or in the client's `prediction`, the tokens that followed the same text there are
// offered to the verify step as drafts (model::ops::CopyWin). Every draft is verified by the model as the draft
// module's are, so they change which rows a step runs, never a token. Host-only and deterministic: every rank runs
// the same matcher over the same tokens and so writes the same StepIn.
//
// Knobs (environment, read once; the same on every rank). DSGB10_COPY=1 turns the feature on; unset or 0: nothing
// below is used and the engine's steps are the ones of before.
//   DSGB10_COPY_ROWS=R     rows of a step, at most, when one request decodes and copies (default 16: the decode
//                          kernels' steps; past 16 the step runs the wide body, whose rows differ from theirs by ulps)
//   DSGB10_COPY_SHARED=R   the same when several requests share the step (default: COPY_ROWS)
//   DSGB10_COPY_MATCH=M    tokens that must match before a copy is offered (default 8)
//   DSGB10_COPY_N0=N       drafts of the first copy step (default 15); each copy accepted whole doubles it (+1)
//   DSGB10_COPY_RESET=0|1  after a rejected copy: back to N0 (0, default) or half the width (1)
//   DSGB10_COPY_MODE=0|1   1: only when the draft module's own gated drafts agree with the copy
//   DSGB10_COPY_ENTER=M    one request alone: matched tokens that move it to copy steps (default 16)
//   DSGB10_COPY_EXIT=S     one request alone: steps without a copy that move it back (default 24)
//   DSGB10_COPY_BACKOFF=0|1 1 (default): every copy in a row that was not accepted whole doubles the match the next
//                          one needs (up to 8x); a copy accepted whole resets it. Text that only half repeats a
//                          source (numbered lines, reformatted code) then stops paying for rejected rows.
//   DSGB10_COPY_MINTOK=N   a reply's first N tokens never copy (default 0): short tool-call turns stay in the
//                          speculative loop and pay for no change of step kind
//   DSGB10_COPY_SW=1       copy drafts (docs/copy.md), one request alone: (a) a change of step kind waits for the step
//                          already queued and uses its tokens, and feeds no token twice (before: the queued step's
//                          tokens were dropped and one more step fed the last token); (b) a visit to copy steps that
//                          has not yet had COPY_ENTER copied tokens accepted ends at its first step without a copy
//                          (a productive one after COPY_EXIT such steps, as before); (c) every visit that ends
//                          unproductive doubles the match the next one needs (up to 8x), a productive one resets it.
//                          0 (default): the behaviour of before
//   DSGB10_COPY_DELTA=1    copy drafts: a request's prompt goes to the other ranks from the first token their history of
//                          the slot does not hold already (at most 3 before the restart point, as without copy
//                          drafts), with a hash of what is not sent; 0 (default): from position 0
//   DSGB10_COPY_LOG=1      a line per request on rank 0's stderr
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace srv {

struct CopyKnobs {
  int on = 0, rows = 16, shared = 16, match = 8, n0 = 15, reset = 0, mode = 0, enter = 16, exit = 24, log = 0, backoff = 1;
  int mintok = 0;
  int sw = 0, delta = 0;  // copy drafts
  static int env(const char* k, int d) {
    const char* v = getenv(k);
    return v && *v ? atoi(v) : d;
  }
  static const CopyKnobs& get() {
    static const CopyKnobs k = [] {
      CopyKnobs c;
      c.on = env("DSGB10_COPY", 0) > 0;
      c.rows = std::max(2, env("DSGB10_COPY_ROWS", 16));
      // DSGB10_COPY_SHARED=0: no copy window in a step that holds two or more requests (a copy adds rows to the
      // shared step, and DSGB10_OEA's re-routing then couples the other requests' experts to them: their texts change)
      c.shared = env("DSGB10_COPY_SHARED", c.rows);
      if (c.shared != 0) c.shared = std::max(2, c.shared);
      c.match = std::max(1, env("DSGB10_COPY_MATCH", 8));
      c.n0 = std::max(1, env("DSGB10_COPY_N0", 15));
      c.reset = env("DSGB10_COPY_RESET", 0);
      c.mode = env("DSGB10_COPY_MODE", 0);
      c.enter = std::max(1, env("DSGB10_COPY_ENTER", 16));
      c.exit = std::max(1, env("DSGB10_COPY_EXIT", 24));
      c.log = env("DSGB10_COPY_LOG", 0);
      c.backoff = env("DSGB10_COPY_BACKOFF", 1);
      c.mintok = std::max(0, env("DSGB10_COPY_MINTOK", 0));
      c.sw = env("DSGB10_COPY_SW", 0) > 0;
      c.delta = env("DSGB10_COPY_DELTA", 0) > 0;
      return c;
    }();
    return k;
  }
};

// Where every run of kGram tokens of a corpus last ended, chained to the earlier ends of the same hash.
class CopyIndex {
 public:
  static constexpr int kGram = 4, kChain = 256, kBack = 64;
  void clear() {
    head_.clear(), prev_.clear();
    n_ = 0;
  }
  int size() const { return n_; }
  // the corpus grew to n tokens (c[0 .. n_) are the ones indexed before)
  void extend(const int32_t* c, int n) {
    if (n <= n_) return;
    if ((size_t)n * 2 > head_.size()) {  // rehash: every gram again
      size_t cap = 1024;
      while (cap < (size_t)n * 3) cap *= 2;
      head_.assign(cap, -1);
      n_ = 0;
    }
    prev_.resize(n, -1);
    const uint32_t mask = (uint32_t)head_.size() - 1;
    for (int e = std::max(n_, kGram - 1); e < n; ++e) {
      const int32_t* g = c + e - kGram + 1;
      if ((g[0] | g[1] | g[2] | g[3]) < 0) continue;  // image positions, separators
      const uint32_t h = hash(g) & mask;
      prev_[e] = head_[h];
      head_[h] = e;
    }
    n_ = n;
  }
  // The corpus index after the occurrence whose text best matches the tail's end (the longest match backwards, up
  // to kBack tokens; among equals the latest), -1 for none; *len: the tokens matched. Occurrences ending at or after
  // `below` are skipped (the tail itself, when it is part of the corpus). near (>= 0): among equal matches the first
  // one at or after it (where the last copy stood: a rewrite goes on further down the same file).
  int find(const int32_t* c, const int32_t* tail, int tn, int below, int* len, int near = -1) const {
    *len = 0;
    if (tn < kGram || head_.empty()) return -1;
    const int32_t* g = tail + tn - kGram;
    if ((g[0] | g[1] | g[2] | g[3]) < 0) return -1;
    int best = -1;
    int e = head_[hash(g) & ((uint32_t)head_.size() - 1)];
    for (int walk = 0; e >= 0 && walk < kChain; e = prev_[e], ++walk) {
      if (e + 1 >= n_ || e >= below) continue;
      if (std::memcmp(c + e - kGram + 1, g, kGram * 4) != 0) continue;
      int l = kGram;
      while (l < tn && l < kBack && e - l >= 0 && c[e - l] == tail[tn - 1 - l] && c[e - l] >= 0) ++l;
      if (l > *len || (l == *len && near >= 0 && e + 1 >= near)) *len = l, best = e + 1;
    }
    return best;
  }

 private:
  static uint32_t hash(const int32_t* g) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < kGram; ++i) h = (h ^ (uint32_t)g[i]) * 16777619u, h ^= h >> 15;
    return h;
  }
  std::vector<int32_t> head_, prev_;
  int n_ = 0;
};

// A corpus and its index; a session slot's prompt is kept between requests and extended when the new one starts
// with the old one (an agent's next turn).
struct CopyCorpus {
  std::vector<int32_t> c;
  CopyIndex ix;
  void set(const int32_t* t, int n) {
    size_t keep = c.size();
    if (keep > (size_t)n || (keep && std::memcmp(c.data(), t, keep * 4) != 0)) keep = 0, c.clear(), ix.clear();
    c.insert(c.end(), t + keep, t + n);
    ix.extend(c.data(), n);
  }
};

// One request's copy state: its sources (the prompt and, when the client sent one, the prediction), where the copy
// stands, how wide it may be, and its figures.
struct CopySrc {
  const CopyCorpus* prompt = nullptr;
  CopyCorpus pred;
  std::vector<int32_t> tail;  // the last tokens of the sequence (kBack at most are read)
  int src = -1, sp = -1;      // the copy: source (0 prompt, 1 prediction) and the index of the token predicted next
  int mlen = 0;               // tokens in a row that matched behind it
  int near[2] = {-1, -1};     // where the last copy of each source stood
  int width = 0;              // drafts of the next copy step
  int idle = 0;               // steps without a copy since the last one
  int fails = 0;              // copies in a row that were not accepted whole (CopyKnobs::backoff)
  int steps = 0, rows = 0, acc = 0, whole = 0;  // copy steps, their drafts, the accepted ones, steps accepted whole
  int visit0 = 0;             // copy drafts: acc when the request's current visit to copy steps began

  // hist: the sequence before this run (n tokens: the prompt and what was generated before); first: the token the
  // run starts from (at position n); pr: the prediction's tokens (may be empty)
  void init(const CopyCorpus* p, const int32_t* hist, int n, int32_t first, const std::vector<int32_t>& pr) {
    prompt = p;
    pred.c.clear(), pred.ix.clear();
    if (!pr.empty()) pred.set(pr.data(), (int)pr.size());
    const int t = std::min(n, (int)CopyIndex::kBack);
    tail.assign(hist + n - t, hist + n);
    src = sp = -1, mlen = 0, idle = 0, fails = 0;
    near[0] = near[1] = -1;
    width = CopyKnobs::get().n0;
    feed(first);
  }
  const CopyCorpus* corpus(int s) const { return s == 1 ? &pred : prompt; }
  // a token the model wrote
  void feed(int32_t t) {
    tail.push_back(t);
    if (tail.size() > 4096) tail.erase(tail.begin(), tail.end() - CopyIndex::kBack);
    if (sp >= 0) {
      const CopyCorpus* c = corpus(src);
      if (sp < (int)c->c.size() && c->c[sp] == t) {
        ++sp, ++mlen;
        if (sp < (int)c->c.size() && c->c[sp] >= 0) return;  // the copy goes on
      }
    }
    if (sp >= 0) near[src] = sp;
    src = sp = -1, mlen = 0;
    int l0 = 0, l1 = 0;
    const int n = (int)tail.size();
    const int a = prompt ? prompt->ix.find(prompt->c.data(), tail.data(), n, (int)prompt->c.size(), &l0, near[0]) : -1;
    const int b = pred.c.empty() ? -1 : pred.ix.find(pred.c.data(), tail.data(), n, (int)pred.c.size(), &l1, near[1]);
    if (b >= 0 && l1 >= l0) src = 1, sp = b, mlen = l1;  // equal matches: the prediction
    else if (a >= 0) src = 0, sp = a, mlen = l0;
  }
  bool ready(int min_match) const { return sp >= 0 && mlen >= (min_match << (CopyKnobs::get().backoff ? std::min(fails, 3) : 0)); }
  // the tokens predicted next, at most cap of them: how many
  int window(int32_t* out, int cap) const {
    if (sp < 0) return 0;
    const CopyCorpus* c = corpus(src);
    int n = 0;
    for (; n < cap && sp + n < (int)c->c.size() && c->c[sp + n] >= 0; ++n) out[n] = c->c[sp + n];
    return n;
  }
  // a copy step of `drafts` drafts ended with `accepted` of them accepted
  void copied(int drafts, int accepted, int max_width) {
    const CopyKnobs& k = CopyKnobs::get();
    ++steps, rows += drafts, acc += accepted, idle = 0;
    if (accepted >= drafts) ++whole, fails = 0, width = std::min(2 * width + 1, max_width);
    else ++fails, width = k.reset ? std::max(k.n0, width / 2) : k.n0;
    width = std::max(1, std::min(width, max_width));
  }
};

// copy drafts (DSGB10_COPY_DELTA): a hash of n token ids (every rank's history of a slot below the first token sent)
inline uint64_t copy_hist_hash(const int32_t* t, int n) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t)n;
  int i = 0;
  for (; i + 2 <= n; i += 2) {
    uint64_t w;
    std::memcpy(&w, t + i, 8);
    h = (h ^ w) * 0xD6E8FEB86659FD93ull, h ^= h >> 32;
  }
  if (i < n) h = (h ^ (uint32_t)t[i]) * 0xD6E8FEB86659FD93ull, h ^= h >> 32;
  return h;
}

}  // namespace srv
