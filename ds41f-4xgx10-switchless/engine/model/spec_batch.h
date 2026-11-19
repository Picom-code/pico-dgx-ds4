// B3 (docs/batching.md): the batch spec step's layout and verdicts, one code for the device (spec.cu's
// k_spec_in_batch, sample.cu's k_sample) and the host (tests/spec_batch_test.cu checks them against a
// reference per entry, no GPU needed).
#pragma once
#include "model/ops.h"

namespace model::ops {

// wide64 (docs/wide64.md): a batch spec step of R > kDecodeRows rows runs the switch body of the smallest bucket
// that holds it, its rows past R pad rows (copies of row R - 1, masked from every state write). Bodies 0 .. 15: R
// rows exactly; 16 + i: kWideBuckets[i] rows.
constexpr int kNumWideBuckets = 8;
__host__ __device__ inline int wide_bucket_rows(int i) {
  const int b[kNumWideBuckets] = {20, 24, 28, 32, 40, 48, 56, 64};
  return b[i];
}
__host__ __device__ inline int spec_body(int R) {  // the switch body of a step of R rows
  if (R <= kDecodeRows) return R - 1;
  int i = 0;
  while (i < kNumWideBuckets - 1 && wide_bucket_rows(i) < R) ++i;
  return kDecodeRows + i;
}
__host__ __device__ inline int spec_body_rows(int R) {  // the rows that body runs
  return R <= kDecodeRows ? R : wide_bucket_rows(spec_body(R) - kDecodeRows);
}

// A batch verify step's rows from the entries' spec states: the entries on (sess_on), in entry order,
// each its next_rows rows (its bonus token, then its drafts) at next_pos0 .., with its slot (sess_slot)
// and its entry index in sess[]; the entries past R repeat the last row's (tokens 0); row0[k] entry k's
// first row (an entry off: where it would have started). Returns R, which the host keeps within
// kMaxRows (the entries' caps) and above 0 (it launches with an entry on).
// copy drafts (DSGB10_EXACT_SOLO, model.cu): xb != 0: a one-entry step may hold 1 + n rows only when it is at most kLowRows
// rows (the bodies of before, bit-equal at 1..8 rows) or its wide bucket is one of xb's (bit i: wide_bucket_rows(i),
// an exact body): n drafts become the most drafts below them that do.
__host__ __device__ inline int copy_fit(int n, int xb) {
  if (!xb) return n;
  while (n > kLowRows - 1) {
    const int R = 1 + n;
    if (R <= kDecodeRows) return kLowRows - 1;
    const int b = spec_body(R) - kDecodeRows;
    if ((xb >> b) & 1) return n;
    n = (b > 0 ? wide_bucket_rows(b - 1) : kDecodeRows) - 1;
  }
  return n;
}

// Copy drafts (ops.h CopyWin): the drafts entry k takes from its window in this step, 0 for none; *at: their first
// token in d.copy_tok. s: the entry's spec state (its first row's token and position, the draft module's choice);
// avail: the rows the step has left for its drafts.
__host__ __device__ inline int spec_copy_pick(const StepIn& d, int k, const SpecState& s, int avail, int* at) {
  for (int w = 0; w < kCopyWin; ++w) {
    const CopyWin& c = d.copy[w];
    if (c.ent != k || c.n < 2 || c.off < 0 || c.off + c.n > kCopyTok) continue;
    const int off = s.next_pos0 - c.pos0;
    if (off < 0 || off >= c.n - 1) return 0;
    const int32_t* t = d.copy_tok + c.off + off;
    if (t[0] != s.next_tok[0]) return 0;  // the window does not hold what the model wrote
    int n = c.n - 1 - off;
    if (n > c.max) n = c.max;
    if (n > d.spos_end[k] - 1 - s.next_pos0) n = d.spos_end[k] - 1 - s.next_pos0;
    if (n > avail) n = avail;
    n = copy_fit(n, d.copy_on >> 8);  // copy drafts: StepIn::copy_on's bits 8.. are the exact buckets of a one-entry step
    const int nk = s.next_rows < 1 ? 1 : s.next_rows > kSessTok ? kSessTok : s.next_rows;
    if (c.mode & 1)  // the draft module's gated drafts say otherwise: theirs
      for (int j = 1; j < nk && j <= n; ++j)
        if (s.next_tok[j] != t[j]) return 0;
    if (n < nk) return 0;  // no more drafts than the module's own: theirs
    *at = c.off + off + 1;
    return n;
  }
  return 0;
}

// ncopy (host, optional): [kMaxSess], the drafts each entry took from a copy window.
__host__ __device__ inline int spec_batch_layout(StepIn& d, const SpecState* ss, int* row0, int* ncopy = nullptr) {
  const int ns = d.nsess < 1 ? 1 : d.nsess > kMaxSess ? kMaxSess : d.nsess;
  int R = 0;
  int base = 0, cap = 0;  // copy drafts: the rows of the entries still to lay out by the module's choice; the step's cap
  if (d.copy_on) {
    for (int k = 0; k < ns; ++k)
      if (d.sess_on[k]) base += ss[k].next_rows < 1 ? 1 : ss[k].next_rows > kSessTok ? kSessTok : ss[k].next_rows;
    cap = d.copy_cap;
    if (base > kDecodeRows && spec_body_rows(base) > cap) cap = spec_body_rows(base);  // a wide bucket's pad rows
    if (cap > kMaxRows) cap = kMaxRows;
  }
  for (int k = 0; k < ns; ++k) {
    row0[k] = R;
    if (ncopy) ncopy[k] = 0;
    if (!d.sess_on[k]) continue;
    int nk = ss[k].next_rows;
    nk = nk < 1 ? 1 : nk > kSessTok ? kSessTok : nk;
    if (d.copy_on) {
      base -= nk;
      int at = 0;
      const int nc = spec_copy_pick(d, k, ss[k], cap - R - base - 1, &at);
      if (nc > 0) {
        if (ncopy) ncopy[k] = nc;
        for (int j = 0; j <= nc && R < kMaxRows; ++j, ++R) {
          d.tokens[R] = j ? d.copy_tok[at + j - 1] : ss[k].next_tok[0];
          d.pos[R] = ss[k].next_pos0 + j;
          d.slot[R] = d.sess_slot[k];
          d.sess[R] = k;
        }
        continue;
      }
    }
    for (int j = 0; j < nk && R < kMaxRows; ++j, ++R) {
      d.tokens[R] = ss[k].next_tok[j];
      d.pos[R] = ss[k].next_pos0 + j;
      d.slot[R] = d.sess_slot[k];
      d.sess[R] = k;
    }
  }
  for (int k = ns; k < kMaxSess; ++k) row0[k] = 0;
  if (R < 1) R = 1;  // never: the host launches with an entry on
  d.rows = R;
  for (int r = R; r < kMaxRows; ++r) {  // the entries past R repeat the last row's (wide64: its token too, so that a
    d.tokens[r] = d.tokens[R - 1];      // wide bucket's pad rows compute row R - 1's values and add no experts)
    d.pos[r] = d.pos[R - 1], d.slot[r] = d.slot[R - 1], d.sess[r] = d.sess[R - 1];
  }
  return R;
}

// B3 phase 2a: the draft passes of a batch step (dspark::Draft::draft_batch). bent [kMaxSess]: the entries on in
// entry order (the compact list; the rest 0); pstab [kMaxSess * kDrafts]: their rows' slot tables, the c-th entry
// on at rows [kDrafts c, kDrafts c + kDrafts) (pass p takes bent + 3 p and pstab + 3 p * kDrafts; the rest slot
// 0, never read). Returns S, the entries on (the switch over the passes runs body S - 1).
// blk7: nd = the batch passes' rows per entry (dspark::draft_rows_batch()), the tables' stride.
__host__ __device__ inline int spec_batch_passes(const StepIn& d, int* bent, int* pstab, int nd = kDrafts) {
  const int ns = d.nsess < 1 ? 1 : d.nsess > kMaxSess ? kMaxSess : d.nsess;
  int S = 0;
  for (int k = 0; k < ns; ++k)
    if (d.sess_on[k]) bent[S++] = k;
  for (int c = S; c < kMaxSess; ++c) bent[c] = 0;
  for (int c = 0; c < kMaxSess; ++c)
    for (int j = 0; j < nd; ++j) pstab[c * nd + j] = c < S ? d.sess_slot[bent[c]] : 0;
  return S;
}

// A batch given step's row0 from the host's layout: sess[r] the row's entry, the entries on in order.
__host__ __device__ inline void spec_batch_row0(const StepIn& d, int* row0) {
  const int ns = d.nsess < 1 ? 1 : d.nsess > kMaxSess ? kMaxSess : d.nsess;
  int r = 0;
  for (int k = 0; k < ns; ++k) {
    row0[k] = r;
    if (!d.sess_on[k]) continue;
    while (r < d.rows && d.sess[r] == k) ++r;
  }
  for (int k = ns; k < kMaxSess; ++k) row0[k] = 0;
}

// The verdicts per entry of a batch spec step: fb[r] row r's sample, in.tokens[r] what it was fed. Entry
// k's rows are its run in sess[]: its first row is final and, in a verify step, each row after it is
// accepted while its token is the sample of the row before it (the one-session rule, over its rows
// alone; no row reads another entry's). v[k]: step_rows 0 for an entry with no rows; else its row0,
// step_rows, rows (the tokens it emits: the accepted drafts, then the token after them), pos0 and bonus
// (the sample of its last final row).
struct SpecVerdict {
  int32_t row0, step_rows, rows, pos0, bonus;
};
__host__ __device__ inline void spec_batch_verdicts(const StepIn& in, const volatile int32_t* fb, bool verify,
                                                    SpecVerdict* v) {
  for (int k = 0; k < kMaxSess; ++k) v[k] = SpecVerdict{0, 0, 0, 0, 0};
  const int R = in.rows < 1 ? 1 : in.rows > kMaxRows ? kMaxRows : in.rows;
  int r0 = 0;
  while (r0 < R) {
    const int k = in.sess[r0] < 0 ? 0 : in.sess[r0] >= kMaxSess ? kMaxSess - 1 : in.sess[r0];
    int nk = 1;
    while (r0 + nk < R && in.sess[r0 + nk] == in.sess[r0]) ++nk;
    int n = nk - 1;
    if (verify) {
      n = 0;
      while (n < nk - 1 && fb[r0 + n] == in.tokens[r0 + n + 1]) ++n;
    }
    v[k] = SpecVerdict{r0, nk, n + 1, in.pos[r0], fb[r0 + n]};
    r0 += nk;
  }
}

}  // namespace model::ops
