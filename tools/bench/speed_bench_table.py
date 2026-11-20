#!/usr/bin/env python3
"""Per-category SPEED-Bench tables from one or more AIPerf run directories.

usage: speed_bench_table.py OUTPREFIX RUNDIR [RUNDIR ...]
  RUNDIR = runs/NAME (as written by tools/bench/speed_bench.sh), holding artifacts/profile_export.jsonl and
  logs/NAME.rank.tlog + logs/NAME.front.tlog (timestamped engine log lines).
  Several RUNDIRs are merged (e.g. the main file plus a later file holding
  only the rows of the three partial categories).
Writes OUTPREFIX.csv (one line per request), OUTPREFIX.md (tables),
OUTPREFIX.json (the numbers).

If the environment variable SPANS names a file of lines "END_ISO8601_UTC DURATION_S"
(requests of other users on a shared endpoint), every request of ours that overlaps
one of them (2 s margin) is marked shared=1. When the same conversation turn exists in
several RUNDIRs, the first one that is not shared is used (so a later re-run replaces
a disturbed measurement); the others are kept in the CSV with used=0.

Token counts are the SERVER's usage.completion_tokens (AIPerf records them
per request as usage_completion_tokens). Stdlib only.
"""
import csv
import json
import os
import re
import statistics as st
import sys

FULL_ROWS = 80
CATS = ["coding", "humanities", "math", "multilingual", "qa", "rag",
        "reasoning", "roleplay", "stem", "summarization", "writing"]
CAP = 8192

RE_FRONT = re.compile(r"^(\d+\.\d+) req (\d+) session \S+ prompt (\d+) \(cached (\d+)\), completion (\d+), (\w+); ttft ([\d.]+) s, ([\d.]+) tok/s")
RE_DRAFT = re.compile(r"^(\d+\.\d+) serve: req (\d+) drafts: (\d+) verify steps, ([\d.]+) drafts verified and ([\d.]+) accepted per step")
RE_SESS = re.compile(r"^(\d+\.\d+) serve: req (\d+) session \S+ prompt (\d+), .*?(\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\), (\w+);")


def load_engine(rundir, name):
    eng = {}
    fp = os.path.join(rundir, "logs", name + ".front.tlog")
    rp = os.path.join(rundir, "logs", name + ".rank.tlog")
    if os.path.exists(fp):
        for l in open(fp):
            m = RE_FRONT.match(l)
            if m:
                eng[int(m[2])] = dict(req=int(m[2]), ts=float(m[1]), prompt=int(m[3]), cached=int(m[4]),
                                      completion=int(m[5]), finish=m[6])
    if os.path.exists(rp):
        for l in open(rp):
            m = RE_DRAFT.match(l)
            if m:
                e = eng.setdefault(int(m[2]), dict(req=int(m[2])))
                e.update(steps=int(m[3]), ver=float(m[4]), acc=float(m[5]))
                continue
            m = RE_SESS.match(l)
            if m:
                e = eng.setdefault(int(m[2]), dict(req=int(m[2])))
                e.update(ts_r=float(m[1]), eng_prompt=int(m[3]), eng_tokens=int(m[4]), eng_ms=float(m[5]),
                         eng_tps=float(m[6]), eng_finish=m[7])
    return eng


def load_run(rundir):
    name = os.path.basename(rundir.rstrip("/"))
    eng = load_engine(rundir, name)
    rows, errors = [], []
    for l in open(os.path.join(rundir, "artifacts", "profile_export.jsonl")):
        d = json.loads(l)
        md, x = d["metadata"], d.get("metrics", {})
        g = lambda k: (x.get(k) or {}).get("value")
        if d.get("error") or g("usage_completion_tokens") is None:
            errors.append(dict(run=name, category=md.get("source_kind"), conversation_id=md.get("conversation_id"),
                               turn=md.get("turn_index"), error=json.dumps(d.get("error"))[:300]))
            continue
        k = g("usage_completion_tokens")
        lat, ttft = g("request_latency"), g("time_to_first_token")
        dec_s = (lat - ttft) / 1000.0
        rows.append(dict(
            run=name, category=md["source_kind"], conversation_id=md["conversation_id"], turn=md["turn_index"],
            start_s=md["request_start_ns"] / 1e9, end_s=md["request_end_ns"] / 1e9,
            prompt_tokens=g("usage_prompt_tokens"), cached_tokens=g("usage_prompt_cache_read_tokens"),
            completion_tokens=k, aiperf_client_tokens=g("output_sequence_length"),
            reasoning_tokens=g("usage_reasoning_tokens"),
            ttft_ms=ttft, latency_ms=lat, decode_s=dec_s,
            decode_tps=((k - 1) / dec_s if k > 1 and dec_s > 0 else None),
            e2e_tps=k / (lat / 1000.0),
            close_delay_ms=(md["request_end_ns"] - md["request_start_ns"]) / 1e6 - lat,
        ))
    # match engine log lines: same prompt and completion tokens, nearest end time
    used = set()
    by_key = {}
    for e in eng.values():
        if "completion" in e:
            by_key.setdefault((e["prompt"], e["completion"]), []).append(e)
    unmatched = 0
    for r in sorted(rows, key=lambda r: r["end_s"]):
        best = None
        for e in by_key.get((r["prompt_tokens"], r["completion_tokens"]), []):
            if e["req"] in used:
                continue
            dt = abs(e["ts"] - r["end_s"])
            if dt < 2.0 and (best is None or dt < best[0]):
                best = (dt, e)
        if best is None:
            unmatched += 1
            r.update(engine_req=None, finish=("length" if r["completion_tokens"] >= CAP else "stop"))
            continue
        e = best[1]
        used.add(e["req"])
        r.update(engine_req=e["req"], finish=e["finish"], verify_steps=e.get("steps"),
                 drafts_verified_per_step=e.get("ver"), drafts_accepted_per_step=e.get("acc"),
                 engine_decode_tps=e.get("eng_tps"))
    foreign = [e for q, e in eng.items() if q not in used and "completion" in e]
    return name, rows, errors, unmatched, foreign


def load_spans():
    import calendar, time
    out = []
    fn = os.environ.get("SPANS")
    if fn and os.path.exists(fn):
        for l in open(fn):
            a = l.split()
            if len(a) >= 2:
                end = calendar.timegm(time.strptime(a[0][:19], "%Y-%m-%dT%H:%M:%S")) + 1
                out.append((end - float(a[1].rstrip("s")) - 3, end + 2))
    return out


def peak_target(rows):
    ev = sorted([(r["start_s"], 1) for r in rows] + [(r["end_s"], -1) for r in rows])
    a = p = 0
    for _, d in ev:
        a += d; p = max(p, a)
    return p


def pct(v, p):
    v = sorted(v)
    if not v:
        return None
    i = (len(v) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def summarize(rows):
    if not rows:
        return None
    dt = [r["decode_tps"] for r in rows if r["decode_tps"] is not None]
    toks = [r["completion_tokens"] for r in rows]
    s = dict(
        requests=len(rows), rows=len({r["conversation_id"] for r in rows}),
        decode_mean=st.mean(dt), decode_median=st.median(dt),
        decode_pooled=sum(r["completion_tokens"] for r in rows) / sum(r["decode_s"] for r in rows),
        e2e_mean=st.mean(r["e2e_tps"] for r in rows),
        e2e_pooled=sum(toks) / sum(r["latency_ms"] / 1000 for r in rows),
        ttft_median=st.median(r["ttft_ms"] for r in rows), ttft_p90=pct([r["ttft_ms"] for r in rows], 0.9),
        out_mean=st.mean(toks), out_median=st.median(toks), out_total=sum(toks),
        cap_hits=sum(1 for r in rows if r["finish"] == "length"),
        close_median=st.median(r["close_delay_ms"] for r in rows), close_p90=pct([r["close_delay_ms"] for r in rows], 0.9),
        close_mean=st.mean(r["close_delay_ms"] for r in rows),
    )
    a = [r for r in rows if r.get("verify_steps")]
    if a:
        steps = sum(r["verify_steps"] for r in a)
        acc = sum(r["verify_steps"] * r["drafts_accepted_per_step"] for r in a)
        ver = sum(r["verify_steps"] * r["drafts_verified_per_step"] for r in a)
        s.update(acc_requests=len(a), verify_steps=steps,
                 tokens_per_step=sum(r["completion_tokens"] for r in a) / steps,
                 accept_length=1 + acc / steps, drafts_verified_per_step=ver / steps,
                 drafts_accepted_per_step=acc / steps, accept_rate=(acc / ver if ver else None))
    return s


def macro(summ, cats, keys):
    out = {}
    for k in keys:
        v = [summ[c][k] for c in cats if summ.get(c) and summ[c].get(k) is not None]
        out[k] = st.mean(v) if v else None
    return out


def f(v, n=1):
    return "-" if v is None else f"{v:.{n}f}"


def main():
    outp = sys.argv[1]
    rows, errors, notes, foreign_all = [], [], [], []
    for rd in sys.argv[2:]:
        name, r, e, um, foreign = load_run(rd)
        rows += r
        errors += e
        foreign_all += foreign
        notes.append(f"{name}: {len(r)} good requests, {len(e)} failed, {um} without an engine log match, "
                     f"{len(foreign)} engine requests in the window that were not ours")
    spans = load_spans()
    for r in rows:
        r["shared"] = int(any(r["start_s"] < b and r["end_s"] > a for a, b in spans))
        r["used"] = 0
    groups = {}
    for r in rows:  # rows are in RUNDIR order
        groups.setdefault((r["conversation_id"], r["turn"]), []).append(r)
    for g in groups.values():
        pick = next((r for r in g if not r["shared"]), g[0])
        pick["used"] = 1
    all_rows = rows
    rows = [r for r in all_rows if r["used"]]
    notes.append(f"requests measured: {len(all_rows)}; distinct conversation turns: {len(rows)}; "
                 f"measurements overlapping another user's request: {sum(r['shared'] for r in all_rows)}; "
                 f"of the turns used in the tables, still overlapping: {sum(r['shared'] for r in rows)}; "
                 f"replaced by a re-run: {sum(1 for g in groups.values() if g[0]['used'] == 0)}")
    cols = ["run", "used", "shared", "category", "conversation_id", "turn", "start_s", "end_s", "prompt_tokens", "cached_tokens",
            "completion_tokens", "aiperf_client_tokens", "reasoning_tokens", "finish", "ttft_ms", "latency_ms",
            "decode_tps", "e2e_tps", "close_delay_ms", "engine_req", "engine_decode_tps", "verify_steps",
            "drafts_verified_per_step", "drafts_accepted_per_step"]
    with open(outp + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(all_rows, key=lambda r: r["start_s"]):
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    summ = {c: summarize([r for r in rows if r["category"] == c]) for c in CATS}
    present = [c for c in CATS if summ[c]]
    complete = [c for c in present if summ[c]["rows"] >= FULL_ROWS]
    partial = [c for c in present if summ[c]["rows"] < FULL_ROWS]
    keys = ["decode_mean", "decode_median", "decode_pooled", "e2e_mean", "e2e_pooled", "ttft_median", "ttft_p90",
            "out_mean", "out_median", "close_median", "tokens_per_step", "accept_length", "accept_rate",
            "drafts_verified_per_step", "drafts_accepted_per_step"]
    overall = {
        "macro_complete": macro(summ, complete, keys), "macro_all": macro(summ, present, keys),
        "pooled_complete": summarize([r for r in rows if r["category"] in complete]),
        "pooled_all": summarize(rows),
    }
    t0, t1 = min(r["start_s"] for r in rows), max(r["end_s"] for r in rows)
    total = dict(wall_s=t1 - t0, output_tokens=sum(r["completion_tokens"] for r in rows))
    total["total_tps"] = total["output_tokens"] / total["wall_s"]
    md = []
    pc, pa = overall["pooled_complete"], overall["pooled_all"]
    md.append("| category | rows | requests | decode tok/s mean | median | pooled | end-to-end tok/s mean | TTFT ms median | p90 | output tokens mean | median | hit 8192 cap | last token to end of stream, ms median |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    def line(label, s, rowtxt, req):
        md.append(f"| {label} | {rowtxt} | {req} | {f(s['decode_mean'])} | {f(s['decode_median'])} | {f(s['decode_pooled'])} | "
                  f"{f(s['e2e_mean'])} | {f(s['ttft_median'],0)} | {f(s['ttft_p90'],0)} | {f(s['out_mean'],0)} | {f(s['out_median'],0)} | {s.get('cap_hits','-')} | {f(s.get('close_median'),0)} |")
    for c in present:
        s = summ[c]
        line(c + (" (PARTIAL)" if c in partial else ""), s, (f"{s['rows']} of {FULL_ROWS}" if partial else s["rows"]), s["requests"])
    if pc:
        line(f"**macro-average over {len(complete)} complete categories**", overall["macro_complete"], "-", "-")
    if partial or not pc:
        line(f"**macro-average, all {len(present)} (partial ones included)**", overall["macro_all"], "-", "-")
    if pc:
        line(f"**pooled over the requests of {len(complete)} complete categories**", pc, pc["rows"], pc["requests"])
    if partial or not pc:
        line("**pooled, all requests**", pa, pa["rows"], pa["requests"])
    md.append("")
    md.append("| category | rows | requests with engine line | verify steps | tokens per verify step | drafts verified per step | drafts accepted per step | acceptance rate |")
    md.append("|---|---|---|---|---|---|---|---|")
    def aline(label, s, rowtxt):
        md.append(f"| {label} | {rowtxt} | {s.get('acc_requests','-')} | {s.get('verify_steps','-')} | {f(s.get('tokens_per_step'),2)} | "
                  f"{f(s.get('drafts_verified_per_step'),2)} | {f(s.get('drafts_accepted_per_step'),2)} | "
                  f"{f(100*s['accept_rate'],1) + ' %' if s.get('accept_rate') is not None else '-'} |")
    for c in present:
        aline(c + (" (PARTIAL)" if c in partial else ""), summ[c], (f"{summ[c]['rows']} of {FULL_ROWS}" if partial else summ[c]["rows"]))
    if pc:
        aline(f"**macro-average, {len(complete)} complete**", overall["macro_complete"], "-")
    if partial or not pc:
        aline(f"**macro-average, all {len(present)}**", overall["macro_all"], "-")
    if pc:
        aline(f"**pooled, {len(complete)} complete**", pc, pc["rows"])
    if partial or not pc:
        aline("**pooled, all**", pa, pa["rows"])
    md.append("")
    md.append(f"Total (meaningful for a single run directory only): {total['output_tokens']} output tokens, first request start to last request end {total['wall_s']:.0f} s = {total['total_tps']:.1f} tok/s.")
    # steady-state throughput: tokens per second while at least N streams were active
    ev = sorted([(r["start_s"], 1) for r in rows] + [(r["end_s"], -1) for r in rows])
    act = 0; peak = 0; prev = None; tfull = 0.0
    for t, d in ev:
        if prev is not None and act >= peak_target(rows):
            tfull += t - prev
        act += d; peak = max(peak, act); prev = t
    total["peak_streams"] = peak; total["seconds_at_peak"] = tfull
    md.append(f"Peak concurrent streams: {peak}; time with all of them active: {tfull:.0f} s.")
    diff = [r["completion_tokens"] - r["aiperf_client_tokens"] for r in rows if r["aiperf_client_tokens"] is not None]
    if diff:
        md.append(f"Token count check: server usage minus AIPerf client count: min {min(diff)}, median {st.median(diff)}, max {max(diff)} over {len(diff)} requests.")
    md.append(f"Time from the last content token to the end of the stream (finish chunk): median {pa['close_median']:.0f} ms, mean {pa['close_mean']:.0f} ms, p90 {pa['close_p90']:.0f} ms. "
              f"Average number of streams generating (sum of first-byte-to-last-token time / wall): {sum(r['latency_ms'] for r in rows) / 1000 / total['wall_s']:.1f}.")
    md.append(f"Reasoning tokens (server usage) summed: {sum((r['reasoning_tokens'] or 0) for r in rows)}.")
    for n in notes:
        md.append("- " + n)
    md.append(f"- failed requests: {len(errors)}")
    for e in errors[:40]:
        md.append(f"  - {e}")
    open(outp + ".md", "w").write("\n".join(md) + "\n")
    json.dump(dict(pooled_all_gen=sum(r["latency_ms"] for r in rows) / 1000 / total["wall_s"],
                   still_shared=sum(r["shared"] for r in rows), per_category=summ, overall=overall, total=total, complete=complete, partial=partial,
                   errors=errors, notes=notes,
                   foreign=[dict(req=e["req"], ts=e["ts"], prompt=e["prompt"], completion=e["completion"]) for e in foreign_all]),
              open(outp + ".json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
