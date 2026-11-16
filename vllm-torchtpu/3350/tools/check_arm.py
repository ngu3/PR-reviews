#!/usr/bin/env python3
"""Completeness check for one mskip_engine.py result (standard library only; runs in the pod and in the analysis).

  python3 check_arm.py RESULT.json [--phases numerics,sweep,rollout]

Without --phases the phases recorded in the file's meta are checked. A phase is complete when:
  numerics  cases n1 / n10 / n40 hold 1 / 10 / 40 rows; every row has a prompt_sha, a finish_reason, a non-empty
            token_ids list and a logprobs list of the same length with no missing value;
  sweep     every requested n is present (meta.sweep_ns; files written before it was recorded: at least one n) and
            holds meta.reps timed repeats (older files: the same number at every point);
  steady    every n in meta.steady_ns is present with meta.steady_reps finite, positive per-step times and as many
            short / long wall times, each long run slower than its short run;
  rollout   wall_s holds 3 repeats and best is set;
  profile   every n in meta.profile_ns has a profiled batch (the traces themselves are checked by prof_mskip.py);
  switch    one batch per entry of meta.switch_seq, in that order, with n rows each, every row complete as in
            numerics;
  graphsel  two passes, each with one batch per entry of meta.graphsel_seq in that order, and every batch has a
            worker report before and after it (a list, one entry per worker, not a collective_rpc error);
            graphsel_check.py judges the contents.
A run that finished all its phases also has done=true (files written before that field existed are judged by the
phases alone). Exit 0 when complete; otherwise print the problems and exit 1.
"""
import argparse
import gzip
import json
import math
import sys

EXPECTED_CASES = {"n1": 1, "n10": 10, "n40": 40}
DEFAULT_REPS = 3


def _finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _bad_logprobs(ids, lps):
    return lps is None or len(lps) != len(ids) or not all(_finite_number(x) for x in lps)


def expected_rank_set(meta, workers=None):
    """Expected worker rank set {0..W-1} from meta.torch_tpu_topology or workers.after_start, else None."""
    topo = (meta or {}).get("torch_tpu_topology")
    if isinstance(topo, str) and topo.strip():
        parts = [p.strip() for p in topo.split(",") if p.strip()]
        if parts and all(p.isdigit() and int(p) > 0 for p in parts):
            prod = 1
            for p in parts:
                prod *= int(p)
            return set(range(prod))
    after_start = (workers or {}).get("after_start")
    if isinstance(after_start, list) and after_start:
        return set(range(len(after_start)))
    return None


def _check_worker_ranks(rep, want_ranks, label):
    if not isinstance(rep, list) or not rep:
        return [f"{label} has no worker report: {str(rep)[:120]}"]
    ranks = []
    for w in rep:
        r = w.get("rank") if isinstance(w, dict) else None
        if not isinstance(r, int) or isinstance(r, bool) or r < 0:
            return [f"{label} has invalid worker rank {r!r}"]
        ranks.append(r)
    if len(ranks) != len(set(ranks)):
        return [f"{label} has duplicate worker ranks {ranks}"]
    rset = set(ranks)
    if want_ranks is not None and rset != want_ranks:
        return [f"{label} has ranks {sorted(rset)}, expected {sorted(want_ranks)}"]
    if want_ranks is None and rset != set(range(len(ranks))):
        return [f"{label} has non-contiguous ranks {sorted(rset)}"]
    return []


def load(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)


def problems(d, phases=None):
    """List of problems of result dict `d` for `phases` (default: the phases in its meta)."""
    out = []
    meta = d.get("meta") or {}
    phases = list(phases) if phases is not None else list(meta.get("phases") or [])
    if not phases:
        return ["no phases recorded"]
    if "done" in d and d["done"] is not True:
        out.append("done is not true")
    if "numerics" in phases:
        cases = d.get("cases") or {}
        for case, n in EXPECTED_CASES.items():
            rows = cases.get(case)
            if rows is None:
                out.append(f"numerics: case {case} missing")
                continue
            if len(rows) != n:
                out.append(f"numerics: case {case} has {len(rows)} rows, expected {n}")
            for i, r in enumerate(rows):
                ids, lps = r.get("token_ids"), r.get("logprobs")
                if not r.get("prompt_sha") or not r.get("finish_reason"):
                    out.append(f"numerics: {case}[{i}] lacks prompt_sha or finish_reason")
                if not ids:
                    out.append(f"numerics: {case}[{i}] has no token ids")
                elif _bad_logprobs(ids, lps):
                    out.append(f"numerics: {case}[{i}] logprobs missing, non-finite, or not one per token")
    if "sweep" in phases:
        sweep = d.get("sweep") or {}
        want = [str(n) for n in meta["sweep_ns"]] if meta.get("sweep_ns") else list(sweep)
        # Files written before meta.reps was recorded: every point must have the same number of repeats.
        first = next(iter(sweep.values()), {})
        reps = int(meta.get("reps") or len(first.get("ms_per_step") or []) or DEFAULT_REPS)
        if not want:
            out.append("sweep: no points")
        for n in want:
            v = sweep.get(n)
            if v is None:
                out.append(f"sweep: n={n} missing")
            elif len(v.get("ms_per_step") or []) != reps:
                out.append(f"sweep: n={n} has {len(v.get('ms_per_step') or [])} repeats, expected {reps}")
    if "steady" in phases:
        st = d.get("steady") or {}
        want = [str(n) for n in meta.get("steady_ns") or []] or list(st)
        reps = int(meta.get("steady_reps") or 0)
        if not want:
            out.append("steady: no points")
        for n in want:
            v = st.get(n)
            if v is None:
                out.append(f"steady: n={n} missing")
                continue
            per, ts, tl = v.get("ms_per_step") or [], v.get("t_short_s") or [], v.get("t_long_s") or []
            if (reps and len(per) != reps) or not per or len(ts) != len(per) or len(tl) != len(per):
                out.append(f"steady: n={n} has {len(per)} / {len(ts)} / {len(tl)} repeats, expected {reps}")
            elif not all(_finite_number(x) and x > 0 for x in per) or any(b <= a for a, b in zip(ts, tl)):
                out.append(f"steady: n={n} has a non-finite or non-positive per-step time")
    if "rollout" in phases:
        ro = d.get("rollout") or {}
        if len(ro.get("wall_s") or []) != 3 or ro.get("best") is None:
            out.append("rollout: missing or not 3 repeats")
    if "prefill_mixed" in phases:
        pm = d.get("prefill_mixed") or {}
        want_reps = int(meta.get("prefill_mixed_reps") or 5)
        expected_cases = ("prefill_n16", "prefill_n64", "mixed_16_long", "mixed_64_stagger")
        if not pm:
            out.append("prefill_mixed: missing")
        for name in expected_cases:
            v = pm.get(name)
            if v is None:
                out.append(f"prefill_mixed: {name} missing")
            else:
                ws = v.get("wall_s") or []
                if len(ws) != want_reps or not all(_finite_number(x) for x in ws) or v.get("best") is None:
                    out.append(f"prefill_mixed: {name} has {len(ws)} repeats, expected {want_reps}")
    if "switch" in phases:
        sw = d.get("switch") or []
        seq = meta.get("switch_seq") or [b.get("n") for b in sw]
        if len(sw) != len(seq) or not seq:
            out.append(f"switch: {len(sw)} batches, expected {len(seq)}")
        for i, (b, n) in enumerate(zip(sw, seq)):
            rows = b.get("rows") or []
            if b.get("n") != n or len(rows) != n:
                out.append(f"switch: batch {i} has n={b.get('n')} and {len(rows)} rows, expected {n}")
            for j, r in enumerate(rows):
                ids, lps = r.get("token_ids"), r.get("logprobs")
                if not r.get("prompt_sha") or not r.get("finish_reason") or not ids:
                    out.append(f"switch: batch {i} row {j} lacks prompt_sha, finish_reason or token ids")
                elif _bad_logprobs(ids, lps):
                    out.append(f"switch: batch {i} row {j} logprobs missing, non-finite, or not one per token")
    if "graphsel" in phases:
        gs = d.get("graphsel") or {}
        seq = meta.get("graphsel_seq") or gs.get("seq") or []
        passes = gs.get("passes") or []
        want_ranks = expected_rank_set(meta, d.get("workers"))
        if len(passes) != 2 or not seq:
            out.append(f"graphsel: {len(passes)} passes, expected 2")
        for i, batches in enumerate(passes):
            if [b.get("n") for b in batches] != list(seq):
                out.append(f"graphsel: pass {i} batches {[b.get('n') for b in batches]}, expected {seq}")
            for j, b in enumerate(batches):
                for key in ("before", "after"):
                    out.extend(_check_worker_ranks(b.get(key), want_ranks, f"graphsel: pass {i} batch {j} {key}"))
                if isinstance(b.get("before"), list) and isinstance(b.get("after"), list):
                    br = [w.get("rank") for w in b["before"] if isinstance(w, dict)]
                    ar = [w.get("rank") for w in b["after"] if isinstance(w, dict)]
                    if set(br) != set(ar):
                        out.append(f"graphsel: pass {i} batch {j} rank sets differ before={sorted(set(br), key=str)} after={sorted(set(ar), key=str)}")
    if "profile" in phases:
        prof = d.get("profile") or {}
        for n in meta.get("profile_ns") or list(prof) or ["?"]:
            if (prof.get(str(n)) or {}).get("wall_s") is None:
                out.append(f"profile: n={n} missing")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--phases", help="comma-separated; default: the phases in the file's meta")
    a = ap.parse_args()
    try:
        d = load(a.result)
    except (OSError, ValueError) as e:
        print(f"CHECK_ARM {a.result}: cannot read: {e}")
        return 1
    p = problems(d, a.phases.split(",") if a.phases else None)
    for x in p:
        print(f"CHECK_ARM {a.result}: {x}")
    if not p:
        print(f"CHECK_ARM {a.result}: complete")
    return 1 if p else 0


if __name__ == "__main__":
    sys.exit(main())
