#!/usr/bin/env python3
"""t22: acceptance of the upstream single-graph branch, same-host performance (steady decode, prefill / mixed,
256-request fixed-length rollout) of `up_on` (branch, flag on) against `upbase_off` (its parent, flag off).

    python3 t22_compare.py RESULTS_DIR [--out t22_perf.json] [--min-sens 0.10]   (*.json / *.json.gz, any depth)

Arms are grouped by host (meta.node). Per host: ref = mean of the upbase_off sessions, on = mean of the up_on sessions;
A/A = second session - first session (each kind); noise(n) = max(|A/A upbase|, |A/A up|, --min-sens) in %.
steady: median over the 7 pairs of (t(545) - t(33)) / 512 at n requests; prefill_mixed: median wall over 9 reps;
rollout: median wall over 3 reps. Checks: every arm done; up_on workers report padding_rows "count" and no stock graph,
upbase_off workers report no policy (parent has none) or a disabled one. KV-block differences are notes.
Exit 0 when all checks pass, 2 otherwise.
"""
import argparse
import glob
import gzip
import json
import os
import statistics
import sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--out")
ap.add_argument("--min-sens", type=float, default=0.10)
a = ap.parse_args()


def kind_of(lab):
    return "_".join(lab.split("_")[:2])


arms = {}
for p in sorted(glob.glob(os.path.join(a.results, "**", "*.json"), recursive=True)
                + glob.glob(os.path.join(a.results, "**", "*.json.gz"), recursive=True)):
    if ".partial." in p:
        continue
    try:
        d = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
    except (OSError, ValueError):
        continue
    if not isinstance(d, dict) or "meta" not in d:
        continue
    if not any(k in d for k in ("steady", "prefill_mixed", "rollout")):
        continue
    arms[d.get("meta", {}).get("label") or os.path.basename(p)[:-5]] = d

problems, notes = [], []
by_host = defaultdict(lambda: defaultdict(list))
for lab, d in arms.items():
    node = d["meta"].get("node") or d["meta"].get("host")
    k = kind_of(lab)
    by_host[node][k].append(lab)
    if not d.get("done"):
        problems.append(f"{lab}: not done")
    w = (d.get("workers") or {}).get("after_start")
    if not (isinstance(w, list) and w):
        problems.append(f"{lab}: no worker report")
        continue
    pol = {((x.get("bf16_policy") or {}).get("padding_rows")) for x in w}
    sg = {json.dumps(x.get("stock_graph")) for x in w}
    want = "count" if k == "up_on" else None
    if pol != {want}:
        problems.append(f"{lab}: padding_rows {sorted(map(str, pol))}, expected {want}")
    if sg != {"null"}:
        problems.append(f"{lab}: stock graph present {sorted(sg)}")


def kv(lab):
    w = (arms[lab].get("workers") or {}).get("after_start") or []
    return sorted({x.get("kv_num_blocks") for x in w if isinstance(x, dict)})


def steady(lab, n):
    v = (arms[lab].get("steady") or {}).get(str(n))
    return statistics.median(v["ms_per_step"]) if v else None


def pm(lab, case):
    v = (arms[lab].get("prefill_mixed") or {}).get(case)
    return statistics.median(v["wall_s"]) if v else None


def ro(lab, _):
    v = arms[lab].get("rollout")
    return statistics.median(v["wall_s"]) if v else None


def pct(x, ref):
    return (x / ref - 1.0) * 100.0


def compare(kinds, getter, keys):
    rows = {}
    for key in keys:
        b = [getter(lab, key) for lab in sorted(kinds.get("upbase_off", []))]
        o = [getter(lab, key) for lab in sorted(kinds.get("up_on", []))]
        b = [x for x in b if x is not None]
        o = [x for x in o if x is not None]
        if not b or not o:
            continue
        ref, on = statistics.mean(b), statistics.mean(o)
        aa_b = pct(b[1], b[0]) if len(b) > 1 else None
        aa_o = pct(o[1], o[0]) if len(o) > 1 else None
        noise = max([abs(x) for x in (aa_b, aa_o) if x is not None] + [a.min_sens])
        d = pct(on, ref)
        rows[str(key)] = dict(ref=round(ref, 6), on=round(on, 6), delta_pct=round(d, 3),
                              per_session_pct=[round(pct(x, ref), 3) for x in o],
                              aa_base_pct=None if aa_b is None else round(aa_b, 3),
                              aa_on_pct=None if aa_o is None else round(aa_o, 3), noise_pct=round(noise, 3),
                              verdict="FASTER" if d < -noise else ("SLOWER" if d > noise else "WITHIN NOISE"))
    return rows


report = dict(hosts={}, problems=problems, notes=notes)
for node, kinds in sorted(by_host.items(), key=lambda kv_: str(kv_[0])):
    labs = sum(kinds.values(), [])
    kvs = {lab: kv(lab) for lab in labs}
    if len({json.dumps(v) for v in kvs.values()}) > 1:
        notes.append(f"{node}: KV blocks differ {kvs}")
    ns = sorted({int(n) for lab in labs for n in (arms[lab].get("steady") or {})})
    cases = sorted({c for lab in labs for c in (arms[lab].get("prefill_mixed") or {})})
    host = dict(arms={k: sorted(v) for k, v in kinds.items()}, kv_blocks=kvs,
                steady=compare(kinds, steady, ns), prefill_mixed=compare(kinds, pm, cases),
                rollout=compare(kinds, ro, ["rollout"]) if any("rollout" in arms[lab] for lab in labs) else {})
    report["hosts"][node] = host
    print(f"## host {node}  arms {json.dumps(host['arms'])}")
    for sec, unit in (("steady", "ms/step"), ("prefill_mixed", "s"), ("rollout", "s")):
        for key, r in host[sec].items():
            print(f"   {sec:<13} {key:>16}: ref {r['ref']:.5f} {unit}, up_on {r['delta_pct']:+.3f}% "
                  f"(sessions {r['per_session_pct']}), A/A base {r['aa_base_pct']} on {r['aa_on_pct']}, "
                  f"noise {r['noise_pct']} -> {r['verdict']}")
if notes:
    print("NOTES:")
    for x in notes:
        print("  ", x)
if problems:
    print("PROBLEMS:")
    for x in problems:
        print("  ", x)
if a.out:
    json.dump(report, open(a.out, "w"), indent=1)
sys.exit(2 if problems else 0)
