#!/usr/bin/env python3
"""Same-host comparison of steady-state decode arms (mskip_engine.py results), any arm kinds.

    python3 same_host_compare.py RESULTS_DIR --ref upbase_off --arms up2_on,upbar_on [--points 1,16,256]
                                 [--min-sens 0.10] [--out report.json]

An arm's kind is the first two fields of its label (<tree>_<on|off>_<rep> -> <tree>_<on|off>). Arms are grouped by
host (meta.node or meta.host; arms without a host identity are rejected). Per host and point n:
  ref = mean over the host's --ref sessions of the median per-step time (ms) at n;
  for every other kind: delta = (mean of its sessions / ref - 1) in %, each session's delta, and its A/A
  (second session vs first, in %) when it has two sessions; noise = max(|A/A| of ref and of the kind, --min-sens).
Each kind is also reported against every single ref session (useful when ref sessions differ in KV block count).
KV block counts per arm are printed. Exit 0 when every host has at least one ref session and every requested kind,
2 otherwise.
"""
import argparse
import glob
import gzip
import json
import os
import statistics
import sys
from collections import defaultdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--points")
    ap.add_argument("--min-sens", type=float, default=0.10)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    kinds = a.arms.split(",")
    arms, problems = {}, []
    for p in sorted(glob.glob(os.path.join(a.results, "**", "*.json*"), recursive=True)):
        if ".partial." in p or not (p.endswith(".json") or p.endswith(".json.gz")):
            continue
        d = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
        if not isinstance(d, dict) or "steady" not in d:
            continue
        arms[d["meta"]["label"]] = d
    by_host = defaultdict(lambda: defaultdict(list))
    for lab, d in arms.items():
        k = "_".join(lab.split("_")[:2])
        if k not in kinds and k != a.ref:
            continue
        node = d["meta"].get("node") or d["meta"].get("host")
        if not node:
            problems.append(f"{lab}: no host identity")
            continue
        by_host[node][k].append(lab)

    def med(lab, n):
        v = arms[lab]["steady"].get(str(n))
        return statistics.median(v["ms_per_step"]) if v else None

    def kv(lab):
        w = (arms[lab].get("workers") or {}).get("after_start") or []
        return sorted({x.get("kv_num_blocks") for x in w})

    report = {"hosts": {}, "problems": problems}
    for node, ks in sorted(by_host.items()):
        if not ks.get(a.ref):
            problems.append(f"{node}: no {a.ref} session")
            continue
        missing = [k for k in kinds if not ks.get(k)]
        if missing:
            problems.append(f"{node}: missing {missing}")
        order = lambda labs: sorted(labs, key=lambda lab: arms[lab]["meta"].get("start", lab))  # noqa: E731
        refs = order(ks[a.ref])
        points = ([int(x) for x in a.points.split(",")] if a.points
                  else sorted({int(n) for lab in refs for n in arms[lab]["steady"]}))
        host = {"kv": {lab: kv(lab) for k in ks for lab in ks[k]}, "points": {}}
        print(f"## host {node}  kv {json.dumps(host['kv'])}")
        for n in points:
            rm = [med(lab, n) for lab in refs if med(lab, n) is not None]
            if not rm:
                continue
            ref = statistics.mean(rm)
            aa_ref = (rm[1] / rm[0] - 1) * 100 if len(rm) > 1 else None
            row = {"ref_ms": round(ref, 5), "ref_sessions_ms": [round(x, 5) for x in rm],
                   "ref_aa_pct": None if aa_ref is None else round(aa_ref, 3)}
            cells = []
            for k in kinds:
                labs = order(ks.get(k, []))
                ms = [med(lab, n) for lab in labs if med(lab, n) is not None]
                if not ms:
                    continue
                aa = (ms[1] / ms[0] - 1) * 100 if len(ms) > 1 else None
                noise = max([abs(x) for x in (aa_ref, aa) if x is not None] + [a.min_sens])
                delta = (statistics.mean(ms) / ref - 1) * 100
                vs_each = {r: round((statistics.mean(ms) / med(r, n) - 1) * 100, 3) for r in refs}
                row[k] = {"delta_pct": round(delta, 3), "sessions_pct": [round((x / ref - 1) * 100, 3) for x in ms],
                          "aa_pct": None if aa is None else round(aa, 3), "noise_pct": round(noise, 3),
                          "vs_each_ref_session_pct": vs_each}
                cells.append(f"{k} {delta:+.3f}% (A/A {row[k]['aa_pct']}, noise {noise:.3f}; vs each ref "
                             + ", ".join(f"{v:+.2f}" for v in vs_each.values()) + ")")
            host["points"][str(n)] = row
            print(f"   n={n:>3} ref {ref:.4f} ms (A/A {row['ref_aa_pct']}) | " + " | ".join(cells))
        report["hosts"][node] = host
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    return 2 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
