#!/usr/bin/env python3
"""t28 verdict: blocking padding-state copies and async-scheduling overlap (pre-registered 2026-11-16 23:50 UTC).

    python3 t28_verdict.py RESULTS_DIR [--points 1,8,16,64,256]

Per host (meta.node) and point n, med = median per-step ms of an arm's steady phase; kinds with two sessions use
their mean. noise = max(|A/A upbase|, |A/A up5|, 0.10%).
  C1 (copy cost): d_nb = up5nb / up5 - 1 (also d_mc for up5mc). Holds at n if on every host d_nb < -noise.
  C2 (overlap):   gap = (up5 / upbase - 1) - (up5sy / upbasesy - 1). Holds at n if on every host gap > noise.
                  Also printed: async_gain_main = upbase / upbasesy - 1 (negative: async scheduling is faster).
  C3 (sanity):    meta.async_scheduling is True for every arm without "sy" in its tree and False for "sy" arms.
"""
import argparse
import glob
import gzip
import json
import os
import statistics
import sys
from collections import defaultdict

KINDS = ("upbasek9574_off", "up5k9574_on", "up5mck9574_on", "up5nbk9574_on", "upbasek9574sy_off", "up5k9574sy_on")


def load(results):
    arms = {}
    for p in sorted(glob.glob(os.path.join(results, "**", "*.json*"), recursive=True)):
        if ".partial." in p or not (p.endswith(".json") or p.endswith(".json.gz")):
            continue
        d = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
        if isinstance(d, dict) and "steady" in d:
            arms[d["meta"]["label"]] = d
    return arms


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--points", default="1,8,16,64,256")
    a = ap.parse_args(argv)
    arms = load(a.results)
    points = [int(x) for x in a.points.split(",")]
    by_host = defaultdict(lambda: defaultdict(list))
    c3 = []
    for lab, d in sorted(arms.items(), key=lambda kv: kv[1]["meta"].get("start", kv[0])):
        kind = "_".join(lab.split("_")[:2])
        if kind not in KINDS:
            continue
        by_host[d["meta"].get("node") or d["meta"].get("host")][kind].append(lab)
        want = "sy_" not in lab
        got = d["meta"].get("async_scheduling")
        c3.append((lab, got, got is want))
    print("C3 async_scheduling per arm:")
    for lab, got, ok in c3:
        print(f"   {lab:24s} {got!s:6s} {'ok' if ok else 'UNEXPECTED'}")
    c3_ok = all(ok for _, _, ok in c3)

    def med(lab, n):
        v = arms[lab]["steady"].get(str(n))
        return statistics.median(v["ms_per_step"]) if v else None

    verdict = {"C1_nb": {}, "C2": {}}
    for host, ks in sorted(by_host.items()):
        missing = [k for k in KINDS if not ks.get(k)]
        if missing:
            print(f"host {host}: missing {missing}")
            return 2
        print(f"\n## host {host}  arms: " + ", ".join(f"{k}x{len(v)}" for k, v in ks.items()))
        for n in points:
            m = {k: [med(lab, n) for lab in ks[k]] for k in KINDS}
            mean = {k: statistics.mean(v) for k, v in m.items()}
            aa = {k: (v[1] / v[0] - 1) * 100 for k, v in m.items() if len(v) > 1}
            noise = max(abs(aa.get("upbasek9574_off", 0)), abs(aa.get("up5k9574_on", 0)), 0.10)
            d_nb = (mean["up5nbk9574_on"] / mean["up5k9574_on"] - 1) * 100
            d_mc = (mean["up5mck9574_on"] / mean["up5k9574_on"] - 1) * 100
            d_async = (mean["up5k9574_on"] / mean["upbasek9574_off"] - 1) * 100
            d_sync = (mean["up5k9574sy_on"] / mean["upbasek9574sy_off"] - 1) * 100
            gap = d_async - d_sync
            gain_main = (mean["upbasek9574_off"] / mean["upbasek9574sy_off"] - 1) * 100
            gain_head = (mean["up5k9574_on"] / mean["up5k9574sy_on"] - 1) * 100
            verdict["C1_nb"].setdefault(n, []).append(d_nb < -noise)
            verdict["C2"].setdefault(n, []).append(gap > noise)
            print(f"   n={n:>3} upbase {mean['upbasek9574_off']:.4f} ms (A/A {aa.get('upbasek9574_off', float('nan')):+.3f}%) "
                  f"up5 A/A {aa.get('up5k9574_on', float('nan')):+.3f}% noise {noise:.3f}%")
            print(f"         C1: up5nb vs up5 {d_nb:+.3f}%  up5mc vs up5 {d_mc:+.3f}%")
            print(f"         C2: head vs main async {d_async:+.3f}%  sync {d_sync:+.3f}%  gap {gap:+.3f}%  | "
                  f"async vs sync: main {gain_main:+.3f}%  head {gain_head:+.3f}%")
    print("\nVerdict (pre-registered; a criterion holds at n only if it holds on every host):")
    print(f"   C3 sanity: {'holds' if c3_ok else 'FAILS (C1 / C2 void)'}")
    for n in points:
        print(f"   n={n:>3}  C1 copies cost time: {'holds' if all(verdict['C1_nb'][n]) else 'does not hold'}"
              f"   C2 copies break async overlap: {'holds' if all(verdict['C2'][n]) else 'does not hold'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
