#!/usr/bin/env python3
"""t21 (PR #113 follow-up) same-host steady-state decode comparison and the pre-registered verdicts.

    python3 t21_compare.py RESULTS_DIR --experiment copy|gatezero [--out t21.json] [--min-sens 0.10]
                           [--min-hosts 2] [--reps 7] [--allow-incomplete]

RESULTS_DIR holds the mskip_engine.py JSONs of plan t21 (<tree>_<on|off>_<label>.json or .json.gz, steady phase,
--probe-workers). Arms are grouped by host (meta.node); every per-step time is the median over the pairs of
(t(545) - t(33)) / 512 at n requests (all n decoding).

Manifest (what a host must have before any verdict is computed; anything missing makes the host INCOMPLETE):
  copy      arms base_off x2 (stock A/A), optans_on, optmc_on, optnb_on; points n = 1, 2, 4, 8, 15, 16, 255, 256
  gatezero  arms base_off x2 (stock A/A), gz_on x2 (gate-zero A/A); points n = 1, 15, 16, 256
  every arm: a host identity (meta.node or meta.host; arms without one are never counted and make the result
  INCOMPLETE), done, every required point present with at least --reps samples; count arms (optans / optmc / optnb)
  report padding_rows "count" and no stock graph on every worker.
The decision needs at least --min-hosts complete hosts. If fewer, the result is INCOMPLETE (exit 2) and no decision is
printed; --allow-incomplete still prints the per-host numbers of the incomplete hosts (labelled INCOMPLETE).

Experiment 1 (copy): only the per-step copy of the valid-row count differs
  (optans: padding mask + count, 2 synchronous copies; optmc: count only, 1; optnb: count non-blocking, 0).
  noise(n) = max(|stock A/A(n)|, --min-sens) in % ;
  per host, n = 1: FASTER if (variant - optans) / optans < -noise(1); the pre-registered adoption threshold is met
  only if on every complete host the same variant is FASTER at n = 1 and its sign is negative at n = 2, 4, 8;
  full bucket n = 16 / 256: SMALLER if (variant - ref) < (optans - ref) - noise, required on every host.
Experiment 2 (gatezero): ref = mean of the host's stock sessions, gz = mean of its gz sessions,
  noise(n) = max(|stock A/A|, |gz A/A|, --min-sens);
  old regression GONE on a host if (gz - ref) / ref <= noise at n = 15, 16 and 256; benefit KEPT if n = 1 <= -20%;
  the decision requires both on every complete host.
KV-block differences within a host (flag-on arms can allocate the padding state before the KV cache, one block fewer)
are listed as notes. Exit 0 when the result is complete, 2 when INCOMPLETE or a check fails.
"""
import argparse
import glob
import gzip
import json
import os
import statistics
import sys
from collections import defaultdict

MANIFEST = {
    "copy": {"arms": {"base_off": 2, "optans_on": 1, "optmc_on": 1, "optnb_on": 1},
             "points": (1, 2, 4, 8, 15, 16, 255, 256)},
    "gatezero": {"arms": {"base_off": 2, "gz_on": 2}, "points": (1, 15, 16, 256)},
}
COUNT_ARMS = ("optans_on", "optmc_on", "optnb_on")


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--experiment", required=True, choices=sorted(MANIFEST))
    ap.add_argument("--out")
    ap.add_argument("--min-sens", type=float, default=0.10)
    ap.add_argument("--min-hosts", type=int, default=2)
    ap.add_argument("--reps", type=int, default=7, help="samples required per point")
    ap.add_argument("--allow-incomplete", action="store_true")
    return ap.parse_args(argv)


def kind_of(lab):
    return "_".join(lab.split("_")[:2])


def load_arms(results):
    arms = {}
    paths = sorted(glob.glob(os.path.join(results, "**", "*.json"), recursive=True)
                   + glob.glob(os.path.join(results, "**", "*.json.gz"), recursive=True))
    for p in paths:
        if ".partial." in p:
            continue
        try:
            d = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
        except (OSError, ValueError):
            continue
        if "steady" not in d:
            continue
        arms[d.get("meta", {}).get("label") or os.path.basename(p).split(".json")[0]] = d
    return arms


def pct(x, ref):
    return (x / ref - 1.0) * 100.0


def r3(x):
    return None if x is None else round(x, 3)


def analyze(a):
    man = MANIFEST[a.experiment]
    arms = load_arms(a.results)
    problems, notes = [], []
    by_host = defaultdict(lambda: defaultdict(list))
    for lab, d in arms.items():
        k = kind_of(lab)
        if k not in man["arms"]:
            continue  # arms of the other experiment
        node = (d.get("meta") or {}).get("node") or (d.get("meta") or {}).get("host")
        if not node:
            # Same-host pairing cannot be shown without a host identity: never count such arms as a host.
            problems.append(f"{lab}: no host identity (meta.node / meta.host); not counted")
            continue
        by_host[node][k].append(lab)

    def samples(lab, n):
        v = (arms[lab].get("steady") or {}).get(str(n))
        return list(v["ms_per_step"]) if v and v.get("ms_per_step") else []

    def med(lab, n):
        return statistics.median(samples(lab, n))

    report = {"experiment": a.experiment, "manifest": {"arms": man["arms"], "points": list(man["points"]),
                                                       "reps": a.reps},
              "hosts": {}, "problems": problems, "notes": notes}
    complete_hosts = []
    for node, kinds in sorted(by_host.items(), key=lambda kv_: str(kv_[0])):
        labs = sorted(sum(kinds.values(), []))
        missing = []
        for k, need in man["arms"].items():
            have = len(kinds.get(k, []))
            if have < need:
                missing.append(f"arm {k}: {have} of {need}")
        for lab in labs:
            d = arms[lab]
            if not d.get("done"):
                missing.append(f"{lab}: not done")
            for n in man["points"]:
                got = len(samples(lab, n))
                if got < a.reps:
                    missing.append(f"{lab}: n={n} has {got} of {a.reps} samples")
            if kind_of(lab) in COUNT_ARMS:
                w = (d.get("workers") or {}).get("after_start")
                if not (isinstance(w, list) and w):
                    missing.append(f"{lab}: no worker report")
                else:
                    pol = {((x.get("bf16_policy") or {}).get("padding_rows")) for x in w}
                    sg = {json.dumps(x.get("stock_graph")) for x in w}
                    if pol != {"count"}:
                        problems.append(f"{node}: {lab}: padding_rows {sorted(map(str, pol))}, expected count")
                    if sg != {"null"}:
                        problems.append(f"{node}: {lab}: stock graph present {sorted(sg)}")
        kvs = {}
        for lab in labs:
            w = (arms[lab].get("workers") or {}).get("after_start")
            kvs[lab] = sorted({x.get("kv_num_blocks") for x in w}) if isinstance(w, list) else None
        if len({json.dumps(v) for v in kvs.values() if v is not None}) > 1:
            notes.append(f"{node}: KV blocks differ {kvs}")
        host = {"arms": {k: sorted(v) for k, v in kinds.items()}, "kv_blocks": kvs, "missing": missing,
                "status": "INCOMPLETE" if missing else "COMPLETE", "steady": {}, "verdicts": {}}
        report["hosts"][node] = host
        if missing and not a.allow_incomplete:
            continue
        bases = sorted(kinds.get("base_off", []), key=lambda lab: arms[lab]["meta"].get("start", lab))
        for n in man["points"]:
            bm = [med(b, n) for b in bases if samples(b, n)]
            if not bm:
                continue
            ref = statistics.mean(bm)
            row = {"ref_ms": round(ref, 5), "stock_ms": [round(x, 5) for x in bm],
                   "aa_pct": r3(pct(bm[1], bm[0])) if len(bm) > 1 else None}
            for k in COUNT_ARMS:
                for lab in kinds.get(k, []):
                    if samples(lab, n):
                        row[k] = {"ms": round(med(lab, n), 5), "vs_ref_pct": r3(pct(med(lab, n), ref))}
            if "optans_on" in row:
                for k in ("optmc_on", "optnb_on"):
                    if k in row:
                        row[k]["vs_optans_pct"] = r3(pct(row[k]["ms"], row["optans_on"]["ms"]))
            gz = [med(lab, n) for lab in sorted(kinds.get("gz_on", [])) if samples(lab, n)]
            if gz:
                row["gz_on"] = {"ms": [round(x, 5) for x in gz], "vs_ref_pct": r3(pct(statistics.mean(gz), ref)),
                                "gz_aa_pct": r3(pct(gz[1], gz[0])) if len(gz) > 1 else None}
            host["steady"][str(n)] = row
        if missing:
            continue  # numbers printed for --allow-incomplete, no verdicts
        complete_hosts.append(node)
        st = host["steady"]

        def noise(n, extra=()):
            row = st[str(n)]
            vals = [abs(row["aa_pct"])] + [abs(x) for x in extra if x is not None]
            return max(vals + [a.min_sens])

        if a.experiment == "copy":
            for k in ("optmc_on", "optnb_on"):
                d1 = st["1"][k]["vs_optans_pct"]
                small = [st[str(n)][k]["vs_optans_pct"] for n in (2, 4, 8)]
                full = {}
                for n in (16, 256):
                    diff = st[str(n)][k]["vs_ref_pct"] - st[str(n)]["optans_on"]["vs_ref_pct"]
                    full[str(n)] = {"vs_ref_minus_optans_pct": r3(diff), "noise": r3(noise(n)),
                                    "verdict": "SMALLER" if diff < -noise(n) else "NOT SMALLER"}
                host["verdicts"][k] = {"n1_vs_optans_pct": d1, "noise_n1": r3(noise(1)),
                                       "n1": "FASTER" if d1 < -noise(1) else "NOT FASTER",
                                       "n2_4_8_vs_optans_pct": small,
                                       "n2_4_8_all_negative": all(x < 0 for x in small), "full_bucket": full}
        else:
            g = {}
            for n in (15, 16, 256):
                row = st[str(n)]
                nz = noise(n, [row["gz_on"]["gz_aa_pct"]])
                g[str(n)] = {"gz_vs_ref_pct": row["gz_on"]["vs_ref_pct"], "noise": r3(nz),
                             "verdict": "WITHIN NOISE OR FASTER" if row["gz_on"]["vs_ref_pct"] <= nz else "SLOWER"}
            g["regression_gone"] = all(g[str(n)]["verdict"] != "SLOWER" for n in (15, 16, 256))
            g["n1_vs_ref_pct"] = st["1"]["gz_on"]["vs_ref_pct"]
            g["benefit_kept"] = g["n1_vs_ref_pct"] <= -20.0
            host["verdicts"]["gatezero"] = g

    complete = len(complete_hosts) >= a.min_hosts and not problems
    report["complete_hosts"] = complete_hosts
    report["status"] = "COMPLETE" if complete else "INCOMPLETE"
    if len(complete_hosts) < a.min_hosts:
        problems.append(f"{len(complete_hosts)} complete host(s), the decision needs {a.min_hosts}")
    if complete:
        hv = [report["hosts"][h]["verdicts"] for h in complete_hosts]
        if a.experiment == "copy":
            dec = {}
            for k in ("optmc_on", "optnb_on"):
                dec[k] = {"threshold_met": all(v[k]["n1"] == "FASTER" and v[k]["n2_4_8_all_negative"] for v in hv),
                          "full_bucket_smaller": all(all(x["verdict"] == "SMALLER"
                                                         for x in v[k]["full_bucket"].values()) for v in hv)}
            dec["adoption_threshold_met"] = any(dec[k]["threshold_met"] for k in ("optmc_on", "optnb_on"))
        else:
            dec = {"regression_gone": all(v["gatezero"]["regression_gone"] for v in hv),
                   "benefit_kept": all(v["gatezero"]["benefit_kept"] for v in hv)}
            dec["full_acceptance"] = dec["regression_gone"] and dec["benefit_kept"]
        report["decision"] = dec
    return report


def show(report):
    for node, host in report["hosts"].items():
        print(f"== host {node} [{host['status']}]: arms {host['arms']}")
        for m in host["missing"]:
            print(f"   missing: {m}")
        for n, row in host["steady"].items():
            parts = [f"n={n:>3} ref {row['ref_ms']:.4f} ms A/A {row['aa_pct']}%"]
            for k in COUNT_ARMS:
                if k in row:
                    extra = f", vs optans {row[k]['vs_optans_pct']:+.3f}%" if "vs_optans_pct" in row[k] else ""
                    parts.append(f"{k} {row[k]['vs_ref_pct']:+.3f}%{extra}")
            if "gz_on" in row:
                parts.append(f"gz_on {row['gz_on']['vs_ref_pct']:+.3f}% (gz A/A {row['gz_on']['gz_aa_pct']}%)")
            print("   " + " | ".join(parts))
        for e, v in host["verdicts"].items():
            print(f"   verdict {e}: {json.dumps(v)}")
    print(f"STATUS {report['status']} ({len(report['complete_hosts'])} complete host(s))")
    if "decision" in report:
        print(f"DECISION {json.dumps(report['decision'])}")
    if report["notes"]:
        print("NOTES:\n  " + "\n  ".join(report["notes"]))
    if report["problems"]:
        print("PROBLEMS:\n  " + "\n  ".join(report["problems"]))


def main(argv=None):
    a = parse_args(argv)
    report = analyze(a)
    show(report)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
