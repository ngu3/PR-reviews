#!/usr/bin/env python3
"""N1: compare the switch phase (padding-state transitions in one warm engine) across arms and within each arm.

    python3 n1_compare.py RESULT_DIR [RESULT_DIR ...] --ref base_off_s1 [--arms final_on_s1,final_off_s1,...]
    python3 n1_compare.py RESULT_DIR [RESULT_DIR ...] --manifest manifest_t16.json

Loads <label>.json(.gz) files that ran the switch phase (mskip_engine.py --phases switch), skipping *.partial.json*.
Checks:
    complete  at least one candidate arm is present, every required arm (from --manifest / --arms) is present and
              unambiguous, and every arm passes check_arm.py for the switch phase (else exit 2, INCOMPLETE);
    same seq  every arm ran the same request-count sequence on the same prompts (prompt hashes);
    warm      inside each arm, batches with the same n give identical rows (token ids, chosen-token logprobs, finish
              reasons; so also lengths): repeats after other padding states do not change the output;
    vs ref    batch by batch, every arm's rows equal the reference arm's rows exactly.
Exit 0 when all hold (PASS), 1 on any mismatch (FAIL, details printed), 2 when a result is incomplete.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_arm import load, problems  # noqa: E402


def key(r):
    return (r["prompt_sha"], tuple(r["token_ids"]), tuple(r["logprobs"] or ()), r["finish_reason"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", help="reference arm label (optional when --manifest gives 'reference')")
    ap.add_argument("--arms", help="comma-separated candidate labels (default: manifest 'identical', else every other "
                                   "complete switch run)")
    ap.add_argument("--manifest", help="JSON with 'reference' and required 'identical' candidate labels")
    a = ap.parse_args()

    ref_label = a.ref
    manifest_arms = None
    if a.manifest:
        m = json.load(open(a.manifest))
        mref = m.get("reference")
        if ref_label and mref and ref_label != mref:
            print(f"INCOMPLETE: --ref {ref_label} disagrees with manifest reference {mref}")
            return 2
        ref_label = ref_label or mref
        manifest_arms = list(m.get("identical") or [])
    if not ref_label:
        print("INCOMPLETE: --ref or --manifest with 'reference' is required")
        return 2

    runs, dup = {}, []
    for d in a.dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.json*"), recursive=True)):
            base = os.path.basename(p)
            if ".partial.json" in base:
                continue
            label = base.split(".json")[0]
            try:
                r = load(p)
            except (OSError, ValueError):
                continue
            if isinstance(r, dict) and "switch" in ((r.get("meta") or {}).get("phases") or []):
                if label in runs:
                    dup.append(label)
                runs[label] = r
    if dup:
        print("INCOMPLETE: duplicate labels", sorted(set(dup)))
        return 2

    if a.arms is not None:
        arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    elif manifest_arms is not None:
        arms = manifest_arms
    else:
        arms = sorted(x for x in runs if x != ref_label)

    if not arms:
        print(f"INCOMPLETE: no candidate arms to compare against {ref_label}")
        return 2
    if ref_label in arms:
        print(f"INCOMPLETE: reference {ref_label} listed as a candidate arm")
        return 2
    if ref_label not in runs or any(x not in runs for x in arms):
        print("INCOMPLETE: missing", [x for x in [ref_label] + arms if x not in runs])
        return 2
    bad = {x: problems(runs[x], ["switch"]) for x in set(arms) | {ref_label}}
    if any(bad.values()):
        for x, p in bad.items():
            for q in p:
                print(f"INCOMPLETE {x}: {q}")
        return 2
    fails = []
    ref = runs[ref_label]["switch"]
    seq = [b["n"] for b in ref]
    for x in arms:
        sw = runs[x]["switch"]
        if [b["n"] for b in sw] != seq:
            fails.append(f"{x}: sequence {[b['n'] for b in sw]} != {seq}")
            continue
        # warm repeats inside the arm
        first = {}
        for b in sw:
            rows = [key(r) for r in b["rows"]]
            if b["n"] in first and first[b["n"]] != rows:
                diff = sum(1 for u, v in zip(first[b["n"]], rows) if u != v)
                fails.append(f"{x}: batch {b['i']} (n={b['n']}) differs from the first n={b['n']} batch in {diff} rows")
            first.setdefault(b["n"], rows)
        # against the reference arm
        for b, br in zip(sw, ref):
            rows, rrows = [key(r) for r in b["rows"]], [key(r) for r in br["rows"]]
            if [r[0] for r in rows] != [r[0] for r in rrows]:
                fails.append(f"{x}: batch {b['i']} prompts differ from {ref_label}")
                continue
            diff = [j for j, (u, v) in enumerate(zip(rows, rrows)) if u != v]
            if diff:
                j = diff[0]
                u, v = b["rows"][j], br["rows"][j]
                k = next((t for t, (p, q) in enumerate(zip(u["token_ids"], v["token_ids"])) if p != q), None)
                fails.append(f"{x}: batch {b['i']} (n={b['n']}) differs from {ref_label} in {len(diff)} rows; row {j}: "
                             f"first token diff at {k}, lengths {len(u['token_ids'])} / {len(v['token_ids'])}, "
                             f"finish {u['finish_reason']} / {v['finish_reason']}")
    tokens = sum(len(r["token_ids"]) for b in ref for r in b["rows"])
    for f in fails:
        print("FAIL", f)
    print(f"{'N1 PASS' if not fails else 'N1 FAIL'}: {len(arms)} arms vs {ref_label}; sequence {seq}; "
          f"{tokens} reference tokens")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
