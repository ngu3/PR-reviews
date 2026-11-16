#!/usr/bin/env python3
"""Compare the GSM8K quality phase (mskip_engine.py --phases ...,quality) of arms against a reference arm.

    python3 quality_compare.py RESULTS_DIR --ref upbase_off --arms up3_on,up3_off [--out report.json]

An arm's kind is the first two fields of its label (<tree>_<on|off>_<rep> -> <tree>_<on|off>). For every pass
(greedy, sampled) and kind: accuracy (strict "#### <number>" and flexible last number, verl's GSM8K rule), output
tokens, requests cut at max_tokens, and against the reference: requests with identical token ids, identical
answers, and the first differing request. Exit 0 when every requested kind has both passes with the reference's
prompts in the same order, 2 otherwise.
"""
import argparse
import glob
import gzip
import json
import os
import sys


def load(p):
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    kinds = [a.ref] + a.arms.split(",")
    found = {}
    for p in sorted(glob.glob(os.path.join(a.results, "**", "*.json*"), recursive=True)):
        if ".partial." in p or not (p.endswith(".json") or p.endswith(".json.gz")):
            continue
        d = load(p)
        if not isinstance(d, dict) or "quality" not in d:
            continue
        label = (d.get("meta") or {}).get("label") or os.path.basename(p).split(".json")[0]
        kind = "_".join(label.split("_")[:2])
        if kind in kinds:
            found.setdefault(kind, []).append((label, d["quality"]))
    report, ok = {}, True
    for kind in kinds:
        if kind not in found:
            print(f"MISSING kind {kind}")
            ok = False
    if not ok:
        return 2
    ref_label, ref_q = found[a.ref][0]
    for name in ("greedy", "sampled"):
        rp = ref_q.get(name)
        if rp is None:
            print(f"MISSING {name} in {ref_label}")
            return 2
        print(f"## {name} (temperature {rp.get('temperature')}, max_tokens {rp.get('max_tokens')}, n {rp['n']})")
        for kind in kinds:
            for label, q in found[kind]:
                p = q.get(name)
                if p is None:
                    print(f"   {label}: MISSING")
                    ok = False
                    continue
                line = (f"   {label:24s} strict {p['correct_strict']}/{p['n']} ({100 * p['acc_strict']:.2f}%)"
                        f"  flexible {p['correct_flexible']}/{p['n']} ({100 * p['acc_flexible']:.2f}%)"
                        f"  tokens {p['tokens']}  cut at max_tokens {p['finish_length']}")
                entry = dict(label=label, acc_strict=p["acc_strict"], acc_flexible=p["acc_flexible"],
                             tokens=p["tokens"], finish_length=p["finish_length"])
                if label != ref_label:
                    rr, xr = rp["rows"], p["rows"]
                    if [r["prompt_sha"] for r in rr] != [r["prompt_sha"] for r in xr]:
                        line += "  PROMPTS DIFFER"
                        ok = False
                    same_ids = sum(r["token_ids"] == x["token_ids"] for r, x in zip(rr, xr))
                    same_ans = sum(r["pred_strict"] == x["pred_strict"] for r, x in zip(rr, xr))
                    first = next((i for i, (r, x) in enumerate(zip(rr, xr)) if r["token_ids"] != x["token_ids"]),
                                 None)
                    line += f"  | vs {ref_label}: identical token ids {same_ids}/{len(rr)}, same answer {same_ans}"
                    if first is not None:
                        line += f", first difference at request {first}"
                    entry.update(identical_token_ids=same_ids, same_answer=same_ans, first_difference=first)
                print(line)
                report.setdefault(name, []).append(entry)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
