#!/usr/bin/env python3
"""Compare served_check.py results (C1 / S1) across arms of the same layout.

  python3 served_compare.py RESULT_DIR [RESULT_DIR ...] --ref base_off_d1 --arms final_on_d1,final_off_d1

Every arm, the reference included, is checked for completeness first; arms are compared only when all are complete:
  complete    done is true and no error is recorded; meta.phases is not empty and every listed phase is present
              with its expected requests in order (single: prompts 0-9, concurrent: 0-15, uneven: 16-18); every row
              has token ids, a finish reason and one finite logprob per token. Files named *.partial.json are not
              read, and a label found twice is ambiguous. Anything else -> INCOMPLETE (exit 2), no comparison.
  layout      every arm has the reference's tp / dp / ep / extra server env and the same phases -> otherwise FAIL.
  single      deterministic scheduling (one request at a time): every arm must equal the reference exactly
              (token ids, chosen-token logprobs, finish reasons) -> PASS / FAIL.
  concurrent, uneven
              batch composition depends on arrival timing: reported as identical / different (diagnostic only).
Exit 0 PASS, 1 FAIL, 2 INCOMPLETE.
"""
import argparse
import glob
import gzip
import json
import math
import os
import sys

EXPECTED_PROMPTS = {"single": list(range(10)), "concurrent": list(range(16)), "uneven": [16, 17, 18]}


def key(r):
    return (r.get("prompt"), tuple(r.get("token_ids") or ()), tuple(r.get("logprobs") or ()), r.get("finish_reason"))


def problems(r):
    """Why a served_check.py result is not a complete run (empty list when it is)."""
    out = []
    meta = r.get("meta") or {}
    if r.get("done") is not True:
        out.append("done is not true")
    for e in r.get("errors") or []:
        out.append(f"recorded error: {e}")
    phases = meta.get("phases") or []
    if not phases:
        out.append("no phases recorded")
    for ph in phases:
        want = EXPECTED_PROMPTS.get(ph)
        if want is None:
            out.append(f"{ph}: unknown phase")
            continue
        rows = (r.get(ph) or {}).get("rows")
        if rows is None:
            out.append(f"{ph}: missing")
            continue
        got = [x.get("prompt") for x in rows]
        if got != want:
            out.append(f"{ph}: requests {got}, expected {want}")
        for x in rows:
            ids, lps = x.get("token_ids"), x.get("logprobs")
            if "error" in x:
                out.append(f"{ph}: prompt {x.get('prompt')}: {x['error']}")
            elif not ids or not x.get("finish_reason"):
                out.append(f"{ph}: prompt {x.get('prompt')}: no token ids or finish reason")
            elif lps is None or len(lps) != len(ids) or any(
                    not isinstance(v, (int, float)) or not math.isfinite(v) for v in lps):
                out.append(f"{ph}: prompt {x.get('prompt')}: logprobs missing, not one per token or not finite")
    return out


def load(dirs):
    runs, dup = {}, set()
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)
                        + glob.glob(os.path.join(d, "**", "*.json.gz"), recursive=True)):
            if ".partial." in os.path.basename(p):
                continue
            try:
                r = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
            except (OSError, ValueError):
                continue
            if isinstance(r, dict) and "cmd" in (r.get("meta") or {}):
                label = r["meta"]["label"]
                if label in runs:
                    dup.add(label)
                runs[label] = r
    return runs, dup


def layout(r):
    m = r["meta"]
    return (m.get("tp"), m.get("dp"), m.get("ep"), tuple(m.get("env_extra") or ()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arms", required=True)
    a = ap.parse_args()
    runs, dup = load(a.dirs)
    arms = a.arms.split(",")
    labels = [a.ref] + arms
    missing = [x for x in labels if x not in runs]
    ambiguous = [x for x in labels if x in dup]
    incomplete = {x: problems(runs[x]) for x in labels if x in runs}
    incomplete = {x: p for x, p in incomplete.items() if p}
    if missing or ambiguous or incomplete:
        if missing:
            print("INCOMPLETE: missing", missing)
        if ambiguous:
            print("INCOMPLETE: label found in more than one file", ambiguous)
        for x, p in incomplete.items():
            for q in p[:10]:
                print(f"INCOMPLETE: {x}: {q}")
            if len(p) > 10:
                print(f"INCOMPLETE: {x}: ... {len(p) - 10} more")
        print(f"SERVED INCOMPLETE: no comparison ({len(labels)} arms requested)")
        return 2
    fails = []
    ref = runs[a.ref]
    for x in arms:
        r = runs[x]
        if layout(r) != layout(ref):
            fails.append(f"{x}: layout {layout(r)} != reference {layout(ref)}")
        if r["meta"]["phases"] != ref["meta"]["phases"]:
            fails.append(f"{x}: phases {r['meta']['phases']} != reference {ref['meta']['phases']}")
            continue
        for ph in ref["meta"]["phases"]:
            rows, rrows = [key(v) for v in r[ph]["rows"]], [key(v) for v in ref[ph]["rows"]]
            diff = [i for i, (u, v) in enumerate(zip(rows, rrows)) if u != v]
            toks = sum(len(v["token_ids"]) for v in r[ph]["rows"])
            if ph == "single":
                if diff:
                    fails.append(f"{x}: single differs from {a.ref} in requests {[r[ph]['rows'][i]['prompt'] for i in diff]}")
                print(f"  {x} single: {len(rows) - len(diff)}/{len(rows)} requests identical to {a.ref} ({toks} tokens)")
            else:
                print(f"  {x} {ph}: {len(rows) - len(diff)}/{len(rows)} identical to {a.ref} (diagnostic; {toks} tokens)")
    for f in fails:
        print("FAIL", f)
    print(f"{'SERVED PASS' if not fails else 'SERVED FAIL'}: {len(arms)} arms vs {a.ref}; layout tp/dp/ep/env = {layout(ref)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
