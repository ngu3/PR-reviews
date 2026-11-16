#!/usr/bin/env python3
"""Summarize the MoeSkip test runs (<arm>.json or <arm>.json.gz from mskip_engine.py, any depth below each RESULTS_DIR).

  python3 analyze_mskip.py RESULTS_DIR [RESULTS_DIR ...] [--ref LABEL] [--fig FIG.png] [--check [--manifest M.json]]

Arms: <tree>_<flag>_<rep>; tree base = vllm-torchtpu f640d734, fix = + the unquantized MoeSkip wiring (5f53fe77),
final = + the cached flag passed to the kernel prebuild and selection (7e4d170a; the t3 image has 5384ef82, same src/); flag on = TPU_MOE_SKIP_PADDED_TOKENS=1,
off = unset. <arm>.partial.json (a failed arm) is never read as a run.

Every run must be complete for the phases its meta lists (check_arm.py: all numerics cases with the right number of
prompts and one logprob per token, every requested sweep point and repeat, the rollout repeats); otherwise the script
lists the problems and exits 2 (--allow-incomplete reports them and goes on).

Prints markdown tables:
  1. runs: group, node, tree, flag, whether unquantized.py zeroes padded routes, file hashes, engine init time;
  2. numerics: bit-identical groups over the three greedy cases (n1 / n10 / n40: token ids and chosen-token
     logprobs), and every run against the reference (--ref): identical prompts per case, share of equal logprobs over
     the tokens before the two outputs diverge, and the nonzero |delta logprob| median / p99 / max (nats);
  3. decode time per step (ms) against the number of requests: median of the repeats per run, mean per configuration,
     and the same-host pairs (both arms of a group ran on one host, one after the other);
  4. the rollout-shaped batch (256 requests with Round-3 response lengths): wall time per run and per pair.
Writes summary.json into the first RESULTS_DIR; --fig draws the figure.

--check exits 1 unless the reference exists and every checked run is bit-identical to it: the runs listed in
--manifest ({"reference": LABEL, "identical": [LABEL, ...]}; each must be present), or else every run with numerics.
Use it on in-process runs (VLLM_ENABLE_V1_MULTIPROCESSING=0): with the default multi-process engine the batch of the
first step can depend on when the requests reach the engine core, and batched outputs differ between runs even for
the same code (results/t2).

  python3 analyze_mskip.py ../results/t2 --fig ../figures/fig_mskip_decode_step.png
  python3 analyze_mskip.py ../results/t2det --ref base_off_p1 --check --manifest manifest_t2det.json
  python3 analyze_mskip.py ../results/t3 ../results/t2det --ref base_off_p1 --check --manifest manifest_t3.json
"""
import argparse
import glob
import gzip
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_arm import problems  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--fig")
ap.add_argument("--check", action="store_true")
ap.add_argument("--manifest", help="JSON with the reference and the runs that must be bit-identical to it")
ap.add_argument("--ref", default="base_off_a", help="reference run for the numerics comparison")
ap.add_argument("--allow-incomplete", action="store_true")
a = ap.parse_args()

CASES = ("n1", "n10", "n40")
REF = a.ref
R, SRC = {}, {}
for root in a.results:
    for p in sorted(glob.glob(os.path.join(root, "**", "*.json*"), recursive=True)):
        base = os.path.basename(p)
        if (base.startswith("summary") or base.startswith("manifest") or ".partial." in base
                or not (base.endswith(".json") or base.endswith(".json.gz"))):
            continue
        d = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
        if "meta" not in d or ("single" in d and "sha256_16" not in d["meta"]):
            continue
        label = d["meta"]["label"]
        if label in R:
            sys.exit(f"duplicate run {label}: {SRC[label]} and {p}")
        R[label], SRC[label] = d, p
labels = sorted(R, key=lambda k: (k.split("_")[0] != "base", k))

incomplete = {k: problems(R[k]) for k in labels}
incomplete = {k: v for k, v in incomplete.items() if v}
for k, v in incomplete.items():
    print(f"INCOMPLETE {k} ({SRC[k]}): " + "; ".join(v[:5]) + (f"; ... {len(v) - 5} more" if len(v) > 5 else ""))
if incomplete and not a.allow_incomplete:
    sys.exit(2)


def cfg(k):
    return k.rsplit("_", 1)[0]


def signature(k):
    return tuple(tuple(r["token_ids"]) + tuple(r["logprobs"] or ()) for c in CASES for r in R[k]["cases"][c])


def compare(x_label, y_label, case):
    xs, ys = R[x_label]["cases"][case], R[y_label]["cases"][case]
    if len(xs) != len(ys) or any(x["prompt_sha"] != y["prompt_sha"] for x, y in zip(xs, ys)):
        sys.exit(f"{x_label} and {y_label} have different prompts in case {case}")
    same, diffs = 0, []
    for x, y in zip(xs, ys):
        ta, tb = x["token_ids"], y["token_ids"]
        same += ta == tb and x["logprobs"] == y["logprobs"]
        n = next((j for j, (u, v) in enumerate(zip(ta, tb)) if u != v), min(len(ta), len(tb)))
        diffs += [abs(x["logprobs"][j] - y["logprobs"][j]) for j in range(n)]
    nz = sorted(d for d in diffs if d > 0)
    return dict(prompts=len(xs), identical_prompts=same, tokens_before_divergence=len(diffs),
                equal_logprob_share=(len(diffs) - len(nz)) / len(diffs) if diffs else 1.0,
                nonzero_median=statistics.median(nz) if nz else 0.0,
                nonzero_p99=nz[min(len(nz) - 1, int(0.99 * len(nz)))] if nz else 0.0, max=nz[-1] if nz else 0.0)


summary = {"runs": {}, "numerics": {}, "sweep": {}, "rollout": {}, "pairs": {}}

# 1. runs
print("## runs\n")
print("| run | group | node | moe_skip flag | unquantized.py zeroes padded routes | unquantized.py | token_padding.py "
      "| engine init (s) |")
print("|---|---|---|---|---|---|---|---:|")
for k in labels:
    m = R[k]["meta"]
    h = m["sha256_16"]
    summary["runs"][k] = {x: m.get(x) for x in ("group", "node", "host", "moe_skip_env", "moe_skip_flag",
                                                 "unquantized_has_fix", "sha256_16", "engine_init_s", "vllm",
                                                 "vllm_torchtpu_file")}
    print(f"| {k} | {m.get('group')} | {m.get('node')} | {m['moe_skip_flag']} | {m['unquantized_has_fix']} | "
          f"`{h['layers/adapter/quantization/unquantized.py'][:8]}` | `{h['layers/adapter/token_padding.py'][:8]}` | "
          f"{m.get('engine_init_s')} |")

# 2. numerics
num = [k for k in labels if "numerics" in (R[k]["meta"].get("phases") or []) and k not in incomplete]
groups = {}
for k in num:
    groups.setdefault(signature(k), []).append(k)
summary["numerics"]["identical_groups"] = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
print("\n## numerics\n\nbit-identical groups:", summary["numerics"]["identical_groups"])
if REF in num:
    print(f"\n| run vs {REF} | n1 identical | n10 identical | n40 identical | equal logprobs before divergence "
          "(n10 / n40) | nonzero abs delta median / p99 / max (worst case) |")
    print("|---|---:|---:|---:|---|---|")
    summary["numerics"]["vs_" + REF] = {}
    for k in num:
        if k == REF:
            continue
        c = {case: compare(REF, k, case) for case in CASES}
        summary["numerics"]["vs_" + REF][k] = c
        worst = max(c.values(), key=lambda v: v["max"])
        print(f"| {k} | {c['n1']['identical_prompts']}/1 | {c['n10']['identical_prompts']}/10 | "
              f"{c['n40']['identical_prompts']}/40 | {c['n10']['equal_logprob_share']:.1%} / "
              f"{c['n40']['equal_logprob_share']:.1%} | {worst['nonzero_median']:.1e} / {worst['nonzero_p99']:.3f} / "
              f"{worst['max']:.3f} |")

# 3. decode time per step
sw = [k for k in labels if "sweep" in R[k]]
ns = sorted({int(n) for k in sw for n in R[k]["sweep"]})
# median of the timed repeats: one repeat can be off when the prefill of a batch is split differently
best = {k: {int(n): statistics.median(v["ms_per_step"]) for n, v in R[k]["sweep"].items()} for k in sw}
ORDER = ("base_off", "base_on", "fix_off", "fix_on", "final_off", "final_on")
cfgs = sorted({cfg(k) for k in sw}, key=lambda c: ORDER.index(c) if c in ORDER else 9)


def _mean_or_none(values):
    values = list(values)
    return statistics.mean(values) if values else None


# Runs of one configuration may cover different request counts (results directories with different sweeps).
mean_cfg = {c: {n: _mean_or_none(best[k][n] for k in sw if cfg(k) == c and n in best[k]) for n in ns} for c in cfgs}
summary["sweep"] = {"median_ms_per_step": best, "mean_by_config": mean_cfg}
print("\n## decode time per step (ms, median of the repeats; mean over runs of a configuration)\n")
print("| requests | " + " | ".join(f"{c} (runs)" for c in cfgs) + " | fix_on / base_off |")
print("|---:|" + "---:|" * (len(cfgs) + 1))
for n in ns:
    cells = []
    for c in cfgs:
        vals = [best[k][n] for k in sw if cfg(k) == c and n in best[k]]
        cells.append(f"{statistics.mean(vals):.3f} ({len(vals)}; {min(vals):.3f}–{max(vals):.3f})" if vals else "—")
    f_on, b_off = (mean_cfg.get("fix_on") or {}).get(n), (mean_cfg.get("base_off") or {}).get(n)
    ratio = (f_on / b_off - 1) if f_on is not None and b_off is not None else None
    print(f"| {n} | " + " | ".join(cells) + (f" | {ratio:+.1%} |" if ratio is not None else " | |"))

# same-host pairs: arms of one group that ran on the same node
by_group = {}
for k in labels:
    m = R[k]["meta"]
    by_group.setdefault((m.get("group"), m.get("node")), []).append(k)
pairs = []
for (g, node), ks in sorted(by_group.items(), key=lambda x: str(x[0])):
    for i, x in enumerate(ks):
        for y in ks[i + 1:]:
            if cfg(x) != cfg(y) and ((x in best and y in best) or ('rollout' in R[x] and 'rollout' in R[y])):
                # put the reference configuration second: base_off < fix_off < base_on < fix_on
                order = ("final_on", "fix_on", "base_on", "final_off", "fix_off", "base_off")
                t, r = sorted((x, y), key=lambda z: order.index(cfg(z)) if cfg(z) in order else 9)
                pairs.append((g, node, t, r))
if pairs:
    print("\n## same-host pairs: decode time per step, test / reference - 1\n")
    print("| group | test vs reference | " + " | ".join(f"n={n}" for n in ns) + " | rollout wall |")
    print("|---|---|" + "---:|" * (len(ns) + 1))
    for g, node, t, r in pairs:
        both = [n for n in ns if n in best.get(t, {}) and n in best.get(r, {})]
        cells = [f"{best[t][n] / best[r][n] - 1:+.1%}" if n in both else "" for n in ns]
        rw = (f"{R[t]['rollout']['best'] / R[r]['rollout']['best'] - 1:+.1%}"
              if "rollout" in R[t] and "rollout" in R[r] else "")
        summary["pairs"][f"{g}:{t}/{r}"] = dict(node=node, step={n: best[t][n] / best[r][n] - 1 for n in both},
                                                rollout=rw)
        print(f"| {g} | {t} / {r} | " + " | ".join(cells) + f" | {rw} |")

# 4. rollout-shaped batch
ro = [k for k in labels if "rollout" in R[k]]
if ro:
    print("\n## rollout-shaped batch (256 requests, Round-3 lengths), wall time (s)\n")
    print("| run | group | reps | best |")
    print("|---|---|---|---:|")
    for k in ro:
        r = R[k]["rollout"]
        summary["rollout"][k] = r
        print(f"| {k} | {R[k]['meta'].get('group')} | {' / '.join(f'{x:.2f}' for x in r['wall_s'])} | {r['best']:.2f} |")
    for c in cfgs:
        vals = [R[k]["rollout"]["best"] for k in ro if cfg(k) == c]
        if vals:
            print(f"\n{c}: mean of best {statistics.mean(vals):.2f} s over {len(vals)} runs "
                  f"({min(vals):.2f}–{max(vals):.2f})", end="")
    print()

json.dump(summary, open(os.path.join(a.results[0], "summary.json"), "w"), indent=1, sort_keys=True)

if a.fig and sw:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"base_off": "#444444", "base_on": "#9a9a9a", "fix_off": "#1f77b4", "fix_on": "#d62728",
              "final_off": "#17becf", "final_on": "#ff7f0e"}
    names = {"base_off": "main, flag off", "base_on": "main, flag on (no fix)", "fix_off": "fix, flag off",
             "fix_on": "fix, flag on", "final_off": "final fix, flag off", "final_on": "final fix, flag on"}
    ncol = 3 if ro else 2
    fig, axs = plt.subplots(1, ncol, figsize=(6.2 * ncol, 4.6))
    ax1, ax2 = axs[0], axs[1]
    for c in cfgs:
        runs = [k for k in sw if cfg(k) == c]
        ys = [mean_cfg[c][n] if mean_cfg[c][n] is not None else float("nan") for n in ns]
        dashed = c in ("base_on", "fix_off")
        ax1.plot(ns, ys, marker="o", ms=3.5, color=colors.get(c), ls="--" if dashed else "-", lw=1.2 if dashed else 1.8,
                 label=f"{names.get(c, c)} ({len(runs)} run{'s' if len(runs) > 1 else ''})")
    for b in (16, 32, 64, 128, 256):
        if ns[0] <= b <= ns[-1]:
            ax1.axvline(b, color="#cccccc", lw=0.8, ls=":")
            ax2.axvline(b, color="#cccccc", lw=0.8, ls=":")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels(ns, fontsize=8)
        ax.set_xlabel("requests decoding together (dotted: token buckets 16 / 32 / 64 ...)")
        ax.grid(alpha=0.25)
    ax1.set_yscale("log")
    yt = [4, 5, 6, 8, 10, 15, 20, 30]
    ax1.set_yticks(yt)
    ax1.set_yticklabels(yt)
    ax1.set_ylabel("ms per decode step (median of repeats, mean over runs)")
    ax1.set_title("(a) decode step time")
    ax1.legend(fontsize=8)
    if {"fix_on", "base_off"} <= set(cfgs):
        base = mean_cfg["base_off"]

        def _rel(x, ref):
            return 100 * (x / ref - 1) if x is not None and ref is not None else float("nan")

        for c in ("fix_on", "base_on", "fix_off"):
            if c not in cfgs:
                continue
            runs = [k for k in sw if cfg(k) == c]
            ax2.plot(ns, [_rel(mean_cfg[c][n], base[n]) for n in ns], marker="o", ms=3.5, color=colors[c],
                     ls="--" if c != "fix_on" else "-", lw=1.8 if c == "fix_on" else 1.2,
                     label=f"{names[c]} (mean of {len(runs)})")
            if c == "fix_on" and len(runs) > 1:
                lo = [_rel(min((best[k][n] for k in runs if n in best[k]), default=None), base[n]) for n in ns]
                hi = [_rel(max((best[k][n] for k in runs if n in best[k]), default=None), base[n]) for n in ns]
                ax2.fill_between(ns, lo, hi, color=colors[c], alpha=0.15, lw=0)
        for n in ns:
            v = _rel(mean_cfg["fix_on"][n], base[n])
            if n in (1, 16, 24) and v == v:
                ax2.annotate(f"{v:+.1f}%", (n, v), textcoords="offset points", xytext=(4, -12 if v < 0 else 6),
                             fontsize=8, color=colors["fix_on"])
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_ylabel("change vs main, flag off (%)")
        ax2.set_title("(b) change in decode step time")
        ax2.legend(fontsize=8, loc="lower right")
    if ro:
        ax3 = axs[2]
        xs, hs, cs, labels_ = [], [], [], []
        order = [c for c in ("base_off", "base_on", "fix_off", "fix_on") if any(cfg(k) == c for k in ro)]
        x = 0
        for c in order:
            for k in sorted(k for k in ro if cfg(k) == c):
                xs.append(x)
                hs.append(R[k]["rollout"]["best"])
                cs.append(colors[c])
                labels_.append(k)
                x += 1
            x += 0.6
        ax3.bar(xs, hs, color=cs, width=0.8)
        for xx, hh in zip(xs, hs):
            ax3.text(xx, hh + 0.15, f"{hh:.2f}", ha="center", fontsize=7)
        ax3.set_xticks(xs)
        ax3.set_xticklabels(labels_, rotation=60, fontsize=7, ha="right")
        ax3.set_ylim(0, max(hs) * 1.12)
        ax3.set_ylabel("wall time, best of 3 (s)")
        ax3.set_title("(c) 256 requests with Round-3 response lengths")
        ax3.grid(alpha=0.25, axis="y")
    fig.suptitle("MoeSkip in the bf16 MoE: Qwen3.5-35B-A3B, TP=4 on one TPU v7x host, vllm-torchtpu main (f640d734)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(a.fig, dpi=130)
    print("wrote", a.fig)

if a.check:
    if REF not in num:
        sys.exit(f"CHECK FAIL: reference {REF} is missing or has no complete numerics")
    if a.manifest:
        man = json.load(open(a.manifest))
        if man.get("reference", REF) != REF:
            sys.exit(f"CHECK FAIL: manifest reference {man['reference']} != --ref {REF}")
        checked = list(man["identical"])
        missing = [k for k in checked if k not in num]
        if missing:
            sys.exit(f"CHECK FAIL: runs missing or without complete numerics: {missing}")
    else:
        checked = [k for k in num if k != REF]
    if not checked:
        sys.exit("CHECK FAIL: nothing to compare with the reference")
    ref_sig = signature(REF)
    differ = [k for k in checked if signature(k) != ref_sig]
    print("CHECK", "PASS" if not differ else "FAIL",
          f"({len(checked)} runs against {REF}; differ: {differ if differ else 'none'})")
    raise SystemExit(0 if not differ else 1)
