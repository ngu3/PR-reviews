#!/usr/bin/env python3
"""N2: direct TPU check of BF16 MoE real-row outputs under Option A (gcount + stock fallback).

    PYTHONPATH=/opt/arm-opta/src python3 moe_direct_check.py --out /out/n2_opta.json

Fixed random BF16 inputs (seeded): hidden states [T, 2048], expert weights in the layout the bf16 method hands to
the kernel after process_weights_after_loading (w1 [E, 2048, 2*I], w2 [E, I, 2048]; E=256, I=128, the TP=4 shard of
Qwen3.5-35B-A3B), and top-8 routes from random router logits (softmax, top-k, renormalized, BF16 weights).

For every bucket T and number of valid rows n (rows n..T-1 are padding), and several routing seeds:
    stock   fused_moe_gmm(weights, ids, skip_padded_tokens=False, padding_rows=None)  (no row_info passed)
    gcount  fused_moe_gmm(weights, ids, skip_padded_tokens=True, padding_rows="count",
                          row_info=TokenPaddingState.local_num_valid)  (no zero_routing_weights_for_padding)

Pass criteria (N2):
  1. Padded rows deliberately retain strictly non-zero routing weights (no front weight zeroing).
  2. Real-row MoE outputs of gcount are bit-identical to stock (and repeatable), finite, and non-trivial;
     every call completes, including n=0 (all padding).
  3. Perturbing padded-row activations and non-zero routing weights leaves gcount real-row outputs bit-identical
     to stock (proving padded-row exclusion is by row_info valid count, not by zero weights).
  4. Real rows with mixed zero-gate + positive-gate slots (and an all-zero-gate real row when n >= 2) while
     padded rows remain non-zero produce real-row MoE outputs bit-identical to stock.
  5. Stock branch (skip_padded_tokens=False, padding_rows=None, row_info omitted) produces identical outputs
     across all T rows regardless of a stale local_num_valid in TokenPaddingState (no row_info leak).

Each call runs inside torch.compile(backend="tpu", fullgraph=True, dynamic=False), matching runner compilation.
Limits: one chip, no EP / DP / pipelined collective; random weights, not the checkpoint; the kernel and
TokenPaddingState are the real ones from the selected source tree.
"""
import argparse
import json
import sys
import time

import torch
import torch_tpu  # noqa: F401  registers the "tpu" device
from torch_tpu._internal import compile as _register_tpu_backend  # noqa: F401  torch.compile(backend="tpu")

from vllm_torchtpu.layers.adapter import fused_moe as fm
from vllm_torchtpu.layers.adapter import token_padding as tp

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--buckets", default="16:0,1,5,12,15,16;32:17,24,31,32;64:33,64;256:250,255,256")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--experts", type=int, default=256)
ap.add_argument("--inter", type=int, default=128)
ap.add_argument("--hidden", type=int, default=2048)
ap.add_argument("--topk", type=int, default=8)
a = ap.parse_args()

dev = torch.device("tpu")
E, I, K, TOPK = a.experts, a.inter, a.hidden, a.topk
gen = torch.Generator().manual_seed(1234)
w1 = (torch.randn(E, K, 2 * I, generator=gen) * 0.02).to(torch.bfloat16).to(dev)
w2 = (torch.randn(E, I, K, generator=gen) * 0.02).to(torch.bfloat16).to(dev)

# Prebuild both stock and count-filter kernels outside tracing, as process_weights_after_loading does.
# The row-count mode was the string padding_rows="count" before the PaddedRowFilter enum; accept either API.
try:
    from vllm_torchtpu.layers.core.fused_moe_gmm import PaddedRowFilter as _PRF
    _STOCK_KW, _COUNT_KW, _API = {}, {"padded_row_filter": _PRF.ROW_COUNT}, "enum"
except ImportError:
    _STOCK_KW, _COUNT_KW, _API = {"padding_rows": None}, {"padding_rows": "count"}, "string"
print("ROW_FILTER_API", _API, flush=True)
fm.prebuild_fused_moe_kernel(topk=TOPK, activation="silu", use_ep=False, skip_padded_tokens=False, **_STOCK_KW)
fm.prebuild_fused_moe_kernel(topk=TOPK, activation="silu", use_ep=False, skip_padded_tokens=True, **_COUNT_KW)


def _moe_stock(x, weights, ids):
    return fm.fused_moe_gmm(
        x, w1, w2, None, None, None, None, weights, ids, None, TOPK, "silu",
        skip_padded_tokens=False, **_STOCK_KW,
    )


# The TokenPaddingState count was renamed local_num_valid -> local_valid_tokens (track_valid_count ->
# track_valid_tokens) on the final upstream branch; accept either API.
_get_count = getattr(tp, "get_local_valid_tokens", None) or tp.get_local_num_valid
_TRACK_KW = ("track_valid_tokens" if "track_valid_tokens" in tp.TokenPaddingState.create.__code__.co_varnames
             else "track_valid_count")


def _moe_gcount(x, weights, ids):
    row_info = _get_count()
    assert row_info is not None, f"{_TRACK_KW}=True must allocate the valid-row count"
    return fm.fused_moe_gmm(
        x, w1, w2, None, None, None, None, weights, ids, None, TOPK, "silu",
        skip_padded_tokens=True, row_info=row_info, **_COUNT_KW,
    )


moe_stock = torch.compile(_moe_stock, backend="tpu", fullgraph=True, dynamic=False)
moe_gcount = torch.compile(_moe_gcount, backend="tpu", fullgraph=True, dynamic=False)
state = tp.TokenPaddingState.create(4096, dev, **{_TRACK_KW: True})
tp.set_padding_state(state)


def routes(T, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator=g)
    probs = torch.softmax(logits, dim=-1)
    w, ids = torch.topk(probs, TOPK, dim=-1)
    w = (w / w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
    # Guarantee strictly positive BF16 weights on all rows so padded rows are non-zero.
    w = torch.clamp(w, min=1e-3).to(torch.bfloat16)
    x = (torch.randn(T, K, generator=g)).to(torch.bfloat16)
    return x.to(dev), w.to(dev), ids.to(torch.int32).to(dev)


def bits(t):
    return t.detach().to("cpu").contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t.detach().to("cpu")


results, ok = [], True
for spec in a.buckets.split(";"):
    T, ns = spec.split(":")
    T = int(T)
    for n in [int(v) for v in ns.split(",")]:
        for seed in [int(s) for s in a.seeds.split(",")]:
            t0 = time.time()
            x, w, ids = routes(T, 1000 * T + seed)

            # Stock branch must ignore any stale local_num_valid in TokenPaddingState across all T rows.
            state.update(1 if T > 1 else 0, T)
            ref_with_stale = bits(moe_stock(x, w, ids))
            state.update(T, T)
            ref = bits(moe_stock(x, w, ids))
            off = bits(moe_stock(x, w, ids))

            # gcount branch: NO zero_routing_weights_for_padding; padded rows keep non-zero weights.
            state.update(n, T)
            on = bits(moe_gcount(x, w, ids))
            on2 = bits(moe_gcount(x, w, ids))

            # Perturb padded rows' hidden states & weights (still non-zero): real rows must stay bit-identical.
            if n < T:
                x_pert = x.clone()
                w_pert = w.clone()
                x_pert[n:] = (-3.0 * x[n:] + 0.5).to(torch.bfloat16)
                w_pert[n:] = (w[n:].flip(-1) * 2.0 + 0.1).to(torch.bfloat16)
                on_pert = bits(moe_gcount(x_pert, w_pert, ids))
                padded_perturb_real_rows_bitwise = bool(torch.equal(on_pert[:n], ref[:n]))
            else:
                padded_perturb_real_rows_bitwise = True

            # Real rows with mixed zero-gate + positive-gate slots (and all-zero-gate real row when n >= 2),
            # while padded rows still have non-zero gates.
            if n >= 1:
                w_zreal = w.clone()
                w_zreal[0, 0] = 0.0
                if TOPK > 2:
                    w_zreal[0, 2] = 0.0
                if n >= 2:
                    w_zreal[n - 1, :] = 0.0
                ref_zreal = bits(moe_stock(x, w_zreal, ids))
                state.update(n, T)
                on_zreal = bits(moe_gcount(x, w_zreal, ids))
                zero_gate_real_rows_bitwise = bool(torch.equal(on_zreal[:n], ref_zreal[:n]))
            else:
                zero_gate_real_rows_bitwise = True

            r = dict(
                T=T,
                n=n,
                seed=seed,
                padded_gates_nonzero=bool((w[n:].to("cpu").float() > 0).all()) if n < T else True,
                stock_ignores_stale_row_info=bool(torch.equal(ref_with_stale, ref)),
                stock_repeat_bitwise=bool(torch.equal(off, ref)),
                gcount_real_rows_bitwise=bool(torch.equal(on[:n], ref[:n])),
                gcount_repeat_bitwise=bool(torch.equal(on, on2)),
                padded_perturb_real_rows_bitwise=padded_perturb_real_rows_bitwise,
                zero_gate_real_rows_bitwise=zero_gate_real_rows_bitwise,
                finite=bool(torch.isfinite(on[:n].view(torch.bfloat16).float()).all()) if n else True,
                # the comparison is not vacuous: most real-row output elements of stock are non-zero
                ref_nontrivial=bool((ref[:n].view(torch.bfloat16).float() != 0).float().mean() > 0.5) if n else True,
                seconds=round(time.time() - t0, 2),
            )
            r["pass"] = all(v for k, v in r.items() if isinstance(v, bool))
            # informational: padded rows of gcount differ from stock because row_info excluded them
            r["padded_rows_changed"] = (int(((on[n:] != ref[n:]).any(dim=-1)).sum()) if n < T else 0, T - n)
            r["ref_abs_mean"] = float(ref[:n].view(torch.bfloat16).float().abs().mean()) if n else 0.0
            if not r["pass"] and n:
                diff = (on[:n].view(torch.bfloat16).float() - ref[:n].view(torch.bfloat16).float()).abs()
                r["max_abs_diff"] = float(diff.max())
                r["rows_differ"] = int((diff.amax(dim=-1) > 0).sum())
            ok &= r["pass"]
            results.append(r)
            print(json.dumps(r), flush=True)
tp.set_padding_state(None)
summary = dict(
    ok=ok,
    cases=len(results),
    failed=[r for r in results if not r["pass"]],
    interface="count",
    row_filter_api=_API,
    config=dict(
        E=E, I=I, K=K, topk=TOPK, buckets=a.buckets, seeds=a.seeds,
        vllm_torchtpu=fm.__file__,
    ),
)
json.dump(dict(summary=summary, results=results), open(a.out, "w"), indent=1)
print("N2", "PASS" if ok else "FAIL", f"({len(results)} cases)")
sys.exit(0 if ok else 1)
