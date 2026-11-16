# Why n=16 slowed more on 15be35d: device-trace attribution of the gcount full-bucket regression (t21 profiles)

> Provenance: device traces from the t21 profile phase on vllm-torchtpu nightly `15be35d` (vLLM 0.33.0, libtpu 0.0.50), TP=4, Qwen3.5-35B-A3B BF16, 32 decode steps after warm-up, rank0 `/device:TPU:0`. A = stock (`15be35d`), B = an earlier version of this change applied to `15be35d` (it also contained a full-bucket stock graph, disabled for these runs, so B runs the same gcount program path); it is not the exact head `fe5cbee`. t18 = the same measurement on nightly `f640d734`. Host names are pseudonymized. The raw traces (about 0.4 GB per arm) are not included.


Data: t21 `profile` phase (32 decode steps), rank0 `/device:TPU:0` unless stated, steps `[6:32)` (same slice as t18), A = stock `base_off`, B = gcount single graph `optans_on`, paired by step index.
- **Host A = c3** (`host-13`): `base_off_e1` vs `optans_on_e1`. KV blocks: 9573 / 9573.
- **Host B = c1** (`host-16`): `base_off_a1` vs `optans_on_a1`. KV blocks: **9574 / 9573**.
- For comparison, the t18 traces (`t18/g0` clean, `t18/g1` with overlay-wait markers; all arms 9574 blocks) were downloaded and run through **the same scripts**. They reproduce the published t18 numbers exactly: forward +35.93 / +36.51 µs, filter group +14.92 / +14.85 µs, unchanged ops +20.03 / +20.61 µs.
- One c3 step is excluded from the per-op means: step 14, where B was +471.6 µs, all of it a one-off idle gap before a SparseCore `scatter_offload_async_start`. Paired medians include all steps.

## TL;DR

- **The n=16 regression is all device-side, in the forward module:** +75.7 µs on c3 and +93.5 µs on c1, positive on 26/26 steps and on all 4 TPU cores. The inter-step idle gap is unchanged (Δ +0.02 / −0.09 µs). This matches the steady-timing deltas of the same sessions (+82.2 µs = +1.07% on c3, +92.4 µs = +1.21% on c1).
- **The routing-filter group did not change: +14.97 / +14.89 µs (t18: +14.92).** The hoisted `%lt` `pred[128]` row mask is still read **from HBM** in exactly the 10 full-attention layers (3, 7, …, 39). There, `iota_or_fusion` takes 0.694 µs per call and `fusion s32[128]` 0.687 µs (c3), about 13.8 µs per step together. The 30 GDN layers read a VMEM copy (0.006–0.009 µs per call).
- **All of the growth is in "unchanged" ops: +20.0 µs (t18) → +59.0 µs (c3) / +78.8 µs (c1).** Instance-level split:
  - **Single-instance spikes** (overlay-wait signature): +15.4 → +18.2 / +25.8 µs. In t18 g1, 20/20 spike instances matched overlay-wait deltas.
  - **Diffuse per-call slowdowns:** +2.6 → **+39.1 / +54.7 µs**.
  - Diffuse growers: the MoE-combine `fusion bf16[16,2048]` (3rd per layer, every layer, 0.225 → ~0.52 µs) **+11.6 µs**, new; GDN in-proj `fusion bf16[16,3072]` +0.32 µs per call, **+9.5 µs**, new; t18's `sort s32[128]` speedup gone (**+8.6 to +9.3 µs** relative); `convert_reduce_fusion` diffuse +2 µs.
  - Overlay-like spikes moved around: new on `fusion bf16[16,16]` ×2, all-reduce ×2 more, `convert_reduce` L18, `router_topk` L1; gone from `scatter_custom_fusion`, `fusion bf16[16,2560]`, `reshape.qkv_ref` and others.
- **Likely cause (hypothesis, strongly supported): a different compiled-program layout, not new work.**
  - The stock program has the **same program id and the same per-op behaviour** on f640d734 (t18) and 15be35d (t21 c1). No spike changes, forward −9.5 µs.
  - Comparing t18 gcount with t21 gcount over all executed ops, the per-op HLO text **differs only in the KV-cache dimension (9574 → 9573 blocks)**. Yet t21 gcount runs **+48 µs (c1) / +57 µs (c3)** slower than t18 gcount, all of it in unchanged ops.
  - Every t21 gcount arm reports 9573 blocks; t18 had 9574 everywhere.
  - The stock program is also sensitive to this: in profiles, 9573 vs 9574 is +27 µs (cross-host). In steady timing on the same host it is +26.5 µs on z3 and +4.7 µs on z4 at n=16, and +60 / +44 µs at n=256.
  - c1's larger delta (+93.5 vs +75.7 µs) comes from its baseline: c1's stock run (9574 blocks) is about 27 µs faster than c3's (9573), while the gcount runs on the two hosts differ by only 7.6 µs. Most of that 27 µs is all-reduce (+19 µs), which also varies between runs, so this cannot be split cleanly into a KV-size effect and a host effect.

## 1. Host vs device, n=16 (paired medians over 25 step intervals; `host_dev_t21.txt`)

| Host | Step wall | Forward module | Fwd XLA ops sum | Idle inside fwd | Other XLA modules | Inter-step idle gap (A → B) |
|---|---:|---:|---:|---:|---:|---:|
| c3 | +76.95 µs | **+75.94 µs** | +73.97 µs | +1.62 µs | +0.12 µs | **+0.02 µs** (29.93 → 29.94) |
| c1 | +93.47 µs | **+93.47 µs** | +92.13 µs | +1.09 µs | −0.65 µs | **−0.09 µs** (29.70 → 29.70) |
| t18 g0 (ref.) | +38.04 µs | +35.22 µs | — | — | +1.64 µs | +0.02 µs |

- **All 4 cores, forward paired median (`cores_t21.txt`):** c3 +74.19 to +75.67 µs; c1 +92.57 to +93.51 µs; 26/26 positive on every core. t18 g0: +35.5 to +37.8 µs.
- **Host side:** `_token_padding_update` takes 1.3 → 329 µs per step on c3 and 1.2 → 348 µs on c1, with the Python tracer on. In t18 it was 0.9 → 5.9 µs. B also adds 2× `H2D Dispatch` (about 32 µs per step).
  - At n=16 this is completely hidden behind the ~6 ms device window (inter-step gap unchanged), so it is **not** part of the n=16 regression.
  - It is tracer-inflated. The steady n=1 timing shows no host bottleneck, so the size of the cost without the tracer is unknown.

## 2. Per-op attribution, n=16 (rank0 TPU:0, kept-step means; `attr_c3_n16.txt`, `attr_c1_n16.txt`, `joint_n16.txt`)

### 2a. Groups (µs per step, B − A)

| Component | t18 g0 | t18 g1 (ov) | **t21 c3** | **t21 c1** |
|---|---:|---:|---:|---:|
| Forward, paired median | +35.47 | +38.22 | **+75.67** | **+93.51** |
| Forward, kept-step mean | +35.93 | +36.51 | +75.60 | +94.26 |
| Idle inside forward | +0.98 | +1.05 | +1.62 | +0.60 |
| **Routing-filter group** | +14.92 | +14.85 | **+14.97** (2.35 → 17.32) | **+14.89** (2.24 → 17.12) |
| **Unchanged ops** (same kind and count) | +20.03 | +20.61 | **+59.01** | **+78.78** |
| … spike instances (\|Δ\| ≥ 2 µs per instance) | +15.37 | +14.86 | +18.16 | +25.77 |
| … diffuse remainder | +2.64 | +6.85 | **+39.06** | **+54.68** |
| … diffuse in the 30 GDN / 10 full-attention layers | +1.5 / +1.2 | +4.6 / +2.2 | +34.1 / +4.9 | +44.2 / +10.5 |
| Measured overlay waits (t18 g1 only) | — | 45.5 → 59.0 (+13.5) | n/a | n/a |

Spike and diffuse values are sums of per-instance medians, so they add up to +57.2 µs (c3) and +80.5 µs (c1) rather than exactly the group totals.

**Validation (t18 g1, overlay markers on):** each of the 20 spike instances has a matching overlay-wait delta. Examples: all-reduce L0 +4.16 µs vs overlay +3.95; `scatter_custom_fusion` L1 +8.83 vs +8.56; `fusion bf16[16,2560]` L19 +6.30 vs +6.18. In total the spikes are +14.86 µs against +15.51 µs of overlay-wait delta. The non-spike instances carry only −2.07 µs of overlay waits. **So in t18, spikes ≈ IMEM overlay waits, and the diffuse part is not marked overlay waits.**

### 2b. Routing-filter ops: the hoisted mask is still HBM-resident in the full-attention layers (`filtermap_n16.txt`)

Read from the op HLO text in the trace: an operand layout without `S(1)` means the operand is in HBM.

| Op (B) | Mask operand | Count/step | Layers | Per call (c3 / c1 / t18 g0) |
|---|---|---:|---|---:|
| `iota_or_fusion s32[128]` | `%lt` `pred[128]` in **HBM** | 10 | 3, 7, 11, …, 39 (full attention) | **0.694 / 0.687 / 0.686 µs** |
| `fusion s32[128]` | `%lt` in **HBM** | 10 | same | **0.687 / 0.679 / 0.679 µs** |
| `iota_or_fusion s32[128]` | `%copy-done` (VMEM) | 30 | GDN layers | 0.009 µs |
| `fusion s32[128]` | `%copy-done` (VMEM) | 30 | GDN layers | 0.006 µs |
| `copy-start` + `copy-done` `pred[128]` | from `%lt` | 30 + 30 | GDN layers | 0.007 + 0.002 µs |
| `lt pred[128]` (ENTRY) | — | 1 | — | 0.705 µs |

So in the full-attention layers: 10 × (0.694 + 0.687) ≈ **13.8 µs per step (c3), the same as t18 (13.8 µs)**. This part of the regression did not grow.

### 2c. Top unchanged ops by delta, n=16 (µs per step; per-call times A → B)

| # | Op kind (calls/step) | c3 Δ (per call A → B) | c1 Δ (per call A → B) | t18 g0 Δ | Type (t21) |
|---:|---|---:|---:|---:|---|
| 1 | `distributed_all_reduce bf16[16,2048]` (81) | **+24.60** (10.760 → 11.064) | **+45.38** (10.528 → 11.088) | +21.51 | 5 spike instances +21.8 / +22.4; diffuse +0.7 (c3) vs +24.8 (c1, from the stock run's faster all-reduce) |
| 2 | `convert_reduce_fusion bf16[16,2048]` (40) | **+17.77** (0.492 → 0.936) | **+17.78** (0.491 → 0.936) | +9.13 | diffuse +11.2 (every layer, +0.25–0.30 per call) + spike L18 +6.6 |
| 3 | `fusion bf16[16,16]` (30, GDN) | **+12.64** (0.286 → 0.708) | **+12.59** | +0.08 | 2 spikes (L10, L28: 0.24 → 6.5) |
| 4 | `sort s32[128]` (40) | −11.81 (2.524 → 2.228) | +2.84 (2.160 → 2.231) | −8.53 | spikes only; c3's stock program has a 11.66 µs spike at L1; diffuse ≈ 0 (t18: −8.5) |
| 5 | `fusion bf16[16,2048]` (121) | **+11.64** (0.611 → 0.707) | **+11.51** (0.609 → 0.704) | −0.11 | diffuse, **entirely the 3rd instance per layer** (MoE combine after `convert_reduce_fusion`, 0.225 → +0.23–0.31 per call, all 40 layers) |
| 6 | `router_topk f32[16,8]` (40) | −11.37 (2.352 → 2.067) | −12.71 (2.385 → 2.067) | −19.08 | diffuse −8.5 / −9.7; t18 also had spike offsets of −8.9 |
| 7 | `fusion bf16[16,3072]` (30, GDN in-proj; reads a 12.6 MB weight from HBM) | **+9.56** (6.545 → 6.864) | **+9.60** (6.530 → 6.850) | +6.80 | diffuse, all 30 layers (+0.32 per call); in t18 it was one overlay spike at L28 |
| 8 | `fusion f32[16]` (131) | −6.68 | −6.55 | −6.32 | spike in the stock program at L19 (same in t18) |
| 9 | `gmm_v2` gate/up (40) | +5.29 (37.51 → 37.65) | +4.88 | +4.62 | diffuse, ~0.3% |
| 10 | `gmm_v2` down (40) | +4.40 (34.17 → 34.28) | +4.14 | +3.73 | diffuse, ~0.3% |
| 11 | `fused_conv1d_gdn_batched_b4_c1` (30) | −3.54 (13.25 → 13.13) | −3.55 | −7.87 | ±2.5 µs spikes |
| 12 | `gather_fusion bf16[128,2048]` (80) | +3.25 | +3.26 | +3.41 | spike L36 +3.0 (same as t18) |
| 13 | `select_reduce_fusion.lhs_group_sizes_ref s32[256]` (40) | +2.04 (0.627 → 0.678) | −3.96 (0.778 → 0.679) | −3.98 | stock-program spike at L1 on 9574 only |
| 14 | `fused_conv1d_gdn_per_seq_b1_c16` (30) | −0.68 | −0.90 | −2.83 | diffuse |
| 15 | `copy-done bf16[16,3072]` (58) | +0.40 | +0.39 | +0.48 | — |

- **Gone since t18** (all were overlay spikes): `scatter_custom_fusion` +10.77 → −0.01 / −0.02; `fusion bf16[16,2560]` +6.00 → −0.11 / +0.13; `reshape.qkv_ref` +4.78 → +0.40 / −0.45; `as_strided` +2.28 → +0.26 / −0.03.
- **HLO text of the unchanged ops** (`hlotext_c3_n16.txt`): A and B are identical, including shapes, tiling and memory spaces, for every unchanged op except the two direct mask consumers. `convert_reduce_fusion` now takes `%fusion` instead of `%broadcast_compare_fusion`, and `select_reduce_fusion` takes `%fusion` instead of `%compare_convert_fusion`. The same holds in t18.
- **Op order** (`prof_align.py`): A and B align except for the filter ops. The unaligned-op cost is +9.3 µs vs +9.2 µs in t18, so there is no reordering of unchanged ops.

## 3. What grew from ~+36 µs (t18) to ~+76 / +94 µs (t21)

| Component | t18 g0 | t21 c3 | t21 c1 | Change (c3 / c1) |
|---|---:|---:|---:|---:|
| Routing-filter group (incl. HBM mask, ~13.8 µs) | +14.92 | +14.97 | +14.89 | **±0** |
| Idle inside forward | +0.98 | +1.62 | +0.60 | +0.6 / −0.4 |
| Overlay-like spikes, unchanged ops | +15.37 | +18.16 | +25.77 | **+2.8 / +10.4** (relocated: all-reduce spikes 3 → 5 instances, +9.7; `fusion bf16[16,16]` ×2, +12.6; `convert_reduce` L18, +6.6; `router_topk` L1, +6.1; minus `scatter_custom` / `fusion 2560` / `fusion 3072` / `reshape.qkv` / `as_strided`, about −31) |
| Diffuse, unchanged ops | +2.64 | +39.06 | +54.68 | **+36.4 / +52.0** |
| … MoE combine: `fusion bf16[16,2048]` 3rd per layer | 0.0 | +11.6 | +11.5 | +11.6 |
| … GDN in-proj `fusion bf16[16,3072]` | +0.5 | +9.5 | +9.6 | +9.0 |
| … `sort s32[128]` (t18 speedup lost) | −8.5 | +0.8 | +0.1 | +9.3 / +8.6 |
| … `convert_reduce_fusion` | +9.1 | +11.2 | +11.2 | +2.0 |
| … `fused_conv1d_gdn_per_seq` | −3.1 | −0.6 | −0.9 | +2.4 / +2.2 |
| … `distributed_all_reduce` | +5.7 | +0.7 | +24.8 | −5.0 / +19.1 (varies by host and run) |
| … `gmm_v2` ×2 | +10.1 | +9.5 | +8.5 | ≈0 |

**Cross-program comparisons** (`joint_cross_n16.txt`, `hlotext_B18_vs_B21.txt`, `hlotext_A18_vs_A21c1.txt`):

| Comparison | What differs | Fwd Δ (paired median) | Spike Δ | Diffuse Δ |
|---|---|---:|---:|---:|
| t18 g0 stock → t21 c1 stock | nothing visible: same program id 5030679110655176831, no HLO-text difference in any op; other host and day | **−9.48 µs** | 0.00 (no spike moved) | −8.12 (all-reduce −13.9) |
| t21 c1 gcount → t21 c3 gcount | nothing visible: same program id 84464400566252523; other host | +7.55 µs | 0.00 | +2.99 |
| t21 c1 stock (9574) → t21 c3 stock (9573) | KV blocks; other host | +27.06 µs | +1.21 | +25.88 (all-reduce +19.0, `sort` +14.5) |
| **t18 gcount (9574) → t21 gcount (9573), c1** | **only the KV dimension**, in the 80 GDN-conv / RPA custom calls (30 + 30 + 10 + 10); all other op HLO identical | **+48.06 µs** | +6.27 | **+48.06** |
| t18 gcount → t21 gcount, c3 | same | **+57.06 µs** | +6.58 | +51.75 |

- **Steady-timing natural experiment** (stock, same host, different KV blocks; t21 `res/*.json`):
  - z3 (`9jkz`): 9574 → 9573 is **+26.5 µs at n=16** (7.6138 → 7.6403 ms) and +60.1 µs at n=256.
  - z4 (`7wrn`): +4.7 µs at n=16 and +44.3 µs at n=256.
  - Same-KV A/A on c3 and c4 is 4–19 µs at n=16 and 16–20 µs at n=256.
- **KV blocks across runs:** every t21 gcount arm (opta / optmc / optnb on all 6 hosts) has 9573; t21 stock arms are a mix of 9573 and 9574; all t18 arms had 9574.
- **Conclusion:**
  - The extra ~40 µs (c3) / ~58 µs (c1) over t18 sits entirely in ops that the patch does not touch.
  - It is reproducible across hosts for the same program, and it appears together with the gcount program being compiled with one fewer KV block (9573).
  - The 15be35d nightly itself did not change the stock device program.
  - **Hypothesis:** the full-bucket overhead of the gcount graph depends on program layout (buffer and overlay placement), which depends on the KV-cache size. 9573 happens to be a worse layout for gcount than 9574, and the stock baseline is also layout-sensitive. That would explain why c3 (stock 9573) shows +75.7 µs and c1 (stock 9574) +93.5 µs, though the c1/c3 baseline gap is confounded with host and all-reduce variance.

## 4. n=1 sanity check (`attr_c3_n1.txt`, `attr_c1_n1.txt`, `cores_t21.txt`)

| Host | Forward A → B (median) | Paired Δ | Routed GMMs (gate/up; down) | Fixed extras (same as n=16) | Steady (no profiler) |
|---|---|---:|---|---|---|
| c3 | 5116.4 → 3579.8 µs | **−1528.1 µs (−29.9%)** | −847.7 (33.43 → 12.24 per call); −752.8 (30.50 → 11.68) | filter +14.96; `convert_reduce` +18.1; all-reduce +17.7; `fusion 16,16` +12.6; `fusion 16,2048` +11.8; `fusion 3072` +10.0 | 5.3566 → 3.8274 ms (**−28.5%**) |
| c1 | 5093.8 → 3583.0 µs | **−1509.5 µs (−29.6%)** | −845.2; −751.8 | filter +14.90; `convert_reduce` +18.1; all-reduce +44.4; `fusion 16,16` +12.6; `fusion 3072` +9.9 | 5.3068 → 3.7954 ms (**−28.5%**) |

- **gcount is much faster at n=1, as expected.** The same ~+60–85 µs fixed overhead is present, because it is the same program for every n in the 16-row bucket. It is dwarfed by the −1.6 ms GMM saving.
- **Caveat on n=1 profiles:**
  - With the Python tracer on, B is host-bound at n=1: inter-step idle gap +1160 µs (c3) and +1646 µs (c1); `_token_padding_update` takes 417–484 µs.
  - The other three cores then wait for rank0 inside collectives. Their forward durations are −796 to −1034 µs (c3) and −86 to +110 µs (c1).
  - So only rank0 TPU:0 (the last core to launch) and the steady timing are valid measures at n=1.

## 5. What these traces cannot determine

- **Without overlay-wait markers:**
  - The t21 spike instances cannot be confirmed as IMEM overlay waits. They match the signature validated in t18 (single instances, ~6.3 µs or ~4–5 µs jumps, same positions on both hosts), but this is still an inference.
  - Overlay wait counts per step are unknown for t21 (t18: 64 → 57).
  - It is unknown whether the diffuse slowdowns contain many small overlay waits. In t18 they did not: −2.07 µs in non-spike instances.
- **Without an HLO dump or compiled-program info:**
  - The ENTRY instruction count (t18: 5730 → 5835), buffer assignment and VMEM offsets, MSA prefetch decisions beyond what the op text shows, and LLO scheduling are not visible.
  - So we **cannot say why ops with byte-identical HLO text** (MoE-combine `fusion bf16[16,2048]`, `fusion bf16[16,3072]`, `sort`) **became 0.1–0.3 µs per call slower** in the 9573-block gcount program.
  - Hypotheses, none tested:
    - (H1) VMEM placement or bank conflicts / DMA-queue contention from a different buffer assignment;
    - (H2) background IMEM overlay loads competing for HBM bandwidth with weight-streaming ops (e.g. the 12.6 MB in-proj read);
    - (H3) stalls behind `convert_reduce_fusion`, which now consumes the new mask fusion.
- **Not established:**
  - That the KV dimension is the *only* compile-input difference between t18 gcount and t21 gcount. Only the executed-op HLO text is visible, and the libtpu build is not recorded in the trace. The identical stock program id and spike map across nightlies argue against a compiler change.
  - That 9573 → 9574 would restore ~+36 µs.
  - Confirming either would need an HLO dump of t21 `optans` vs t18 `gcount`, or a same-host run with `num_blocks` pinned to 9574 for both arms.
- **Host side:** the size of the per-step host cost without the tracer (329–484 µs with it) is unknown. It does not affect n=16 (hidden) and does not show in steady n=1 timing.
