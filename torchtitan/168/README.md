# torchtitan #168: review and benchmark evidence (`TPUSelectiveAC` `recompute_bmm` and `save_pallas_attention`)

[Upstream pull request](https://github.com/google-pytorch/torchtitan/pull/168) · reviewed head `fac85b8bd1e3` on base `cfbb77688269`

We tested both flags of `TPUSelectiveAC` on TPU v7x with Qwen3.5-0.8B GRPO training (16 trainer + 16 rollout chips): a 2×2 fixed-batch replay with XProf and AOT FX / XLA HLO inspection, and a 3-seed (1, 42, 101) 30-step GSM8K run.

**Summary.**

- Both flags behave as described and compose.
- With both on (`C4_tt`), the replay `fwd_bwd` drops from 5.130 s to 4.910 s (−4.3%), and peak trainer HBM drops by 4.05 GB (−16.8%).
- In end-to-end training, peak HBM drops by 3.49 GB (−14.8%). The step time change is within run-to-run noise.
- `save_pallas_attention` alone stays within the replay's run-to-run noise floor.
- `recompute_bmm` gives a small, reproducible numerical difference in one training step (see §2.1).
- Over 3 seeds × 30 steps we detected no accuracy difference (pooled step-30 difference −0.73 pp, 95% CI [−1.98, +0.52]). This does not prove equivalence.

---

## 1. Setup

- **Hardware.** 32 TPU v7x chips per run:
  - 16 trainer chips (one `2x2x4` slice, 32 logical devices, FSDP2 `dp=32`, per-block `torch.compile` in `eager_blocks` mode);
  - 16 rollout chips (`4 × 2x2x1`, 16 vLLM engines with TP=2).
  - All replay arms ran on the same trainer slice.
- **Model and workload.**
  - Model: `Qwen/Qwen3.5-0.8B` (24 layers = 18 Gated DeltaNet (GDN) + 6 full-attention layers with Pallas splash attention, `model_kernels=none`, i.e. the XLA GDN path).
  - Data: GSM8K, 128 prompts × 16 rollouts per step, max prompt / response 256 / 4,096 tokens, `lr=2.5e-6`.
- **Software.** Python 3.12.12, torch 2.13.0+cpu, torch_tpu 0.2.0, libtpu 0.0.50, jax 0.12.0.1, vllm 0.32.0.
- **TorchTitan version.** TorchTitan `aaf742e4`, with `torchtitan/experiments/tpu/components/activation_checkpoint.py` replaced by the file from this PR:
  - the file is [`patches/activation_checkpoint_pr168.py`](patches/activation_checkpoint_pr168.py) (SHA-256 `3d4fd13f…`), with the diff in [`patches/torchtitan-pr168-activation-checkpoint.diff`](patches/torchtitan-pr168-activation-checkpoint.diff);
  - each run checked the file and the resulting policy (`recompute_bmm`, `save_pallas_attention`, whether `aten.bmm` / splash are in the save set) at startup.
- **GDN loop version.** The GDN chunk loop in `aaf742e4` takes each chunk from one `unbind` before the loop. On `main` the loop indexes each chunk inside the loop. The saved-tensor and `bmm` counts below are from `aaf742e4`; we did not measure them on `main`. This matches the caveat in the PR description.

## 2. Fixed-batch 2×2 replay

- **Batch.** One real step-1 GRPO batch (2,328,294 valid tokens over 32 ranks, SHA-256 `e3d67e4f…`), frozen.
- **Micro-batch layout.** Locked to `[14336, 14336, 14336, 14336, 8192, 8192]`.
- **Procedure.** Each arm ran 1 compile step, then 5 timed iterations with the profiler off, then 1 iteration with XProf and FX / HLO dumps.

Data: [`summaries/stage1_replay_comparison.json`](summaries/stage1_replay_comparison.json), [`summaries/ab_compile_token_summary.json`](summaries/ab_compile_token_summary.json), [`summaries/hlo_fx_comparison_14336.json`](summaries/hlo_fx_comparison_14336.json), [`summaries/xprof_7cat_comparison.json`](summaries/xprof_7cat_comparison.json), [`summaries/rank0_warm_timings_*.json`](summaries/).

| Arm | `recompute_bmm` | `save_pallas_attention` | GDN saved tensors / layer (row 14336) | Full-attn saved tensors / layer | Splash calls in backward | `fwd_bwd` mean (s) | vs. `A_sel` | Peak HBM (GB) | `pg_loss` / `grad_norm` |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| `A_sel` (baseline) | F | F | 699 (604.74 MiB) | 18 (256.42 MiB) | 2 | 5.1302 | — | 24.10 | 0.090924 / 0.333176 |
| `C3_ft` | F | **T** | 699 (604.74 MiB) | **20 (312.86 MiB)** | **1** | 5.0412 | −0.089 s (−1.7%) | 24.43 (+0.32) | 0.090924 / 0.333170 |
| `C2_tf` | **T** | F | **26 (269.24 MiB)** | 18 (256.42 MiB) | 2 | 4.9796 | −0.151 s (−2.9%) | **19.70 (−4.40)** | 0.090961 / 0.332724 |
| **`C4_tt`** | **T** | **T** | **26 (269.24 MiB)** | **20 (312.86 MiB)** | **1** | **4.9099** | **−0.220 s (−4.3%)** | **20.06 (−4.05)** | 0.090961 / 0.332724 |

All four arms ran 836 XLA modules per step.

![2x2 replay and saved activations](figures/pr168_fig1_factorial_replay.png)

*Left: replay `fwd_bwd` (mean, min–max over 5 iterations) and peak HBM. Right: saved activations per GDN and full-attention layer at row 14,336.*

### 2.1 Numerics against the run-to-run noise floor

Data: [`summaries/noise_floor_comparison.json`](summaries/noise_floor_comparison.json). ΔW is the parameter update of one optimizer step on the frozen batch.

| Comparison | `pg_loss` abs diff | `grad_norm` rel diff | ΔW cosine | ΔW sign agreement | ΔW max abs diff |
|---|---:|---:|---:|---:|---:|
| `A_sel` vs. `A_sel` repeat, same slice | 0 | 7.16e-6 | 0.999938 | 99.97% | 2.89e-7 |
| `C4_tt` vs. `C4_tt` repeats | 0 | 0 | 1.000000 | 100.00% | 0 |
| `A_sel` vs. `C3_ft` (`save_pallas_attention` alone) | 0 | 1.79e-5 | 0.999960 | 99.98% | 3.20e-7 |
| `C2_tf` vs. `C4_tt` (`save_pallas_attention` on top of `recompute_bmm`) | 0 | 0 | 1.000000 | 100.00% | 0 |
| `A_sel` vs. `C4_tt`, same slice, two pairs | 3.72e-5 | 1.35e-3 – 1.36e-3 | 0.988937 – 0.988949 | 99.20% – 99.21% | 4.99e-7 |

- **`save_pallas_attention`.** `A_sel` vs. `C3_ft` is within the `A_sel` repeat floor (0.999960 vs. 0.999938), and adding it to `recompute_bmm` changes nothing bit-for-bit.
- **`recompute_bmm`.** It changes the single-step result by more than the noise floor (ΔW cosine 0.9889). The change is reproducible: `C4_tt` repeats are bit-identical.
  - The likely reason is the one in the PR description: recomputed ops are compiled into the backward program, where XLA fuses them differently (backward HLO fusions at row 14336 go from 2,286 to 2,833).
  - We did not isolate which fusion changes the values.
  - The tight equivalence flag in `stage1_replay_comparison.json` (`passed`) is true for `C3_ft` and false for `C2_tf` / `C4_tt` for this reason.
  - The multi-seed run in §3 checks whether this matters for training.

### 2.2 Where the time and memory go

- **`recompute_bmm`.** It removes `aten.bmm.default` from the save set.
  - In `aaf742e4`'s GDN chunk loop (row 14336) the forward runs 673 small `bmm` per layer, and selective AC saved each output.
  - Recomputing them cuts saved tensors per GDN layer from 699 (604.74 MiB) to 26 (269.24 MiB), about 5.9 GiB less across 18 layers.
  - This matches the PR's `27 + 3 * seq_len / 64` formula at 14,336 tokens.
- **`save_pallas_attention`.** It wraps the save set in the by-name splash-attention matcher (as in `AttentionSavingSelectiveAC`), so the splash forward output and logsumexp are saved (+56.44 MiB per full-attention layer).
  - Backward no longer re-runs the forward kernel: splash calls in the backward HLO go from 2 to 1.
  - XProf full-attention backward time drops from 0.4982 s to 0.4276 s (−0.0706 s in `C3_ft`, −0.0715 s in `C4_tt`), consistent with the −0.089 s replay gain.

## 3. Multi-seed 30-step GSM8K

`A_sel` vs. `C4_tt`, seeds 1 / 42 / 101, full 1,319-prompt greedy validation at steps 0 / 10 / 20 / 30. Data: [`summaries/multiseed_comparison.json`](summaries/multiseed_comparison.json).

| | `A_sel` (3 seeds) | `C4_tt` (3 seeds) |
|---|---:|---:|
| Step time excl. checkpoint (s) | 27.24 ± 0.48 | 26.96 ± 0.75 |
| `update_actor` engine time (s) | 4.98 ± 0.14 | 4.83 ± 0.54 |
| Trainer engine MFU | 5.83% | 5.99% |
| Peak trainer HBM (GB) | 23.53 | 20.04 (−3.49) |
| Step-30 accuracy | 79.45% ± 1.00 pp | 78.72% ± 1.10 pp |

Per-seed step-30 accuracy (`A_sel` → `C4_tt`): seed 1 79.83% → 78.17%, seed 42 80.21% → 79.98%, seed 101 78.32% → 78.01%.

Seed 1 also has an earlier `A_sel` run (`a_sel_s1_manifest`: same arm and seed, launched separately without checkpoints, fewer metrics recorded) at 77.71%. That is 2.1 pp below the main seed-1 `A_sel` run, which shows the size of run-to-run variation at 30 steps. The figure and JSON label the two seed-1 baselines `matched` (main) and `manifest` (earlier).

**Paired McNemar, step 30, `C4_tt − A_sel`:**

| Comparison | n | Difference | 95% CI |
|---|---:|---:|---|
| Seed 1 (main `A_sel` run) | 1,319 | −1.67 pp | [−3.77, +0.43] |
| Seed 1 (earlier `A_sel` run) | 1,319 | +0.45 pp | [−1.66, +2.57] |
| Seed 42 | 1,319 | −0.23 pp | [−2.43, +1.97] |
| Seed 101 | 1,319 | −0.30 pp | [−2.50, +1.89] |
| Pooled, 3 seeds (main seed-1 run) | 3,957 | −0.73 pp | [−1.98, +0.52] |
| Pooled, 3 seeds (earlier seed-1 run) | 3,957 | −0.03 pp | [−1.28, +1.23] |

- Every CI spans 0, so no difference was detected.
- The pooled n = 3,957 is 3 seeds × the same 1,319 prompts. The pooled test does not account for the correlation of a prompt across seeds, so the pooled CI is optimistic.
- This is a short run and does not prove equivalence. The lower bound of the pooled interval is about −2 pp.

![3-seed GSM8K](figures/pr168_fig3_multiseed_gsm8k.png)

## 4. Scope

- **Model and kernel path.** Dense Qwen3.5-0.8B with the XLA GDN path (`model_kernels=none`), where the chunk recurrence uses `aten.bmm`. Not tested:
  - the Tokamax KDA kernel path;
  - MoE models;
  - context parallelism.
- **TorchTitan version.** `aaf742e4` with this PR's file swapped in, not `main` (see §1). Counts on `main` may differ.
- **Compile mode and buckets.** Per-block `torch.compile` (`eager_blocks`). Token buckets used in steady state were 8192 and 14336; 16384 was pre-warmed but not selected.
- **Numerics.** Single-step ΔW checks used one frozen batch.
- **Integration note for callers.** At the time of testing (2026-11-17), `verl-hardware-plugin`'s TorchTitan TPU engine imported `TPUSelectiveAC` from the old module path `torchtitan.experiments.tpu.activation_checkpoint` inside a `try / except ImportError` that falls back to core `SelectiveAC`. A caller wiring these flags should import from `torchtitan.experiments.tpu.components.activation_checkpoint` and fail closed instead of silently falling back.

## 5. Reproduce the checks

From `torchtitan/168/`:

```bash
# Check the summary JSONs against the numbers above (Python 3 standard library only):
python3 tools/verify_pr168_evidence.py

# Regenerate the figures from summaries/ (needs matplotlib and numpy):
python3 tools/plot_168.py
```
