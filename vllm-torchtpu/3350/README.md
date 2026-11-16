# vllm-torchtpu #3350: evidence for BF16 MoE padded-token skipping

[Upstream pull request](https://github.com/vllm-project/vllm-torchtpu/pull/3350) · head `fe5cbee79` (commits `08a78448e`, `fe5cbee79`) on main `6cc233d90`.

Raw results (gzip JSON written by [`tools/mskip_engine.py`](tools/mskip_engine.py) and the direct check), the analysis scripts, and their outputs. Host names are pseudonymized (`host-NN`, stable across experiments); nothing else in the result files was changed.

## Source trees

| Tree (label prefix) | Source | Used for |
|---|---|---|
| `upbase` | vllm-torchtpu `6cc233d90` (main, the PR's parent) | baseline in every comparison |
| `up2` | `fe5cbee79` (the PR head) | unit tests, direct MoE check, flag-on and flag-off numerics, t23 / t24 performance |
| `up` | `ce4a30948`: the PR before the count was renamed to `local_valid_tokens`; the only difference from the head is that rename ([diff](patches/ce4a30948-to-fe5cbee-rename.diff)) | t22 performance, EP fallback, served path, first numerics |
| `upbar` | `fe5cbee79` + [per-layer mask variant](patches/0001-Compute-the-valid-row-mask-per-MoE-layer-optimizatio.patch) (not in the PR) | t23 variant |
| `base`, `gz` | `15be35dad`, and `15be35dad` + [zero-weight approach](patches/gate-zero-on-15be35d/) | gate-zero comparison quoted in the PR |

Arms are labelled `<tree>_<on|off>_<rep>`; `on` means `TPU_MOE_SKIP_PADDED_TOKENS=1`. The three arms of the head:

| Arm | Source | Flag | Where |
|---|---|---|---|
| Baseline | `6cc233d90` | off | every group |
| Candidate off | `fe5cbee79` | off | `results/t22c/c0-head-flag-off` |
| Candidate on | `fe5cbee79` | on | `results/t22b`, `results/t23/d*` (`up2_on_*`), `results/t24` (`up2k*`) |

## Environment

- Image: `vllm/vllm-torchtpu@sha256:f1685130e5ec6e13c37d982345d6c34ce6b7763c74e982a14226208b77bf9bd7` (nightly-20261116-15be35d-v0.33.0; vLLM 0.33.0, torch 2.13.0, torch_tpu 0.2.0, JAX 0.11.2, libtpu 0.0.50) with each tree's `src/` copied to `/opt/arm-<tree>`.
- An arm runs `tools/mskip_engine.py` in a fresh process with its own `HOME` / `TMPDIR` (own compile caches), `PYTHONPATH=/opt/arm-<tree>/src:/workspace/vllm-torchtpu`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, and `TPU_MOE_SKIP_PADDED_TOKENS=1` for flag-on arms. Arms of one group run one after another on one TPU v7x `2x2x1` host, TP=4.
- Model: Qwen3.5-35B-A3B, BF16 text weights. Prompts: GSM8K test set with the chat template.
- KV cache: flag-on arms allocate the count before the KV cache and got 9,573 blocks; flag-off arms got 9,573 or 9,574. The workers' block counts are in each result (`workers.after_start[].kv_num_blocks`). t24 pins the block count.

## PR claims and their sources

| Claim | Results | Output |
|---|---|---|
| Unit tests 836 (head) / 772 (parent), Pyrefly baseline | `t22/a0-correctness`, `t22b/b0-head-correctness` (`unittest_status.txt`) | [correctness](summaries/correctness.txt) |
| Direct MoE check: 45 cases = 15 selected (T, n) × 3 seeds (T=16: n=0,1,5,12,15,16; T=32: 17,24,31,32; T=64: 33,64; T=256: 250,255,256) | `t22/a0-correctness/n2_up`, `t22b/b1-head-n2/n2_up2` | [correctness](summaries/correctness.txt) |
| Numerics vs parent: head on and off, earlier tree on and off | `t22b/b0`, `t22c/c0`, `t22/a0` | [correctness](summaries/correctness.txt) |
| EP fallback, served path | `t22/a4-ep-fallback`, `t22/a5-served` | [correctness](summaries/correctness.txt) |
| Steady decode, 4 hosts | `t22/a1`, `a2`, `a6`, `a7` | [perf_t22_main](summaries/perf_t22_main.txt) |
| Other bucket points, full buckets per session | `t22/a8`, `a9` | [extra points](summaries/perf_t22_extra_points.txt), [per session](summaries/perf_t22_full_buckets_per_session.txt) |
| 256-request rollout | `t22/a3-rollout` | [rollout](summaries/perf_t22_rollout.txt) |
| Head performance; per-layer mask variant (not adopted) | `t23/d1-perf`, `t23/d2-perf`, `t23/d4-correctness` | [t23](summaries/perf_t23_head_and_barrier.txt) |
| Zero-weight approach on the same nightly | `t21-gatezero/z3`, `z4` | [gate-zero](summaries/gatezero_t21.txt) |
| n=16 device-trace attribution | (traces not included) | [trace report](traces/n16_attribution_15be35d.md) |

## Reproduce the outputs

From this folder (Python 3, standard library):

```bash
python3 tools/analyze_mskip.py results/t22c/c0-head-flag-off --check --ref upbase_off_f1
python3 tools/n1_compare.py results/t22c/c0-head-flag-off --ref upbase_off_f1 --arms up2_off_f1
python3 tools/served_compare.py results/t22/a5-served --ref upbase_off_d1 --arms up_on_d1
python3 tools/t22_compare.py results/t22                  # every t22 host: steady, prefill/mixed, rollout
python3 tools/same_host_compare.py results/t22/a9-perf-extra --ref upbase_off --arms up_on --points 32,64,128
python3 tools/same_host_compare.py results/t23/d1-perf --ref upbase_off --arms up2_on,upbar_on
python3 tools/t21_compare.py results/t21-gatezero --experiment gatezero
```

Engine runs (TPU, per arm):

```bash
# numerics + staggered batch-size switching in one engine
python3 tools/mskip_engine.py --label up2_off_f1 --out OUT --model MODEL --data GSM8K_PARQUET --phases numerics,switch \
    --switch-stagger 13 --switch-seq 16,10,16,1,16,15,12,16,3,17,32,16,256,250,255,256 --probe-workers
# steady decode: per step = (t(545 tokens) - t(33 tokens)) / 512, 7 pairs per point
python3 tools/mskip_engine.py --label up2_on_p1 --out OUT --model MODEL --data GSM8K_PARQUET --phases steady \
    --steady-ns 1,2,4,8,15,16,17,255,256 --steady-lens 33,545 --steady-reps 7 --probe-workers
# direct MoE check on one chip
python3 tools/moe_direct_check.py --out n2.json
```
