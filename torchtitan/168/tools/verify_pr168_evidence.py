#!/usr/bin/env python3
"""Verify the committed public evidence for google-pytorch/torchtitan PR #168 (Python 3 stdlib only)."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUM_DIR = ROOT / "summaries"
PATCH_DIR = ROOT / "patches"


class TestPR168PublicEvidence(unittest.TestCase):
    def test_overlay_sha256(self) -> None:
        p = PATCH_DIR / "activation_checkpoint_pr168.py"
        self.assertTrue(p.exists())
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(digest, "3d4fd13f8c554fc97783728a2ec4efbfa763931bdf0da2eaf970fef5a46c9a55")

    def test_stage1_fixed_batch_numerics_and_speedup(self) -> None:
        s1 = json.loads((SUM_DIR / "stage1_replay_comparison.json").read_text(encoding="utf-8"))
        ab = json.loads((SUM_DIR / "ab_compile_token_summary.json").read_text(encoding="utf-8"))["stage1_replay"]
        arms = s1["arms"]
        # C3_ft has 0.0 pg_loss diff and cosine 0.99996001 (>0.9999)
        self.assertEqual(arms["C3_ft"]["pg_loss_abs_diff"], 0.0)
        self.assertAlmostEqual(arms["C3_ft"]["delta_w"]["cosine"], 0.99996001, places=7)
        self.assertTrue(arms["C3_ft"]["passed"])
        # C2_tf and C4_tt match each other bit-for-bit on pg_loss, grad_norm, and delta_w, and have cosine 0.98893663 vs A_sel
        self.assertEqual(arms["C2_tf"]["pg_loss_abs_diff"], arms["C4_tt"]["pg_loss_abs_diff"])
        self.assertEqual(arms["C2_tf"]["grad_norm_rel_diff"], arms["C4_tt"]["grad_norm_rel_diff"])
        self.assertEqual(arms["C2_tf"]["delta_w"], arms["C4_tt"]["delta_w"])
        self.assertAlmostEqual(arms["C4_tt"]["delta_w"]["cosine"], 0.98893663, places=7)
        self.assertAlmostEqual(arms["C4_tt"]["delta_w"]["sign_agreement"], 0.99200523, places=7)
        # C4_tt is faster and lower-HBM than A_sel
        self.assertLess(ab["C4_tt"]["fwd_bwd_s"], ab["A_sel"]["fwd_bwd_s"] - 0.20)
        self.assertLess(ab["C4_tt"]["peak_hbm_gb"], ab["A_sel"]["peak_hbm_gb"] - 3.90)

    def test_xprof_module_counts(self) -> None:
        xp = json.loads((SUM_DIR / "xprof_7cat_comparison.json").read_text(encoding="utf-8"))["summaries"]
        self.assertEqual(xp["A_sel"]["num_xla_modules"], 836)
        self.assertEqual(xp["C4_tt"]["num_xla_modules"], 836)
        self.assertEqual(sorted(xp), ["A_sel", "C2_tf", "C3_ft", "C4_tt"])

    def test_multiseed_30step_mcnemar_cis_span_zero(self) -> None:
        ms = json.loads((SUM_DIR / "multiseed_comparison.json").read_text(encoding="utf-8"))
        pm = ms["paired_mcnemar"]
        for k in (
            "gsc_seed1_C4_minus_A_matched",
            "gsc_seed1_C4_minus_A_manifest",
            "gsc_seed42_C4_minus_A",
            "gsc_seed101_C4_minus_A",
            "pooled_3seed_C4_minus_A_matched",
            "pooled_3seed_C4_minus_A_with_manifest_s1",
        ):
            self.assertTrue(pm[k]["30"]["ci95_spans_zero"], f"{k} step 30 CI did not span 0")
        c4_grp = ms["groups"]["gsc_C4_tt_3seed"]
        a_grp = ms["groups"]["gsc_A_sel_3seed_matched"]
        self.assertAlmostEqual(c4_grp["max_memory_allocated_gb"]["mean"], 20.0354, places=3)
        self.assertAlmostEqual(a_grp["max_memory_allocated_gb"]["mean"], 23.5298, places=3)

    def test_readme_matches_committed_json(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("0.999960", readme)
        self.assertIn("0.999938", readme)
        self.assertIn("0.988937", readme)
        self.assertIn("0.988949", readme)
        self.assertIn("0.090924", readme)
        self.assertIn("256.42 MiB", readme)
        self.assertIn("312.86 MiB", readme)
        self.assertNotIn("0.130315", readme)

    def test_noise_floor_comparison(self) -> None:
        nf = json.loads((SUM_DIR / "noise_floor_comparison.json").read_text(encoding="utf-8"))
        pw = nf["pairwise_comparisons"]
        # Same-slice A_sel vs A_sel
        aa = pw["A_sel_rep1_vs_A_sel_rep2_same_slice"]
        self.assertEqual(aa["pg_loss_abs_diff"], 0.0)
        self.assertAlmostEqual(aa["grad_norm_rel_diff"], 7.16e-6, places=7)
        self.assertAlmostEqual(aa["delta_w"]["cosine"], 0.99993772, places=7)
        # Same-slice C4_tt vs C4_tt (bit-identical)
        c4c4 = pw["C4_tt_rep1_vs_C4_tt_rep2_same_slice"]
        self.assertEqual(c4c4["pg_loss_abs_diff"], 0.0)
        self.assertEqual(c4c4["grad_norm_rel_diff"], 0.0)
        self.assertEqual(c4c4["delta_w"]["cosine"], 1.0)
        self.assertEqual(c4c4["delta_w"]["max_abs_diff"], 0.0)
        # Same-slice A_sel vs C4_tt (deterministic fusion shift)
        ac4_1 = pw["A_sel_rep1_vs_C4_tt_rep1_same_slice"]
        ac4_2 = pw["A_sel_rep2_vs_C4_tt_rep2_same_slice"]
        self.assertAlmostEqual(ac4_1["pg_loss_abs_diff"], 3.724e-5, places=7)
        self.assertAlmostEqual(ac4_1["delta_w"]["cosine"], 0.98894885, places=7)
        self.assertAlmostEqual(ac4_2["delta_w"]["cosine"], 0.98893663, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
