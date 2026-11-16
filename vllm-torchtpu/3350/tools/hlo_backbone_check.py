#!/usr/bin/env python3
"""Compare the decode backbone programs (modules *fused_add_rms_norm*) of XLA dumps (MSKIP_XLA_DUMP=1).

    python3 hlo_backbone_check.py LABEL=DUMP_DIR LABEL=DUMP_DIR [LABEL=DUMP_DIR ...]

Each after_optimizations HLO file is normalized with tools/hlo_dump_compare.py (debug info and source-tree paths
masked) and hashed. For every pair of dumps given after the first one, the script prints whether the backbone programs
are identical (as sorted lists of hashes) and how many modules the two dumps share as a multiset.
"""
import collections
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlo_dump_compare as h  # noqa: E402


def scan(d):
    hashes, backbone = collections.Counter(), []
    for p in sorted(glob.glob(os.path.join(d, "*.after_optimizations.txt"))):
        x = hashlib.sha256(h.normalize(open(p, errors="replace").read()).encode()).hexdigest()[:16]
        hashes[x] += 1
        if "fused_add_rms_norm" in os.path.basename(p):
            backbone.append(x)
    return hashes, sorted(backbone)


def main():
    dumps = [a.split("=", 1) for a in sys.argv[1:]]
    res = {}
    for label, d in dumps:
        res[label] = scan(d)
        print(f"{label}: {sum(res[label][0].values())} modules, {len(res[label][1])} backbone (fused_add_rms_norm)")
    labels = [label for label, _ in dumps]
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ha, hb = res[a][0], res[b][0]
            print(f"{a} vs {b}: backbone identical={res[a][1] == res[b][1]}; modules common={sum((ha & hb).values())}, "
                  f"only A={sum((ha - hb).values())}, only B={sum((hb - ha).values())}")


if __name__ == "__main__":
    main()
