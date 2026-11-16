#!/usr/bin/env python3
"""Do two engine arms compile the same programs? Compare their XLA dumps (MSKIP_XLA_DUMP=1).

    python3 hlo_dump_compare.py A_DUMP_DIR B_DUMP_DIR [--stage after_optimizations]

Every <stage> HLO text file of a dump is normalized and hashed: metadata={...} attributes, the HloModule line's module
name, the debug tables (FileNames, FunctionNames, FileLocations, StackFrames) and comments are removed, and the source
tree path (/opt/arm-<tree>/) is masked, also inside base64-encoded custom-call payloads (Mosaic kernels), which are
replaced by a hash of their decoded, masked bytes. The script reports how many modules each dump has, how many normalized
programs are common to both (as a multiset), and lists the unmatched ones with their sizes. Exit 0 when the two
multisets are equal, 1 otherwise.
"""
import argparse
import base64
import binascii
import collections
import glob
import hashlib
import os
import re
import sys

META = re.compile(r",?\s*metadata=\{[^{}]*\}")
MODULE = re.compile(r"^HloModule\s+\S+", re.M)
# Source trees are copied to /opt/arm-<tree>; the path ends up in debug strings, also inside the base64-encoded Mosaic
# kernel payloads of custom calls.
TREE = re.compile(r"/opt/arm-[A-Za-z0-9]+/")
TREE_B = re.compile(rb"/opt/arm-[A-Za-z0-9]+/")
B64 = re.compile(r"[A-Za-z0-9+/]{64,}={0,2}")
DEBUG_TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")


def _payload(m):
    blob = m.group(0)
    try:
        raw = base64.b64decode(blob, validate=True)
    except (binascii.Error, ValueError):
        return blob
    return "b64:" + hashlib.sha256(TREE_B.sub(b"/opt/arm-T/", raw)).hexdigest()[:16]


def normalize(text):
    """Drop debug info (metadata, the source tables, tree paths) and keep the program."""
    text = META.sub("", text)
    text = MODULE.sub("HloModule M", text)
    out, skip = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in DEBUG_TABLES:
            skip = True
            continue
        if skip and (not stripped or stripped[0].isdigit()):
            continue
        skip = False
        if stripped.startswith("//"):
            continue
        out.append(B64.sub(_payload, TREE.sub("/opt/arm-T/", line.rstrip())))
    return "\n".join(out)


def load(d, stage):
    progs = collections.Counter()
    names = collections.defaultdict(list)
    sizes = {}
    files = sorted(glob.glob(os.path.join(d, "**", f"*.{stage}.txt"), recursive=True))
    for p in files:
        t = normalize(open(p, errors="replace").read())
        h = hashlib.sha256(t.encode()).hexdigest()[:16]
        progs[h] += 1
        names[h].append(os.path.basename(p))
        sizes[h] = len(t)
    return progs, names, sizes, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--stage", default="after_optimizations")
    a = ap.parse_args()
    pa, na, sa, fa = load(a.a, a.stage)
    pb, nb, sb, fb = load(a.b, a.stage)
    common = sum((pa & pb).values())
    print(f"A {a.a}: {fa} {a.stage} files, {len(pa)} distinct normalized programs")
    print(f"B {a.b}: {fb} {a.stage} files, {len(pb)} distinct normalized programs")
    print(f"common (multiset): {common}; only in A: {sum((pa - pb).values())}; only in B: {sum((pb - pa).values())}")
    for tag, only, names, sizes in (("A", pa - pb, na, sa), ("B", pb - pa, nb, sb)):
        for h, c in sorted(only.items(), key=lambda kv: -sizes[kv[0]])[:30]:
            print(f"  only in {tag}: {h} x{c} {sizes[h]} chars  e.g. {names[h][0]}")
    return 0 if pa == pb else 1


if __name__ == "__main__":
    sys.exit(main())
