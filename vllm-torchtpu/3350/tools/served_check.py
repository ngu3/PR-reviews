#!/usr/bin/env python3
"""C1 / S1: one `vllm serve` (OpenAI API server) with a given parallel layout; greedy outputs with token ids.

    python3 served_check.py --label base_off_d1 --out /out [--tp 2 --dp 2 --ep] [--env K=V ...] [--phases ...]

The tree and flag come from the environment (PYTHONPATH=/opt/arm-<tree>/src, TPU_MOE_SKIP_PADDED_TOKENS=1 for on),
set by run_served.sh. --env adds variables for the server (e.g. TPU_MOE_COLLECTION_CHUNK_SIZE=16 for the pipelined
collective). Writes <out>/<label>.json and <out>/<label>.server.log.

Phases (prompts are the GSM8K test prompts with the model's chat template, as in mskip_engine.py; greedy, 128 new
tokens, chosen-token logprobs, token ids via return_tokens_as_token_ids):
    single      prompts 0-9, one request at a time: each request is alone in the engine, and with DP > 1 the other
                            DP ranks run with no valid token (the empty-rank case). Scheduling is deterministic, so outputs must
                            match across arms of the same layout.
    concurrent  prompts 0-15 sent together (16 threads); the batch composition depends on arrival timing, so the
                            outputs are recorded and compared only as a diagnostic.
    uneven      prompts 16-18 sent together: with DP = 2 the ranks get unequal numbers of requests.
Each phase also stores the server's /metrics counters before and after (vllm:* request and token totals) and any
HTTP error; the run fails (exit 1) on a server start failure, an HTTP error or a non-finite logprob.
"""
import argparse
import concurrent.futures
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--out", default="/out")
ap.add_argument("--model", default="/data/assets/hf/Qwen3.5-35B-A3B-text")
ap.add_argument("--data", default="/data/data/gsm8k/test.parquet")
ap.add_argument("--tp", type=int, default=4)
ap.add_argument("--dp", type=int, default=1)
ap.add_argument("--ep", action="store_true")
ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the server process")
ap.add_argument("--phases", default="single,concurrent,uneven")
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--startup-timeout", type=int, default=2400)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
base = f"http://127.0.0.1:{a.port}"


def http(path, payload=None, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode() if payload is not None else None,
                                                              headers={"Content-Type": "application/json"}, method="POST" if payload else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if path.startswith("/v1") else body


def metrics():
    try:
        text = http("/metrics", timeout=30)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(vllm:[a-z_]+_total)(\{[^}]*\})? ([0-9.e+]+)$", line)
        if m:
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


import pandas as pd  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.model)
df = pd.read_parquet(a.data)
prompts = [tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True) for m in df["prompt"][:32]]

env = dict(os.environ)
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v
import shutil  # noqa: E402

vllm_bin = shutil.which("vllm")
cmd = ([vllm_bin] if vllm_bin else [sys.executable, "-m", "vllm.entrypoints.cli.main"]) + ["serve", a.model, "--port", str(a.port),
              "--tensor-parallel-size", str(a.tp), "--data-parallel-size", str(a.dp), "--max-model-len", "4352",
              "--max-num-seqs", "256", "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.70",
              "--seed", "0", "--max-logprobs", "1"]
if a.ep:
    cmd.append("--enable-expert-parallel")
meta = dict(label=a.label, node=os.environ.get("NODE_NAME"), pod=os.environ.get("POD_NAME"), tp=a.tp, dp=a.dp,
                        ep=a.ep, env_extra=a.env, moe_skip_env=os.environ.get("TPU_MOE_SKIP_PADDED_TOKENS"),
                        pythonpath=os.environ.get("PYTHONPATH"), phases=a.phases.split(","), max_tokens=a.max_tokens, cmd=cmd)
result = dict(meta=meta, errors=[])


def save():
    json.dump(result, open(f"{a.out}/{a.label}.json", "w"))


log = open(f"{a.out}/{a.label}.server.log", "w")
t0 = time.time()
srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
ready = False
while time.time() - t0 < a.startup_timeout:
    if srv.poll() is not None:
        break
    try:
        http("/health", timeout=10)
        ready = True
        break
    except Exception:  # noqa: BLE001
        time.sleep(10)
meta["server_ready_s"] = round(time.time() - t0, 1)
print("SERVER", "ready" if ready else "FAILED", meta["server_ready_s"], flush=True)


def complete(i):
    payload = dict(model=a.model, prompt=prompts[i], max_tokens=a.max_tokens, temperature=0.0, logprobs=1, seed=0,
                                  return_tokens_as_token_ids=True)
    s = time.time()
    try:
        r = http("/v1/completions", payload)
    except urllib.error.HTTPError as e:
        return dict(prompt=i, error=f"HTTP {e.code}: {e.read().decode()[:300]}")
    except Exception as e:  # noqa: BLE001
        return dict(prompt=i, error=repr(e)[:300])
    c = r["choices"][0]
    lp = c.get("logprobs") or {}
    ids = [int(t.split(":", 1)[1]) for t in lp.get("tokens") or []]
    return dict(prompt=i, token_ids=ids, logprobs=lp.get("token_logprobs"), finish_reason=c.get("finish_reason"),
                            seconds=round(time.time() - s, 3))


def run_phase(name, idx, together):
    m0 = metrics()
    s = time.time()
    if together:
        with concurrent.futures.ThreadPoolExecutor(len(idx)) as ex:
            rows = list(ex.map(complete, idx))
    else:
        rows = [complete(i) for i in idx]
    result[name] = dict(rows=rows, seconds=round(time.time() - s, 2), metrics_before=m0, metrics_after=metrics())
    for r in rows:
        if "error" in r:
            result["errors"].append(f"{name}: prompt {r['prompt']}: {r['error']}")
        elif not r["logprobs"] or any(x is None or not math.isfinite(x) for x in r["logprobs"]):
            result["errors"].append(f"{name}: prompt {r['prompt']}: missing or non-finite logprobs")
    print("PHASE", name, f"{time.time() - s:.1f}s", "tokens", sum(len(r.get("token_ids") or []) for r in rows),
                "errors", sum(1 for r in rows if "error" in r), flush=True)
    save()


try:
    if not ready:
        result["errors"].append("server did not become ready")
    else:
        phases = a.phases.split(",")
        if "single" in phases:
            run_phase("single", list(range(10)), False)
        if "concurrent" in phases:
            run_phase("concurrent", list(range(16)), True)
        if "uneven" in phases:
            run_phase("uneven", [16, 17, 18], True)
finally:
    # The server may already be gone (e.g. its own startup timeout): stopping it must not keep the result from
    # being written, so the identity (meta.cmd, env, server_ready_s) and the errors of a failed start are kept.
    try:
        os.killpg(srv.pid, signal.SIGTERM)
        srv.wait(timeout=60)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(srv.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    meta["server_exit_code"] = srv.poll()
    result["done"] = ready and not result["errors"]
    save()
    print("DONE", a.label, "ok" if result["done"] else f"errors={len(result['errors'])}", flush=True)
os._exit(0 if result["done"] else 1)
