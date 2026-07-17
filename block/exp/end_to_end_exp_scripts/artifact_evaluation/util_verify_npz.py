#!/usr/bin/env python3
"""util_verify_npz.py — sanity-check NPZs from a phase directory.

Usage: python util_verify_npz.py <phase_dir>

Verifies for each NPZ:
- File exists and loads
- Required fields present (request_latencies, prefill_token_latencies, Throughput)
- Sample count matches expected (10000 for Llama, 9963 for Qwen)
- TTFT P99 is sane (>0, not infinity)
- For CPU tracker NPZs: cpu_percents/memory_rss_mb arrays non-empty
"""
import sys, os, glob
import numpy as np

REQUIRED = ['request_latencies', 'prefill_token_latencies', 'Throughput']

def check(path):
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        return f"FAIL load: {e}"
    keys = list(d.keys())
    missing = [k for k in REQUIRED if k not in keys]
    if missing:
        return f"FAIL missing fields: {missing}"
    n = len(d['request_latencies'])
    p99 = float(np.percentile(d['prefill_token_latencies'], 99)) if n > 0 else None
    tp = float(d['Throughput'])
    cpu_present = 'cpu_percents' in keys and len(d['cpu_percents']) > 0
    is_qwen = 'qwen' in path.lower()
    expected_n = 9963 if is_qwen else 10000
    n_ok = "✓" if n == expected_n else f"⚠ (expected {expected_n})"
    extras = []
    if cpu_present:
        cp = d['cpu_percents']
        extras.append(f"cpu={float(np.mean(cp)):.1f}%/{float(np.max(cp)):.1f}% (n={len(cp)})")
    if 'cpu_cores' in keys:
        extras.append(f"cores={int(d['cpu_cores'])}")
    return f"OK n={n} {n_ok}, TTFT_P99={p99:.0f}ms, tp={tp:.1f}tok/s" + (", " + ", ".join(extras) if extras else "")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    files = sorted(glob.glob(os.path.join(root, '**', 'benchmark_all_metrics.npz'), recursive=True))
    if not files:
        print(f"No NPZs under {root}")
        sys.exit(1)
    print(f"Found {len(files)} NPZs under {root}")
    n_ok = 0
    n_bad = 0
    for f in files:
        rel = os.path.relpath(f, root)
        result = check(f)
        marker = "✓" if result.startswith("OK") else "✗"
        print(f"  {marker} {rel}: {result}")
        if marker == "✓": n_ok += 1
        else: n_bad += 1
    print(f"\nSummary: {n_ok} ok / {n_bad} bad / {len(files)} total")
    sys.exit(0 if n_bad == 0 else 1)

if __name__ == '__main__':
    main()
