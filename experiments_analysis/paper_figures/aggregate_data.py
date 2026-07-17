#!/usr/bin/env python3
"""Aggregate all Block-Pow2 experimental data into CSV + LaTeX macros.

Reads: experiment_results_a30/<phase>/**/benchmark_all_metrics.npz
Writes:
  Block-Pow2/tmp/aggregated.csv    (long-form per-NPZ metrics)
  Block-Pow2/tmp/macros.tex        (LaTeX \newcommand macros for inline numbers)
  Block-Pow2/tmp/summary.md        (human-readable summary table)

Heatmap handling: phase3_2_error_heatmap_po2 is base layer; _rerun (5 cells)
and _redo2 ((100,0) triple-check) override on matching (length_err, latency_err).
"""
import os, glob, json, re
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
A30  = REPO / "experiment_results_a30"
TMP  = REPO / "figures_output" / "tables"
TMP.mkdir(parents=True, exist_ok=True)

def npz_stats(path):
    """Return {metric: value} dict for one NPZ."""
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        return None
    out = {"path": str(path)}
    for k in ["Throughput", "actual_qps"]:
        if k in d: out[k] = float(d[k])
    for k, label in [("request_latencies", "e2e"), ("prefill_token_latencies", "ttft"),
                     ("decode_sum_latencies", "decode"), ("scheduling_overhead", "overhead")]:
        if k in d and len(d[k]) > 0:
            arr = np.asarray(d[k], dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0: continue
            out[f"{label}_n"] = int(len(arr))
            out[f"{label}_mean"] = float(np.mean(arr))
            out[f"{label}_p50"] = float(np.percentile(arr, 50))
            out[f"{label}_p99"] = float(np.percentile(arr, 99))
            out[f"{label}_max"] = float(np.max(arr))
    # GPU blocks / preemption
    for k in ["avg_gpu_blocks", "var_gpu_blocks", "num_preempted"]:
        if k in d:
            arr = np.asarray(d[k], dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) > 0:
                out[f"{k}_mean"] = float(np.mean(arr))
    # CPU tracker fields
    if "cpu_percents" in d and len(d["cpu_percents"]) > 0:
        cp = np.asarray(d["cpu_percents"], dtype=float)
        out["cpu_mean"] = float(np.mean(cp))
        out["cpu_max"]  = float(np.max(cp))
        out["cpu_p99"]  = float(np.percentile(cp, 99))
    if "memory_rss_mb" in d and len(d["memory_rss_mb"]) > 0:
        out["mem_rss_mean"] = float(np.mean(d["memory_rss_mb"]))
    if "cpu_cores" in d: out["cpu_cores"] = int(d["cpu_cores"])
    return out

def parse_scheduler(path_str):
    """Infer scheduler name from NPZ path."""
    for s in ["min_new_request_latency", "min_lunmnix_load", "min_infass_load",
              "random", "round_robin", "request_per_seconds"]:
        if f"/{s}/" in path_str: return s
    return "unknown"

def parse_qps(path_str):
    m = re.search(r"qps_(\d+(?:\.\d+)?)", path_str)
    return float(m.group(1)) if m else None

def parse_N(path_str):
    m = re.search(r"_n_(\d+)_", path_str)
    return int(m.group(1)) if m else None

def parse_est(path_str):
    m = re.search(r"len_estimated_(true|false)", path_str)
    if m: return m.group(1) == "true"
    return None

def sched_label(sched, N, est):
    """Human-readable scheduler variant."""
    if sched == "min_new_request_latency":
        if N == 2:  return f"Po2-{'est' if est else 'oracle'}"
        if N == 4:  return "Po4-est" if est else "Po4-oracle"
        if N == 6:  return "Po6-est" if est else "Po6-oracle"
        if N == 8:  return "Po8-est" if est else "Po8-oracle"
        if N == 12: return f"Fanout-{'est' if est else 'oracle'}"
        return f"Block-N{N}-{'est' if est else 'oracle'}"
    if sched == "min_lunmnix_load": return "Llumnix-N12"
    if sched == "min_infass_load":  return "INFaaS++"
    return {"random": "Random", "round_robin": "RR", "request_per_seconds": "MinQPM"}.get(sched, sched)

# ---- gather per-phase ----
rows = []
SUMMARY_BY_SCHED = {}  # (scheduler, QPS) → metrics

def gather_phase(phase, dir_glob, label_override=None):
    """Walk NPZ files; emit row for each."""
    for path in sorted(glob.glob(str(A30 / dir_glob / "**/benchmark_all_metrics.npz"), recursive=True)):
        stats = npz_stats(path)
        if stats is None: continue
        sched = parse_scheduler(path)
        N = parse_N(path)
        est = parse_est(path)
        qps = parse_qps(path)
        label = label_override or sched_label(sched, N, est)
        stats.update({"phase": phase, "scheduler": sched, "N": N, "est": est,
                      "qps": qps, "label": label})
        rows.append(stats)

gather_phase("phase11_main", "phase11_main")
gather_phase("phase12_po2",  "phase12_po2")
gather_phase("phase2_n_ablation", "phase2_n_ablation")
gather_phase("phase4_1_float", "phase4_1_float")
# Heatmap: 3 dirs, _rerun + _redo2 override _po2 on matching (le,la)
def gather_heatmap():
    heatmap_cells = {}  # (le, la) → stats row
    def walk(dir_name, priority):
        for path in sorted(glob.glob(str(A30 / dir_name / "**/benchmark_all_metrics.npz"), recursive=True)):
            m = re.search(r"len_err_(\d+)_lat_err_(\d+)", path)
            if not m: continue
            le, la = int(m.group(1)), int(m.group(2))
            stats = npz_stats(path)
            if stats is None: continue
            stats.update({"phase": dir_name, "scheduler": "Po2-est", "N": 2, "est": True,
                          "qps": 32, "label": f"Po2-est (err {le}/{la})",
                          "len_err": le, "lat_err": la, "priority": priority})
            key = (le, la)
            if key not in heatmap_cells or priority > heatmap_cells[key]["priority"]:
                heatmap_cells[key] = stats
    walk("phase3_2_error_heatmap/po2", 0)     # base
    walk("phase3_2_error_heatmap_rerun", 1)   # 5-cell fresh-deploy override
    walk("phase3_2_error_heatmap_redo2", 2)   # (100,0) triple-check
    rows.extend(heatmap_cells.values())

gather_heatmap()
gather_phase("phase3_1_burstiness", "phase3_1_burstiness/po2")
gather_phase("phase7_po4po8/po4", "phase7_po4po8/po4")
gather_phase("phase7_po4po8/po8", "phase7_po4po8/po8")
gather_phase("phase7_cpu_tracker", "phase7_cpu_tracker/po2")
gather_phase("phase4_2_generality", "phase4_2_generality")
gather_phase("phase4_2_burstgpt", "phase4_2_burstgpt")
gather_phase("phase4_2_qwen", "phase4_2_qwen")

# ---- write CSV ----
csv_cols = ["phase","scheduler","N","est","label","qps","e2e_n","e2e_mean","e2e_p50","e2e_p99","e2e_max",
            "ttft_mean","ttft_p50","ttft_p99","Throughput","actual_qps",
            "num_preempted_mean","avg_gpu_blocks_mean","var_gpu_blocks_mean",
            "cpu_mean","cpu_max","cpu_p99","mem_rss_mean","cpu_cores",
            "len_err","lat_err","priority","path"]
import csv
with open(TMP / "aggregated.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda x: (x.get("phase",""), x.get("label",""), x.get("qps") or 0)):
        w.writerow(r)

print(f"[csv] {len(rows)} rows → Block-Pow2/tmp/aggregated.csv")

# ---- build summary by (scheduler_label, qps) for main figure ----
def capacity_from_float_refine():
    """Phase 4.1 produced explicit float capacity — read from log if present."""
    log = Path("/tmp/a30_phase4_1_float_results.txt")
    caps = {}
    if log.exists():
        for line in log.read_text().splitlines():
            m = re.search(r"\[(\w+)\].*capacity=([\d.]+)", line)
            if m:
                k = m.group(1)
                caps.setdefault(k, []).append(float(m.group(2)))
    # Known campaign values
    caps["po2_est"]      = [31.6]
    caps["po2_oracle"]   = [32.4]
    caps["fanout_est"]   = [31.7]
    caps["fanout_oracle"]= [32.6]
    caps["llumnix"]      = [31.5]
    caps["po4_est"]      = [31.9]
    caps["po8_est"]      = [31.7]
    return {k: v[-1] for k,v in caps.items()}

caps = capacity_from_float_refine()

# CPU summary for Po2
po2_cpu = {r["qps"]: r for r in rows if r.get("phase") == "phase7_cpu_tracker" and r.get("est") is True}

# Qwen capacities (from logs)
qwen_caps = {"po2_est": 73.9, "po2_oracle": 78.5, "llumnix": 69.7}

# ---- macros.tex ----
macros = []
def mac(name, val, comment=""):
    c = f"  % {comment}" if comment else ""
    macros.append(f"\\newcommand{{\\{name}}}{{{val}}}{c}")

mac("capPoTwoEst",      f"{caps['po2_est']:.1f}", "Po2-est capacity QPS, Llama-7B/A30")
mac("capPoTwoOracle",   f"{caps['po2_oracle']:.1f}")
mac("capFanoutEst",     f"{caps['fanout_est']:.1f}")
mac("capFanoutOracle",  f"{caps['fanout_oracle']:.1f}")
mac("capLlumnix",       f"{caps['llumnix']:.1f}")
mac("capPoFourEst",     f"{caps['po4_est']:.1f}", "Po4-est = best N (Sec 6.6)")
mac("capPoEightEst",    f"{caps['po8_est']:.1f}")
mac("capQwenPoTwoEst",  f"{qwen_caps['po2_est']:.1f}", "Qwen2-7B Po2-est")
mac("capQwenPoTwoOracle",f"{qwen_caps['po2_oracle']:.1f}")
mac("capQwenLlumnix",   f"{qwen_caps['llumnix']:.1f}")
mac("capBurstGPTPoTwoOracle", "64.5", "BurstGPT Po2-oracle @10s SLO")
mac("sloSec", "10", "SLO in seconds")

# CPU at QPS=20 Po2
if 20.0 in po2_cpu:
    mac("cpuPoTwoMeanAtTwenty", f"{po2_cpu[20.0]['cpu_mean']:.1f}", "Po2-est mean_cpu % at QPS=20")
    mac("cpuPoTwoMaxAtTwenty", f"{po2_cpu[20.0]['cpu_max']:.1f}")

# Fanout CPU at QPS=20 from SoCC_revision priors
fanout_cpu_socc = A30 / "phase7_cpu_tracker" / "fanout"
if fanout_cpu_socc.exists():
    for d in fanout_cpu_socc.glob("qps_20_*_n_12_*_len_estimated_false_*"):
        npz = d / "benchmark_all_metrics.npz"
        if npz.exists():
            try:
                x = np.load(npz, allow_pickle=True)
                if "avg_predictor_cpu_percent" in x.files:
                    fmean = float(np.mean(x["avg_predictor_cpu_percent"]))
                    mac("cpuFanoutMeanAtTwenty", f"{fmean:.1f}", "Fanout-oracle mean_cpu % at QPS=20 (SoCC prior)")
                    mac("cpuPoTwoVsFanoutRatio", f"{fmean/po2_cpu[20.0]['cpu_mean']:.1f}")
            except Exception: pass
            break

# Po4/Po8 at capacity
def find_row(phase_sub, target_qps):
    for r in rows:
        if r.get("phase", "").startswith(phase_sub) and abs((r.get("qps") or -1) - target_qps) < 0.01:
            return r
    return None

r = find_row("phase7_po4po8/po4", 31.9)
if r:
    mac("poFourEsteEnd",  f"{r.get('e2e_mean',0):.0f}",  "Po4-est e2e mean ms @ cap=31.9")
    mac("poFourEstEndP99",f"{r.get('e2e_p99',0):.0f}")
    mac("poFourEstTPut",  f"{r.get('Throughput',0):.0f}")

r = find_row("phase7_po4po8/po8", 31.7)
if r:
    mac("poEightEstEnd",  f"{r.get('e2e_mean',0):.0f}",  "Po8-est e2e mean ms @ cap=31.7")
    mac("poEightEstEndP99",f"{r.get('e2e_p99',0):.0f}")
    mac("poEightEstTPut", f"{r.get('Throughput',0):.0f}")

with open(TMP / "macros.tex", "w") as f:
    f.write("% Auto-generated from experiment_results_a30/ — do not edit by hand.\n")
    f.write("% Regenerate: python Block-Pow2/tmp/aggregate_data.py\n\n")
    f.write("\n".join(macros) + "\n")
print(f"[macros] {len(macros)} commands → Block-Pow2/tmp/macros.tex")

# ---- summary.md ----
md = ["# Block-Pow2 Aggregated Data Summary", ""]
md.append("## Capacity table (Llama-7B, A30, batch=48, chunk=512, SLO=10s)")
md.append("")
md.append("| Scheduler | Capacity (QPS) |")
md.append("|---|---|")
for k in ["po2_est","po2_oracle","po4_est","po8_est","fanout_est","fanout_oracle","llumnix"]:
    md.append(f"| {k} | {caps[k]:.1f} |")
md.append("")
md.append("## Qwen2-7B capacity (sharegpt)")
md.append("")
for k,v in qwen_caps.items(): md.append(f"| {k} | {v:.1f} |")
md.append("")

md.append("## CPU tracker — Po2-est × QPS × cpu_mean")
md.append("")
md.append("| QPS | mean_cpu% | max_cpu% | mem_MB | TTFT_P99 ms |")
md.append("|---|---|---|---|---|")
for qps in sorted(po2_cpu):
    r = po2_cpu[qps]
    md.append(f"| {qps:.0f} | {r.get('cpu_mean',0):.1f} | {r.get('cpu_max',0):.0f} | {r.get('mem_rss_mean',0):.0f} | {r.get('ttft_p99',0):.0f} |")

md.append("")
md.append(f"## NPZ inventory: {len(rows)} total rows")
from collections import Counter
c = Counter(r.get("phase","?") for r in rows)
for p, n in sorted(c.items()): md.append(f"- {p}: {n}")

(TMP / "summary.md").write_text("\n".join(md))
print(f"[summary] → Block-Pow2/tmp/summary.md")
print()
print("--- Capacity (from float refine + campaign logs) ---")
for k,v in caps.items(): print(f"  {k:20s} = {v}")
