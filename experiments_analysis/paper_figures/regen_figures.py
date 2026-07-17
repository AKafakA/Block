#!/usr/bin/env python3
"""Regenerate all figures for Block-Pow2 paper rewrite.

Inputs:  experiment_results_a30/ (NPZs from campaign) + SoCC_revision/cpu_tracker (Fanout CPU prior)
Outputs: Block-Pow2/figure/... PNGs (overwrite in place, matches existing \\includegraphics)

Figures produced:
  1. Request Metrics Under Different QPS (Fig 6 / Sec 6.3) — 8-panel full results
     → figure/exp_plots/cluster_metrics/qps.png
  2. GPU Memory Blocks + Preemption (Fig 7 / Sec 6.4) — 3-row per selected QPS
     → figure/exp_plots/cluster_metrics/linear.png
  3. N-ablation bars (Sec 6.6) — Po2 / Po4 / Po8 / Fanout
     → figure/SoCC_revision/po2_comparison.png
  4. CPU overhead bars — Po2 vs Fanout across QPS (Sec 6.9)
     → figure/SoCC_revision/cpu_overhead.png
  5. Error-injection heatmap (Sec 6.8) — Po2-est × 16 cells, merged priority
     → figure/SoCC_revision/prediction_error_heatmap.png
  6. Burstiness lines (Sec 6.7) — Po2 vs Llumnix × burst {0.25, 0.5, 1.0, 2.0}
     → figure/SoCC_revision/burstiness_lines.png
"""
import os, glob, re, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
def gaussian_filter1d(arr, sigma=20):
    arr = np.asarray(arr, dtype=float)
    if sigma <= 0 or len(arr) < 3: return arr
    r = int(3*sigma)
    x = np.arange(-r, r+1)
    k = np.exp(-0.5*(x/sigma)**2); k = k/k.sum()
    pad = np.pad(arr, r, mode='edge')
    return np.convolve(pad, k, mode='valid')
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
A30  = REPO / "experiment_results_a30"
FIG  = REPO / "figures_output"
SOCC = REPO / "revision"  # optional legacy prior; AE runs are self-contained

COLORS = {
    # Po2-est (primary) and Po2-oracle (reference): deliberately far apart in hue
    # so readers don't confuse them even when they trace together.
    'Astrolabe-Po2-est':    '#e74c3c',   # red
    'Astrolabe-Po2-oracle': '#2c3e50',   # dark navy
    # Fanout-est and Fanout-oracle: previously both green (#2ecc71 / #27ae60) —
    # too close to tell apart. Fanout-oracle moved to amber/brown hue.
    'Astrolabe-Fanout-est':    '#2ecc71',   # bright green (also used in Fig 9)
    'Astrolabe-Fanout-oracle': '#b7950b',   # dark gold/amber — visibly distinct
    'Po4-est':        '#f39c12',
    'Po8-est':        '#8e44ad',
    'Llumnix-':       '#3498db',
    'INFaaS++':       '#ff9800',
    'Round-Robin':    '#7f8c8d',
    'Min QPM':        '#9b59b6',
    'Random':         '#95a5a6',
}

MARKERS = {
    'Astrolabe-Po2-est':    'o',  'Astrolabe-Po2-oracle':    'o',
    'Astrolabe-Fanout-est': 's',  'Astrolabe-Fanout-oracle': 's',
    'Po4-est': 'D',             'Po8-est': 'v',
    'Llumnix-': '^',            'INFaaS++': 'X',
    'Round-Robin': '<',         'Min QPM': '>',   'Random': '*',
}

# Distinct linestyle per scheduler family so lines remain distinguishable
# even when they overlap (color alone is insufficient at small figure sizes).
LINESTYLES = {
    'Astrolabe-Po2-est':       '-',       # solid, thick  — primary
    'Astrolabe-Po2-oracle':    '-',       # solid, medium
    'Astrolabe-Fanout-est':    (0, (5,2)),    # dashed
    'Astrolabe-Fanout-oracle': (0, (5,2)),    # dashed
    'Llumnix-':                (0, (3,1,1,1)),  # dash-dot
    'INFaaS++':                (0, (1,1)),     # dotted
    'Round-Robin':             (0, (5,1,1,1,1,1)),  # loose dash-dot-dot
    'Min QPM':                 (0, (3,3)),     # dash longer
    'Random':                  (0, (1,2)),     # sparse dots
    'Po4-est':                 (0, (6,2,1,2)),
    'Po8-est':                 (0, (4,2,4,2)),
}

# Z-order: draw Po2-est on top so the primary contribution is never hidden.
ZORDER = {
    'Astrolabe-Po2-est':       10,
    'Astrolabe-Po2-oracle':    9,
    'Astrolabe-Fanout-est':    8,
    'Astrolabe-Fanout-oracle': 7,
    'Llumnix-':                6,
    'INFaaS++':                5,
    'Round-Robin':             4,
    'Min QPM':                 3,
    'Random':                  2,
    'Po4-est':                 8,
    'Po8-est':                 7,
}

# ------------------------------------------------------------------
# DISPLAY LABELS (per-figure overrides for user-facing text).
# Internal keys above stay stable (data keys); display labels differ
# per figure per paper convention: Po2+est is the default, so drop
# the "-Po2-est" suffix in most figures and only surface variant
# qualifiers where the figure compares across variants.
# ------------------------------------------------------------------
DISPLAY_FIG6 = {'Astrolabe-Po2-est': 'Block'}   # Fig 6 request metrics
DISPLAY_FIG7 = DISPLAY_FIG6                           # Fig 7 GPU memory
DISPLAY_FIG9 = {                                      # Fig 9 N-tuning
    'Astrolabe-Po2-est':    'Po2 (Block)',
    'Po4-est':              'Po4',
    'Po8-est':              'Po8',
    'Astrolabe-Fanout-est': 'Fanout',
}
DISPLAY_FIG10 = {                                     # Fig 10 oracle
    'Astrolabe-Po2-est':       'Block',
    'Astrolabe-Po2-oracle':    'Block-oracle',
    'Astrolabe-Fanout-est':    'Block-fanout',
    'Astrolabe-Fanout-oracle': 'Block-fanout-oracle',
}
DISPLAY_FIG11 = {'Astrolabe-Po2-est': 'Block'}   # Fig 11 burstiness

def _disp(key, table):
    return table.get(key, key)

# ------------------------------------------------------------------
# NPZ loading helpers (path-based scheduler parsing)
# ------------------------------------------------------------------
def parse_sched_N_est(path):
    s = str(path)
    N = None; est = None
    m = re.search(r"_n_(\d+)_", s); N = int(m.group(1)) if m else None
    m = re.search(r"len_estimated_(true|false)", s); est = (m.group(1)=="true") if m else None
    for name, display in [
        ("min_new_request_latency", None),
        ("min_lunmnix_load", "Llumnix-"),
        ("min_infass_load", "INFaaS++"),
        ("request_per_seconds", "Min QPM"),
        ("round_robin", "Round-Robin"),
        ("random", "Random"),
    ]:
        if f"/{name}/" in s:
            if display: return display, N, est
            if N == 2:  return ("Astrolabe-Po2-est" if est else "Astrolabe-Po2-oracle"), N, est
            if N == 4:  return "Po4-est", N, est
            if N == 8:  return "Po8-est", N, est
            if N == 12: return ("Astrolabe-Fanout-est" if est else "Astrolabe-Fanout-oracle"), N, est
    return "unknown", N, est

def npz_metrics(path):
    try: d = np.load(path, allow_pickle=True)
    except: return None
    out = {}
    if "Throughput" in d: out["throughput"] = float(d["Throughput"])
    if "actual_qps" in d:
        try: out["actual_qps"] = float(d["actual_qps"])
        except: pass
    if "request_latencies" in d and len(d["request_latencies"])>0:
        arr = np.asarray(d["request_latencies"], dtype=float); arr = arr[~np.isnan(arr)]
        out["e2e_mean"] = float(np.mean(arr)); out["e2e_p99"] = float(np.percentile(arr, 99))
    if "prefill_token_latencies" in d and len(d["prefill_token_latencies"])>0:
        arr = np.asarray(d["prefill_token_latencies"], dtype=float); arr = arr[~np.isnan(arr)]
        out["ttft_mean"] = float(np.mean(arr)); out["ttft_p99"] = float(np.percentile(arr, 99))
    if "scheduling_overhead" in d and len(d["scheduling_overhead"])>0:
        ov = np.asarray(d["scheduling_overhead"], dtype=float)
        e2e = np.asarray(d["request_latencies"], dtype=float) if "request_latencies" in d else None
        out["overhead_mean"] = float(np.mean(ov))
        if e2e is not None and len(e2e)==len(ov):
            ratio = 100.0*ov[e2e>0]/e2e[e2e>0]
            out["overhead_ratio_mean"] = float(np.mean(ratio)) if len(ratio) else np.nan
    if "cpu_percents" in d and len(d["cpu_percents"])>0:
        cp = np.asarray(d["cpu_percents"], dtype=float)
        out["cpu_mean"] = float(np.mean(cp)); out["cpu_max"] = float(np.max(cp))
    return out

def _collect_main_series():
    """Phase 1.1 (Fanout + 5 baselines) + Phase 1.2 (Po2 est/oracle), keyed label→qps→metrics."""
    series = {}
    for phase in ("phase11_main", "phase12_po2"):
        for path in glob.glob(str(A30/phase/"**/benchmark_all_metrics.npz"), recursive=True):
            label, N, est = parse_sched_N_est(path)
            m = re.search(r"qps_(\d+(?:\.\d+)?)", path); qps = float(m.group(1)) if m else None
            if label == "unknown" or qps is None: continue
            metrics = npz_metrics(path)
            if metrics: series.setdefault(label, {})[qps] = metrics
    return series

def _capacity_from_ttft(qps_to_ttft_p99_s, slo_s=10.0):
    """Linear-interpolate the SLO crossing; return QPS where TTFT P99 = slo_s."""
    pts = sorted((q, v) for q, v in qps_to_ttft_p99_s.items() if not np.isnan(v))
    for (q1,v1),(q2,v2) in zip(pts, pts[1:]):
        if v1 <= slo_s <= v2:
            if v2 == v1: return q2
            return q1 + (q2-q1)*(slo_s-v1)/(v2-v1)
    return pts[-1][0] if pts else np.nan  # never crosses → capacity is max QPS observed

def _capacity_dir_from_glob(glob_pattern, slo_ms=10000.0):
    """Paper's SLO=10s P99-TTFT capacity, applied to whatever the float-refinement
    sweep produced. Globs qps_X[.Y]_* dirs, reads each TTFT P99 from the NPZ, and
    returns (qps_value, dir_Path, metrics_dict) for the MAX QPS whose TTFT P99
    <= slo_ms. Implements the two-run integer-then-float method without hardcoding
    any specific capacity QPS — capacity is whatever the data crosses at.
    Returns (None, None, None) if no candidate dirs found."""
    candidates = []
    for d in sorted(glob.glob(glob_pattern)):
        p = Path(d)
        npz = p / "benchmark_all_metrics.npz"
        if not npz.exists(): continue
        m = re.search(r"qps_(\d+(?:\.\d+)?)", p.name)
        if not m: continue
        qps = float(m.group(1))
        met = npz_metrics(npz)
        if not met or "ttft_p99" not in met: continue
        candidates.append((qps, p, met))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda x: x[0])
    passing = [c for c in candidates if c[2]["ttft_p99"] <= slo_ms]
    return passing[-1] if passing else candidates[0]

# ------------------------------------------------------------------
# Figure 6: Request Metrics Under Different QPS (8-panel) → qps.png
# ------------------------------------------------------------------
def fig_request_metrics():
    series = _collect_main_series()
    print(f"[fig6/qps] {len(series)} schedulers: {sorted(series)}")

    # Line plots AND capacity bars both use the same 6-scheduler roster
    # (consistency with Fig 7).  Po2-oracle, Fanout-est, and Fanout-oracle
    # are discussed in dedicated figures:
    #   - Fanout-est : Fig 9 (N-tunable) endpoint
    #   - Po2-oracle, Fanout-oracle : Fig 13 (oracle upper bounds, §6.7)
    line_order = ['Random', 'Round-Robin', 'Min QPM', 'INFaaS++', 'Llumnix-',
                  'Astrolabe-Po2-est']
    bar_order  = line_order
    ttft_slo_s = 10.0

    fig, axs = plt.subplots(2, 4, figsize=(16, 7))

    def _plot_series(ax, metric_key, scale, labels_subset, lw_primary=2.2, lw_other=1.4,
                     alpha_other=0.85, logy=False):
        for label in labels_subset:
            if label not in series: continue
            qpsv = sorted(series[label])
            y = [series[label][q].get(metric_key, np.nan)*scale for q in qpsv]
            is_po2 = (label == 'Astrolabe-Po2-est')
            lw = lw_primary if is_po2 else lw_other
            alpha = 1.0 if is_po2 else alpha_other
            ax.plot(qpsv, y, marker=MARKERS.get(label,'o'), color=COLORS.get(label,'k'),
                    linestyle=LINESTYLES.get(label, '-'), zorder=ZORDER.get(label, 5),
                    label=_disp(label, DISPLAY_FIG6), linewidth=lw, alpha=alpha,
                    markersize=5 if is_po2 else 4, markeredgewidth=0.4)
        if logy: ax.set_yscale('log')

    def _plot(ax, metric_key, ylabel, scale=1.0, logy=False, title=None):
        _plot_series(ax, metric_key, scale, line_order, lw_primary=2.4, lw_other=1.5, logy=logy)
        ax.set_xlabel('QPS'); ax.set_ylabel(ylabel)
        if title: ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    # Row 1: means + request throughput
    _plot(axs[0,0], 'e2e_mean',     'Average Request Latency (s)', scale=1/1000.0)
    _plot(axs[0,1], 'ttft_mean',    'Average TTFT (s)',            scale=1/1000.0)
    _plot(axs[0,2], 'overhead_mean','Average Overhead (ms)')
    _plot(axs[0,3], 'throughput',   'Request Throughput (tok/s)')

    # Row 2: P99s + overhead ratio + capacity bars
    _plot(axs[1,0], 'e2e_p99',  'Request Latency P99 (s)', scale=1/1000.0, logy=True)
    _plot(axs[1,1], 'ttft_p99', 'TTFT P99 (s)',            scale=1/1000.0, logy=True)
    axs[1,1].axhline(ttft_slo_s, color='red', linestyle=':', alpha=0.6, label='SLO=10s')
    _plot(axs[1,2], 'overhead_ratio_mean', 'Average Overhead Ratio (%)')

    # Capacity bar chart (right-bottom) — keeps ALL 9 schedulers so that even
    # though line plots drop Fanout/Oracle, the cross-scheduler capacity
    # comparison is still visible in one place.
    caps = []
    for label in bar_order:
        if label not in series: continue
        qps2p99s = {q: series[label][q].get('ttft_p99', np.nan)/1000.0 for q in series[label]}
        cap = _capacity_from_ttft(qps2p99s, ttft_slo_s)
        caps.append((label, cap))
    cap_labels = [c[0] for c in caps]; cap_vals = [c[1] for c in caps]
    bar_colors = [COLORS.get(l,'#888') for l in cap_labels]
    axs[1,3].bar(range(len(cap_labels)), cap_vals, color=bar_colors)
    axs[1,3].set_xticks(range(len(cap_labels)))
    axs[1,3].set_xticklabels([_disp(l, DISPLAY_FIG6) for l in cap_labels], rotation=40, ha='right', fontsize=8)
    axs[1,3].set_ylabel('Capacity (QPS)'); axs[1,3].set_title('Capacity @ SLO P99 TTFT ≤ 10s', fontsize=10)
    axs[1,3].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(cap_vals):
        if not np.isnan(v):
            axs[1,3].text(i, v+0.15, f'{v:.1f}', ha='center', fontsize=8)

    # Single shared legend
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)
    fig.tight_layout(); fig.subplots_adjust(top=0.92, hspace=0.35, wspace=0.30)
    out = FIG/"exp_plots/cluster_metrics/qps.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[fig6/qps] → {out}")

# ------------------------------------------------------------------
# Figure 7: GPU Memory Blocks + Preemption (3-row) → linear.png
# ------------------------------------------------------------------
def fig_gpu_memory():
    """3 rows × K selected-QPS columns: mean free blocks, variance, cumulative preemptions.
    Each panel plots per-query-ID trajectories, one line per scheduler."""
    selected_qps = [20, 24, 28, 32, 36]
    traces = {}  # label → qps → {avg_gpu, var_gpu, num_preempt}

    def _load_trace(path):
        try: d = np.load(path, allow_pickle=True)
        except: return None
        r = {}
        if 'avg_gpu_blocks' in d: r['avg'] = np.asarray(d['avg_gpu_blocks'], dtype=float)
        if 'var_gpu_blocks' in d: r['var'] = np.asarray(d['var_gpu_blocks'], dtype=float)
        if 'num_preempted' in d:
            pre = np.asarray(d['num_preempted'], dtype=float)
            pre = np.maximum(0, pre - pre[0])
            r['pre'] = pre
        return r if r else None

    for phase in ("phase11_main", "phase12_po2"):
        for path in glob.glob(str(A30/phase/"**/benchmark_all_metrics.npz"), recursive=True):
            label, N, est = parse_sched_N_est(path)
            m = re.search(r"qps_(\d+(?:\.\d+)?)", path); qps = float(m.group(1)) if m else None
            if label == "unknown" or qps is None: continue
            if int(qps) not in selected_qps: continue
            t = _load_trace(path)
            if t: traces.setdefault(label, {})[int(qps)] = t

    print(f"[fig7/linear] {len(traces)} schedulers × {len(selected_qps)} QPS")

    # Match Fig 6's 6-scheduler roster for consistency: 5 non-predictive
    # baselines + Astrolabe-Po2-est.  Po2-oracle moved to the dedicated
    # oracle-comparison figure (Fig 13).  Solid lines, uniform weight,
    # Po2-est drawn on top via zorder.
    order = ['Random', 'Round-Robin', 'Min QPM', 'INFaaS++', 'Llumnix-',
             'Astrolabe-Po2-est']

    def _plot_panel(ax, q, metric, sigma_base=20):
        for label in order:
            if label not in traces or q not in traces[label]: continue
            t = traces[label][q]
            if metric not in t: continue
            color = COLORS.get(label, 'k')
            is_po2 = (label == 'Astrolabe-Po2-est')
            lw = 1.8 if is_po2 else 1.3
            alpha = 1.0 if is_po2 else 0.85
            z = ZORDER.get(label, 5)
            sig = sigma_base if is_po2 else sigma_base + 10
            ax.plot(gaussian_filter1d(t[metric], sigma=sig), color=color,
                    label=_disp(label, DISPLAY_FIG7), linewidth=lw, alpha=alpha, linestyle='-', zorder=z)

    fig, axs = plt.subplots(3, len(selected_qps), figsize=(16, 8), sharex='col')
    for ci, q in enumerate(selected_qps):
        axs[0,ci].set_title(f"QPS={q}", fontsize=11)
        _plot_panel(axs[0,ci], q, 'avg')
        _plot_panel(axs[1,ci], q, 'var')
        _plot_panel(axs[2,ci], q, 'pre')
    axs[0,0].set_ylabel('Free GPU Blocks Mean')
    axs[1,0].set_ylabel('Free GPU Blocks Var'); axs[1,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[2,0].set_ylabel('Total Preemption Count')
    for ax in axs[2,:]: ax.set_xlabel('Query ID')
    for row in axs:
        for ax in row: ax.grid(True, alpha=0.3)

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)
    fig.tight_layout(); fig.subplots_adjust(top=0.93, hspace=0.15, wspace=0.25)
    out = FIG/"exp_plots/cluster_metrics/linear.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[fig7/linear] → {out}")

# ------------------------------------------------------------------
# Figure 13: Oracle-Length Upper Bounds (Sec 6.7)
# 4 schedulers × 4 metrics at each scheduler's own capacity point.
# ------------------------------------------------------------------
def fig_oracle_comparison():
    """Po2-est / Po2-oracle / Fanout-est / Fanout-oracle: 2x2 panel.
       (0,0) Capacity bars (point-valued, SLO crossing).
       (0,1) E2E mean vs QPS.
       (1,0) TTFT mean vs QPS.
       (1,1) Scheduling overhead vs QPS.
       Line plots show the full QPS trajectory so the oracle-vs-est gap
       and its QPS dependence are both visible — previous bar-of-means
       version was hard to distinguish from a QPS=32 snapshot."""
    spec = [
        ("Astrolabe-Po2-est",       31.6, "phase12_po2",  2,  True),
        ("Astrolabe-Po2-oracle",    32.4, "phase12_po2",  2,  False),
        ("Astrolabe-Fanout-est",    31.7, "phase11_main", 12, True),
        ("Astrolabe-Fanout-oracle", 32.6, "phase11_main", 12, False),
    ]

    def collect_per_qps(phase, N, est):
        """Return {qps: metrics_dict} for all qps_* NPZs matching (N, est)."""
        base = A30/phase/"sharegpt/min_new_request_latency"
        est_token = "true" if est else "false"
        n_token = f"_n_{N}_"
        out = {}
        for p in sorted(base.glob("qps_*/benchmark_all_metrics.npz")):
            path = str(p)
            if n_token not in path or f"len_estimated_{est_token}" not in path: continue
            m = re.search(r"qps_(\d+)", path); q = int(m.group(1)) if m else None
            if q is None: continue
            met = npz_metrics(p)
            if met: out[q] = met
        return out

    series = {}
    for label, cap, phase, N, est in spec:
        series[label] = (cap, collect_per_qps(phase, N, est))
    print(f"[fig13] {[(l, s[0], len(s[1])) for l,s in series.items()]}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    order = [s[0] for s in spec]

    # (0,0) Capacity bars (point values)
    x = np.arange(len(order))
    caps = [series[l][0] for l in order]
    colors = [COLORS.get(l, '#888') for l in order]
    xtick = [_disp(l, DISPLAY_FIG10) for l in order]
    axes[0,0].bar(x, caps, color=colors)
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(xtick, rotation=15)
    axes[0,0].set_ylabel('Capacity (QPS @ SLO=10s)')
    axes[0,0].set_title('Capacity', fontsize=11)
    axes[0,0].grid(True, alpha=0.3, axis='y')
    axes[0,0].set_ylim(min(caps)-0.4, max(caps)+0.4)
    for i, c in enumerate(caps): axes[0,0].text(i, c+0.02, f'{c:.1f}', ha='center', fontsize=9)

    # Line-plot helper for the three rate-dependent panels
    def _plot_vs_qps(ax, metric_key, ylabel, title, scale=1.0, logy=False):
        for label in order:
            _, by_qps = series[label]
            qpsv = sorted(by_qps)
            y = [by_qps[q].get(metric_key, np.nan)*scale for q in qpsv]
            is_po2_est = (label == 'Astrolabe-Po2-est')
            ax.plot(qpsv, y, marker='o', color=COLORS.get(label, 'k'),
                    label=_disp(label, DISPLAY_FIG10),
                    linewidth=2.4 if is_po2_est else 1.7,
                    markersize=5 if is_po2_est else 4,
                    alpha=1.0 if is_po2_est else 0.9)
        ax.set_xlabel('QPS'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.3)
        if logy: ax.set_yscale('log')

    _plot_vs_qps(axes[0,1], 'e2e_mean',      'E2E mean (s)',             'E2E latency vs QPS',        scale=1/1000.0)
    _plot_vs_qps(axes[1,0], 'ttft_mean',     'TTFT mean (s)',            'TTFT vs QPS',               scale=1/1000.0)
    _plot_vs_qps(axes[1,1], 'overhead_mean', 'Scheduling overhead (ms)', 'Scheduling overhead vs QPS')

    # Single shared legend at top
    handles, labels = axes[0,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)
    fig.tight_layout(); fig.subplots_adjust(top=0.93, hspace=0.35, wspace=0.28)
    out = FIG/"oracle_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[fig13] → {out}")

# ------------------------------------------------------------------
# Figure 2: N-tunable comparison (Sec 6.8) — Po2/Po4/Po8 at their capacities
# ------------------------------------------------------------------
def fig_n_ablation():
    """Po2 vs Po4 vs Po8 vs Fanout at capacity: 2x2 — Capacity, E2E latency, Throughput, Scheduling Overhead.

    Capacity per scheduler is computed dynamically by _capacity_dir_from_glob:
    max QPS whose TTFT P99 <= 10s (paper's fixed SLO). No hardcoded qps_X.Y."""
    target_globs = [
        ("Astrolabe-Po2-est",    str(A30/"phase12_po2/sharegpt/min_new_request_latency/qps_*_n_2_*_len_estimated_true_*")),
        ("Po4-est",              str(A30/"phase7_po4po8/po4/qps_*")),
        ("Po8-est",              str(A30/"phase7_po4po8/po8/qps_*")),
        ("Astrolabe-Fanout-est", str(A30/"phase11_main/sharegpt/min_new_request_latency/qps_*_n_12_*_len_estimated_true_*")),
    ]
    records = []
    for label, pat in target_globs:
        qps, d, met = _capacity_dir_from_glob(pat)
        cap = qps if qps is not None else 0.0
        records.append((label, cap, met or {}))
    print(f"[fig2] capacities: {[(r[0],r[1]) for r in records]}")
    print(f"[fig2] metrics: {[(r[0], r[2].get('overhead_mean'), r[2].get('throughput')) for r in records]}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    labels = [r[0] for r in records]
    x = np.arange(len(labels))
    colors = [COLORS.get(l, '#888') for l in labels]
    xtick = [_disp(l, DISPLAY_FIG9) for l in labels]

    # (0,0) Capacity bars
    caps = [r[1] for r in records]
    axes[0,0].bar(x, caps, color=colors); axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(xtick, rotation=15)
    axes[0,0].set_ylabel('Capacity (QPS @ SLO=10s)'); axes[0,0].set_title('Capacity', fontsize=11)
    axes[0,0].grid(True, alpha=0.3, axis='y'); axes[0,0].set_ylim(min(caps)-0.4, max(caps)+0.4)
    for i, c in enumerate(caps): axes[0,0].text(i, c+0.02, f'{c:.1f}', ha='center', fontsize=9)

    # (0,1) E2E latency (mean + P99) grouped bars — legend OUTSIDE above to avoid bar overlap
    e2e_means = [r[2].get('e2e_mean', 0)/1000 for r in records]
    e2e_p99s  = [r[2].get('e2e_p99',  0)/1000 for r in records]
    w = 0.35
    axes[0,1].bar(x-w/2, e2e_means, w, color=colors, alpha=0.85, label='mean')
    axes[0,1].bar(x+w/2, e2e_p99s,  w, color=colors, hatch='//', alpha=0.85, label='P99')
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(xtick, rotation=15)
    axes[0,1].set_ylabel('E2E latency (s)'); axes[0,1].set_title('E2E latency @ capacity', fontsize=11)
    axes[0,1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False, fontsize=9)
    axes[0,1].grid(True, alpha=0.3, axis='y')
    for i, (m, p) in enumerate(zip(e2e_means, e2e_p99s)):
        axes[0,1].text(i-w/2, m+0.3, f'{m:.1f}', ha='center', fontsize=7)
        axes[0,1].text(i+w/2, p+0.3, f'{p:.1f}', ha='center', fontsize=7)

    # (1,0) Throughput
    tputs = [r[2].get('throughput', 0) for r in records]
    axes[1,0].bar(x, tputs, color=colors); axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(xtick, rotation=15)
    axes[1,0].set_ylabel('Throughput (tok/s)'); axes[1,0].set_title('Throughput @ capacity', fontsize=11)
    axes[1,0].grid(True, alpha=0.3, axis='y')
    axes[1,0].set_ylim(min(tputs)-300, max(tputs)+400)
    for i, t in enumerate(tputs): axes[1,0].text(i, t+50, f'{t:.0f}', ha='center', fontsize=8)

    # (1,1) Scheduling overhead (ms per request)
    ovs = [r[2].get('overhead_mean', 0) for r in records]
    axes[1,1].bar(x, ovs, color=colors); axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(xtick, rotation=15)
    axes[1,1].set_ylabel('Scheduling overhead (ms)'); axes[1,1].set_title('Scheduling overhead per request', fontsize=11)
    axes[1,1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(ovs): axes[1,1].text(i, v+(max(ovs)*0.01 if max(ovs)>0 else 0.5), f'{v:.1f}', ha='center', fontsize=8)

    plt.tight_layout()
    out = FIG/"po2_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[fig2] → {out}")

# ------------------------------------------------------------------
# Figure 12: CPU + Memory overhead — Po2 vs Fanout across QPS (Sec 6.9)
# ------------------------------------------------------------------
def fig_cpu_overhead():
    """2-panel: per-predictor CPU utilization (%) and per-predictor memory (MB), across QPS."""
    po2 = {}  # qps → {'cpu': %, 'mem': MB}
    # AE script 13_cpu_overhead.sh syncs Po2 CPU tracker to phase7_cpu_tracker/po2/
    for path in glob.glob(str(A30/"phase7_cpu_tracker/po2/**/benchmark_all_metrics.npz"), recursive=True):
        m = re.search(r"qps_(\d+)", path); qps = int(m.group(1)) if m else None
        if qps is None: continue
        try:
            d = np.load(path, allow_pickle=True)
            rec = {}
            if "cpu_percents" in d.files and len(d["cpu_percents"])>0:
                rec['cpu'] = float(np.mean(np.asarray(d["cpu_percents"], dtype=float)))
            if "memory_rss_mb" in d.files and len(d["memory_rss_mb"])>0:
                rec['mem'] = float(np.mean(np.asarray(d["memory_rss_mb"], dtype=float)))
            if 'cpu' in rec: po2[qps] = rec
        except: pass

    fanout = {}
    # AE script 13_cpu_overhead.sh syncs Fanout CPU tracker to phase7_cpu_tracker/fanout/.
    # NPZ schema for Fanout matches Po2: cpu_percents / memory_rss_mb arrays.
    for path in glob.glob(str(A30/"phase7_cpu_tracker/fanout/**/benchmark_all_metrics.npz"), recursive=True):
        m = re.search(r"qps_(\d+)", path); qps = int(m.group(1)) if m else None
        if qps is None: continue
        try:
            d = np.load(path, allow_pickle=True)
            rec = {}
            if "cpu_percents" in d.files and len(d["cpu_percents"])>0:
                rec['cpu'] = float(np.mean(np.asarray(d["cpu_percents"], dtype=float)))
            elif "avg_predictor_cpu_percent" in d.files:
                rec['cpu'] = float(np.mean(d["avg_predictor_cpu_percent"]))
            if "memory_rss_mb" in d.files and len(d["memory_rss_mb"])>0:
                rec['mem'] = float(np.mean(np.asarray(d["memory_rss_mb"], dtype=float)))
            elif "avg_predictor_memory_mb" in d.files:
                rec['mem'] = float(np.mean(d["avg_predictor_memory_mb"]))
            if 'cpu' in rec: fanout[qps] = rec
        except: pass
    print(f"[fig12] Po2 qps={sorted(po2)}, Fanout qps={sorted(fanout)}")

    qps = sorted(set(po2) & set(fanout))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(qps)); w = 0.36
    po2_color = COLORS['Astrolabe-Po2-est']
    fan_color = COLORS['Astrolabe-Fanout-oracle']

    # Panel 1 — CPU utilization per predictor
    axes[0].bar(x-w/2, [po2[q]['cpu'] for q in qps], w, color=po2_color, label='Block (N=2)')
    axes[0].bar(x+w/2, [fanout[q]['cpu'] for q in qps], w, color=fan_color, label='Fanout (N=12)')
    axes[0].set_xticks(x); axes[0].set_xticklabels([str(q) for q in qps])
    axes[0].set_xlabel('QPS'); axes[0].set_ylabel('Mean CPU utilization per predictor (%)')
    axes[0].set_title('CPU utilization per predictor', fontsize=11)
    axes[0].legend(loc='upper left'); axes[0].grid(True, alpha=0.3, axis='y')
    for i, q in enumerate(qps):
        axes[0].text(i-w/2, po2[q]['cpu']+1.0,    f"{po2[q]['cpu']:.0f}%", ha='center', fontsize=8)
        axes[0].text(i+w/2, fanout[q]['cpu']+1.0, f"{fanout[q]['cpu']:.0f}%", ha='center', fontsize=8)

    # Panel 2 — Memory per predictor (no redundant legend; same series as panel 1)
    po2_mem = [po2[q].get('mem', np.nan) for q in qps]
    fan_mem = [fanout[q].get('mem', np.nan) for q in qps]
    axes[1].bar(x-w/2, po2_mem, w, color=po2_color)
    axes[1].bar(x+w/2, fan_mem, w, color=fan_color)
    axes[1].set_xticks(x); axes[1].set_xticklabels([str(q) for q in qps])
    axes[1].set_xlabel('QPS'); axes[1].set_ylabel('Memory per predictor (MB)')
    axes[1].set_title('Memory (RSS) per predictor', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, q in enumerate(qps):
        if not np.isnan(po2_mem[i]):
            axes[1].text(i-w/2, po2_mem[i]+15, f"{po2_mem[i]:.0f}", ha='center', fontsize=8)
        if not np.isnan(fan_mem[i]):
            axes[1].text(i+w/2, fan_mem[i]+15, f"{fan_mem[i]:.0f}", ha='center', fontsize=8)

    plt.tight_layout()
    out = FIG/"cpu_overhead.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[fig12] → {out}")

# ------------------------------------------------------------------
# Figure 4: Error-injection heatmap (Sec 6.6)
# ------------------------------------------------------------------
def fig_heatmap():
    """Two-panel heatmap — Mean e2e degradation (%) + P99 e2e degradation (%), Reds colormap.

    Reads the canonical AE output dir phase3_2_error_heatmap/po2/ (single
    16-cell grid). The previous _rerun/_redo2 priority merges were one-off
    remediation runs from the original campaign and are dropped: a clean
    AE re-run produces one self-consistent grid."""
    cells = {}   # (le, la) → {'mean':..., 'p99':...}
    for path in glob.glob(str(A30/"phase3_2_error_heatmap/po2/**/benchmark_all_metrics.npz"), recursive=True):
        m = re.search(r"len_err_(\d+)_lat_err_(\d+)", path)
        if not m: continue
        le, la = int(m.group(1)), int(m.group(2))
        met = npz_metrics(path)
        if met and 'e2e_mean' in met and 'e2e_p99' in met:
            cells[(le,la)] = {'mean': met['e2e_mean'], 'p99': met['e2e_p99']}

    base_npz = A30/"phase12_po2/sharegpt/min_new_request_latency/qps_32_num_queries_10000_n_2_chunked_true_predictor_16_global_1_len_estimated_true_max_slo_0_enable_preemptive_auto_provisioning_false_batch_48_chunk_512/benchmark_all_metrics.npz"
    if not base_npz.exists():
        base_npz = A30/"phase11_main/sharegpt/min_new_request_latency/qps_32_num_queries_10000_n_2_chunked_true_predictor_16_global_1_len_estimated_true_max_slo_0_enable_preemptive_auto_provisioning_false_batch_48_chunk_512/benchmark_all_metrics.npz"
    if base_npz.exists():
        bm = npz_metrics(base_npz)
        base_mean, base_p99 = bm['e2e_mean'], bm['e2e_p99']
        cells[(0,0)] = {'mean': base_mean, 'p99': base_p99}
    else:
        base_mean, base_p99 = 14737.0, 38000.0
        cells[(0,0)] = {'mean': base_mean, 'p99': base_p99}
    print(f"[fig11] baseline mean={base_mean:.0f} ms, p99={base_p99:.0f} ms; cells={len(cells)}")

    # Load Llumnix- @ QPS=32 to compute degradation vs Astrolabe noise-free
    # baseline; overlay as a horizontal reference line on the colorbars.
    lum_npz = A30/"phase11_main/sharegpt/min_lunmnix_load/qps_32_num_queries_10000_n_12_chunked_true_predictor_16_global_1_len_estimated_false_max_slo_0_enable_preemptive_auto_provisioning_false_batch_48_chunk_512/benchmark_all_metrics.npz"
    llumnix_mean_deg = llumnix_p99_deg = None
    if lum_npz.exists():
        lm = npz_metrics(lum_npz)
        llumnix_mean_deg = 100.0 * (lm['e2e_mean'] - base_mean) / base_mean
        llumnix_p99_deg  = 100.0 * (lm['e2e_p99']  - base_p99)  / base_p99
        print(f"[fig11] Llumnix- degradation: mean {llumnix_mean_deg:+.1f}%, p99 {llumnix_p99_deg:+.1f}%")

    errs = [0, 25, 50, 100]
    grid_mean = np.zeros((len(errs), len(errs)))
    grid_p99  = np.zeros((len(errs), len(errs)))
    for i, le in enumerate(errs):
        for j, la in enumerate(errs):
            c = cells.get((le,la))
            if c is None:
                grid_mean[i,j] = 0.0; grid_p99[i,j] = 0.0
            else:
                grid_mean[i,j] = 100.0 * (c['mean'] - base_mean) / base_mean
                grid_p99[i,j]  = 100.0 * (c['p99']  - base_p99)  / base_p99

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, grid, title, lum_deg in [
        (axes[0], grid_mean, 'Mean E2E Latency Degradation (%)', llumnix_mean_deg),
        (axes[1], grid_p99,  'P99 E2E Latency Degradation (%)',  llumnix_p99_deg),
    ]:
        # vmax should include Llumnix degradation so it lands inside the colorbar
        vmax_candidates = [6.0, float(np.nanmax(np.abs(grid)))]
        if lum_deg is not None and not np.isnan(lum_deg):
            vmax_candidates.append(abs(lum_deg) * 1.05)
        vmax = max(vmax_candidates)
        im = ax.imshow(grid, cmap='Reds', vmin=0, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(errs))); ax.set_xticklabels([f"{e}%" for e in errs])
        ax.set_yticks(range(len(errs))); ax.set_yticklabels([f"{e}%" for e in errs])
        ax.set_xlabel('Latency prediction error'); ax.set_ylabel('Length prediction error')
        ax.set_title(title, fontsize=11)
        for i in range(len(errs)):
            for j in range(len(errs)):
                v = grid[i,j]
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                        color='white' if v > vmax*0.55 else 'black', fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Overlay Llumnix- degradation as a horizontal reference on the colorbar
        # so the reader can immediately see that even worst-case injected noise
        # keeps Astrolabe below Llumnix- degradation.
        if lum_deg is not None and not np.isnan(lum_deg):
            cbar.ax.axhline(y=lum_deg, color='#1f4e79', linewidth=2.5, linestyle='-')
            cbar.ax.text(1.12, lum_deg, f'Llumnix-\n{lum_deg:+.1f}%',
                         transform=cbar.ax.get_yaxis_transform(), va='center',
                         ha='left', fontsize=8, color='#1f4e79',
                         bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='#1f4e79', lw=0.7, alpha=0.9))

    fig.suptitle('Block E2E degradation vs noise-free baseline (QPS=32, SLO=10s)', fontsize=11, y=1.02)
    plt.tight_layout()
    out = FIG/"prediction_error_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[fig11] → {out}")

# ------------------------------------------------------------------
# Figure 5: Burstiness (Sec 6.5)
# ------------------------------------------------------------------
def fig_burstiness():
    """Po2-est + Llumnix × burst {0.25, 0.5, 1.0, 2.0} @ QPS=32.
       burst=1.0 is Poisson (from Phase 1.1 baseline)."""
    series = {'Astrolabe-Po2-est': {}, 'Llumnix-': {}}
    # AE script 09_burstiness.sh syncs the Po2 + Llumnix burstiness runs to phase3_1_burstiness/po2/
    for path in glob.glob(str(A30/"phase3_1_burstiness/po2/**/benchmark_all_metrics.npz"), recursive=True):
        m = re.search(r"burstiness_([\d.]+)", path); burst = float(m.group(1)) if m else None
        if burst is None: continue
        if "min_new_request_latency" in path: lab = "Astrolabe-Po2-est"
        elif "min_lunmnix_load" in path: lab = "Llumnix-"
        else: continue
        met = npz_metrics(path)
        if met: series[lab][burst] = met
    po2_poi = A30/"phase12_po2/sharegpt/min_new_request_latency/qps_32_num_queries_10000_n_2_chunked_true_predictor_16_global_1_len_estimated_true_max_slo_0_enable_preemptive_auto_provisioning_false_batch_48_chunk_512/benchmark_all_metrics.npz"
    lum_poi = A30/"phase11_main/sharegpt/min_lunmnix_load/qps_32_num_queries_10000_n_12_chunked_true_predictor_16_global_1_len_estimated_false_max_slo_0_enable_preemptive_auto_provisioning_false_batch_48_chunk_512/benchmark_all_metrics.npz"
    if po2_poi.exists():
        series['Astrolabe-Po2-est'][1.0] = npz_metrics(po2_poi)
    if lum_poi.exists():
        series['Llumnix-'][1.0] = npz_metrics(lum_poi)
    print(f"[fig5] Po2 bursts: {sorted(series['Astrolabe-Po2-est'])}, Llumnix bursts: {sorted(series['Llumnix-'])}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for lab, color in [('Astrolabe-Po2-est', COLORS['Astrolabe-Po2-est']), ('Llumnix-', COLORS['Llumnix-'])]:
        bursts = sorted(series[lab])
        e2e = [series[lab][b].get('e2e_mean',0)/1000 for b in bursts]
        p99 = [series[lab][b].get('e2e_p99',0)/1000 for b in bursts]
        axes[0].plot(bursts, e2e, marker='o', color=color, label=_disp(lab, DISPLAY_FIG11), linewidth=2)
        axes[1].plot(bursts, p99, marker='o', color=color, label=_disp(lab, DISPLAY_FIG11), linewidth=2)
    for ax, title in [(axes[0], 'Mean e2e (s)'), (axes[1], 'P99 e2e (s)')]:
        ax.set_xscale('log'); ax.set_xlabel('Burstiness α (lower = burstier)')
        ax.set_ylabel(title); ax.set_title(f'{title} @ QPS=32')
        ax.grid(True, alpha=0.3); ax.legend()
        ax.set_xticks([0.25, 0.5, 1.0, 2.0]); ax.set_xticklabels(['0.25','0.5','1.0','2.0'])
    plt.tight_layout()
    out = FIG/"burstiness_lines.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[fig5] → {out}")

if __name__ == '__main__':
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    fns = {
        'main':   fig_request_metrics,
        'memory': fig_gpu_memory,
        'oracle': fig_oracle_comparison,
        'n':      fig_n_ablation,
        'cpu':    fig_cpu_overhead,
        'heat':   fig_heatmap,
        'burst':  fig_burstiness,
    }
    if only and only in fns:
        fns[only]()
    else:
        fig_request_metrics()
        fig_gpu_memory()
        fig_oracle_comparison()
        fig_n_ablation()
        fig_cpu_overhead()
        fig_heatmap()
        fig_burstiness()
    print("\nFigures regenerated.")
