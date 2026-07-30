"""
Regenerate Astrolabe vs Llumnix 4-panel comparison (Fig 8) from per-QPS
benchmark summary text files (`_logs_logs.txt`) — the format the A100
benchmark harness emits alongside the NPZ.

Panels: throughput, mean_e2e_latency, mean_token_latency, mean_scheduling_overhead.
Each panel plots 3 curves: {Fanout or Po2} × {CP=on, CP=off} + Llumnix.

Input layout (matches AE scripts 14_a100_llumnix.sh + 15_a100_block.sh):
  experiment_results_a100/phase57_block/{po2_cp,po2_nocp,fanout_cp,fanout_nocp}/
      block_qps{N}/block_qps{N}_logs_logs.txt
  experiment_results_a100/llumnix_sweep/
      llumnix_qps{N}/llumnix_qps{N}_logs_logs.txt

Usage:
    python plot_llumnix_aggregate.py --default fanout
    python plot_llumnix_aggregate.py --default po2
"""
import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt

BLOCK_ROOT = "experiment_results_a100/phase57_block"
LLUMNIX_DIR = "experiment_results_a100/llumnix_sweep"

# Plotter parses these four fields from the per-QPS aggregate text file
# (format emitted by benchmark_serving.py):
#   backend X dur_s ... tokens_per_s X qps X
#   mean_token_latency=X(ms), mean_e2e_latency=X(ms), ..., mean_global_scheduling_overhead=X(ms)
_FIELD_PATTERNS = {
    "throughput":       r"tokens_per_s\s+(\d+\.\d+)",
    "mean_e2e_ms":      r"mean_e2e_latency=([\d.]+)",
    "mean_token_ms":    r"mean_token_latency=([\d.]+)",
    "mean_overhead_ms": r"mean_global_scheduling_overhead=([\d.]+)",
}


def _parse_aggregate_file(path):
    """Extract the four plotter fields from a single benchmark summary .txt.
    Returns dict or None if any required field missing."""
    if not os.path.exists(path):
        return None
    try:
        text = open(path).read()
    except OSError:
        return None
    out = {}
    for key, pat in _FIELD_PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            return None
        out[key] = float(m.group(1))
    return out


def load_block_series(config):
    """Read per-QPS aggregates for one Block config (po2_cp / po2_nocp / fanout_cp / fanout_nocp).
    Globs phase57_block/{config}/block_qps*/block_qps*_logs_logs.txt; key by integer QPS."""
    out = {}
    pat = os.path.join(BLOCK_ROOT, config, "block_qps*", "block_qps*_logs_logs.txt")
    for p in sorted(glob.glob(pat)):
        m = re.search(r"block_qps(\d+)", os.path.basename(p))
        if not m:
            continue
        q = int(m.group(1))
        rec = _parse_aggregate_file(p)
        if rec is not None:
            out[q] = rec
    return out


def load_llumnix_series():
    """Read per-QPS aggregates for the Llumnix sweep.
    Globs llumnix_sweep/llumnix_qps*/llumnix_qps*_logs_logs.txt; key by integer QPS."""
    out = {}
    pat = os.path.join(LLUMNIX_DIR, "llumnix_qps*", "llumnix_qps*_logs_logs.txt")
    for p in sorted(glob.glob(pat)):
        m = re.search(r"llumnix_qps(\d+)", os.path.basename(p))
        if not m:
            continue
        q = int(m.group(1))
        rec = _parse_aggregate_file(p)
        if rec is not None:
            out[q] = rec
    return out


def to_xy(d, field):
    qs = sorted(d.keys())
    return qs, [d[q][field] for q in qs]


def plot_panel(ax, series_list, field, ylabel, title, ylog=False):
    for label, data, color, marker, ls in series_list:
        qs, ys = to_xy(data, field)
        ax.plot(qs, ys, marker + ls, color=color, linewidth=2, markersize=8,
                markeredgecolor="black", label=label)
    ax.set_xlabel("QPS", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    if ylog:
        ax.set_yscale("log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--default", choices=["fanout", "po2"], required=True,
                    help="Which Block scheduler to plot as the 'default'")
    ap.add_argument("--out", default=None, help="Output PNG path stem (also writes PDF)")
    args = ap.parse_args()

    if args.default == "fanout":
        cp_cfg, nocp_cfg = "fanout_cp", "fanout_nocp"
        cp_label, nocp_label = "Astrolabe-Fanout (CP)", "Astrolabe-Fanout (no CP)"
        out = args.out or "figures_output/llumnix_comparison_v1_fanout"
    else:
        cp_cfg, nocp_cfg = "po2_cp", "po2_nocp"
        cp_label, nocp_label = "Astrolabe (CP)", "Astrolabe (no CP)"
        out = args.out or "figures_output/llumnix_comparison_v2_po2"

    block_cp = load_block_series(cp_cfg)
    block_nocp = load_block_series(nocp_cfg)
    llumnix = load_llumnix_series()

    if not (block_cp and block_nocp and llumnix):
        print(f"Missing data: {cp_cfg}={len(block_cp)} {nocp_cfg}={len(block_nocp)} llumnix={len(llumnix)}")
        print(f"  expected: {BLOCK_ROOT}/{{{cp_cfg},{nocp_cfg}}}/block_qps*/block_qps*_logs_logs.txt")
        print(f"  expected: {LLUMNIX_DIR}/llumnix_qps*/llumnix_qps*_logs_logs.txt")
        sys.exit(1)

    series = [
        (cp_label,    block_cp,   "#d62728", "o", "-"),
        (nocp_label,  block_nocp, "#2ca02c", "s", "-"),
        ("Llumnix",   llumnix,    "#3498db", "^", "--"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    plot_panel(ax00, series, "throughput",
               "Throughput (tok/s)", "Throughput")
    plot_panel(ax01, series, "mean_e2e_ms",
               "Mean request latency (ms)", "Mean request latency", ylog=True)
    plot_panel(ax10, series, "mean_token_ms",
               "Mean token latency (ms)", "Mean token latency", ylog=True)
    plot_panel(ax11, series, "mean_overhead_ms",
               "Mean scheduling overhead (ms)", "Mean scheduling overhead")

    handles, labels = ax00.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, 1.00), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"Saved: {out}.png and {out}.pdf")

    print("\n=== Numeric highlights ===")
    for q in [16, 19, 20, 28, 36]:
        if q in block_cp and q in block_nocp and q in llumnix:
            print(f"QPS={q}: "
                  f"{cp_label}={block_cp[q]['throughput']:.0f}/{block_cp[q]['mean_e2e_ms']:.0f}ms, "
                  f"{nocp_label}={block_nocp[q]['throughput']:.0f}/{block_nocp[q]['mean_e2e_ms']:.0f}ms, "
                  f"Llumnix={llumnix[q]['throughput']:.0f}/{llumnix[q]['mean_e2e_ms']:.0f}ms")


if __name__ == "__main__":
    main()
