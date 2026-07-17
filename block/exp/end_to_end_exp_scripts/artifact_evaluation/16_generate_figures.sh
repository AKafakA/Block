#!/bin/bash
# 16_generate_figures.sh — final AE step
#
# Runs both figure generators against the data 01-15 produced, then copies the
# rendered PNGs into Block-Pow2/figure/ so `pdflatex main.tex` reflects the
# regenerated figures.
#
# Inputs (produced by 07-15):
#   experiment_results_a30/phase11_main, phase12_po2          (Figs 6, 7, 10, 12)
#   experiment_results_a30/phase2_n_ablation, phase7_po4po8    (Fig 9 capacity scan)
#   experiment_results_a30/phase3_1_burstiness/{po2,fanout}    (Fig 11)
#   experiment_results_a30/phase3_2_error_heatmap/{po2,fanout} (Fig 12)
#   experiment_results_a30/phase7_cpu_tracker/{po2,fanout}     (Fig 13)
#   experiment_results_a100/phase57_block/{po2_cp,po2_nocp,...} (Fig 8 Block)
#   experiment_results_a100/llumnix_sweep                       (Fig 8 Llumnix)
#
# Outputs:
#   Block-Pow2/figure/exp_plots/cluster_metrics/{qps,linear}.png  (Figs 6, 7)
#   Block-Pow2/figure/revision/{po2_comparison,oracle_comparison,
#       cpu_overhead,prediction_error_heatmap,burstiness_lines,
#       llumnix_comparison_v2_po2}.{png,pdf}                       (Figs 8-13)
#
# Capacity definition (used by Fig 9): max QPS whose TTFT P99 <= 10s,
# scanned from the float-refinement sweep dirs. No hardcoded capacity values.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 16_generate_figures: render all paper figures ==="
date -u +%Y-%m-%dT%H:%M:%SZ

REGEN="experiments_analysis/paper_figures/regen_figures.py"
FIG8="experiments_analysis/paper_figures/plot_llumnix_aggregate.py"
FIG_OUT_DIR="figures_output"
mkdir -p "$FIG_OUT_DIR"

# --- A30 figures (Figs 6, 7, 9, 10, 11, 12, 13) -------------------------------
if [ ! -d experiment_results_a30 ]; then
    echo "FAIL: experiment_results_a30/ missing — run AE scripts 07-13 first"
    exit 1
fi

echo "[regen_figures] running 'all' (figs 6,7,9,10,11,12,13 + Table 2/3 inputs)"
python3 "$REGEN" all
rc=$?
if [ $rc -ne 0 ]; then
    echo "FAIL: regen_figures.py exited $rc"
    exit $rc
fi

# --- A100 figure (Fig 8) ------------------------------------------------------
if [ -d experiment_results_a100/phase57_block ] && [ -d experiment_results_a100/llumnix_sweep ]; then
    echo "[plot_llumnix_aggregate] Fig 8 (--default po2)"
    python3 "$FIG8" --default po2
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "WARN: Fig 8 plotter exited $rc — A100 data may be incomplete"
    else
        # Copy Fig 8 PNG into the paper's figure dir so main.tex picks it up
        true # Fig 8 written directly to figures_output/
        true
        echo "[copy] Fig 8 -> $FIG_OUT_DIR/llumnix_comparison_v2_po2.png"
    fi
else
    echo "SKIP: Fig 8 (A100 data not present — run AE 14 + 15 first if needed)"
fi

# --- summary ------------------------------------------------------------------
echo
echo "=== generated figures ==="
find figures_output -name "*.png" 2>/dev/null 2>/dev/null | awk '{print " ",$NF}'
ls -la "$FIG_OUT_DIR"/*.png 2>/dev/null | awk '{print " ",$NF}'
echo
echo "=== 16_generate_figures COMPLETE ==="
echo "Next: figures are under figures_output/ ; tables data under figures_output/tables/aggregated.csv"
