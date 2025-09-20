# Appendix Checklist

1. Verify Block parity
   - [ ] Confirm both baseline and optimized runs generate `request_metrics.csv` for Block and Block*.
   - [ ] Run `simulation_analysis/compare_block_runs.py` and record runtime + parity results.
2. Refresh QPS=32 sweep
   - [ ] Use `run_qps32_schedulers.sh` (fast-predict on if parity holds).
   - [ ] Summarize metrics via `summarize_request_metrics.py`.
   - [ ] Update `qps32_analysis_template.md` with latency and runtime numbers.
3. Paper alignment
   - [ ] Cross-reference simulator vs paper online metric deltas.
   - [ ] Document findings in Markdown + LaTeX drafts.
4. Large-scale (10×) experiments
   - [ ] Launch via `run_large_scale.sh` (fast-predict on if parity holds).
   - [ ] Summarize metrics and populate `large_scale_analysis_template.md`.
5. Final documentation bundle
   - [ ] Consolidate Markdown + LaTeX appendices under `simulation_analysis/docs/`.
   - [ ] Archive raw outputs under `simulation_analysis/full_runs` and `simulation_analysis/large_scale`.
