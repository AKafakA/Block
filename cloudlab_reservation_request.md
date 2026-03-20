# CloudLab Reservation Request

**Dates**: April 13–16, 2026 (3 days)

**Resources Requested**:
- 12× d7525 nodes (NVIDIA A30 GPUs) — Wisconsin cluster
- 2× d8545 nodes (NVIDIA A100-40GB GPUs) — Wisconsin cluster

**Project**: Block — Predictive Scheduling for Distributed LLM Serving

## Purpose

We are preparing the camera-ready revision for our paper accepted at ACM SoCC 2026. The paper presents Block, a predictive load balancing scheduler for distributed large language model (LLM) serving systems. The revision addresses reviewer feedback that requires additional experiments comparing scheduling strategies under varied workload conditions and system configurations.

Specifically, we need to run the following experiments on the A30 cluster (12 nodes, 12 GPU instances):

1. **Generality study**: Evaluate Block's scheduling across different model configurations (batch sizes, chunked prefill sizes), alternative models (Qwen2-7B), and different request trace distributions (BurstGPT dataset). Each configuration requires redeploying the serving system and running capacity search experiments (~1 hour per configuration).

2. **Burstiness and error sensitivity analysis**: Measure scheduling robustness under bursty arrival patterns (gamma-distributed request intervals) and with injected prediction errors, comparing Block against baseline schedulers.

3. **A100 comparison experiments**: Run Block on the A100 cluster (2 nodes, 8 GPU instances) to compare against Llumnix, a state-of-the-art migration-based LLM serving scheduler, under both chunked prefill and non-chunked prefill configurations.

These experiments are time-sensitive as the SoCC 2026 rebuttal period coincides with the reservation dates. The A30 and A100 clusters are essential because our evaluation requires reproducible results on the same hardware used in the original submission.

**Estimated GPU-hours**: ~240 GPU-hours (A30) + ~48 GPU-hours (A100)

Thank you for considering this request.
