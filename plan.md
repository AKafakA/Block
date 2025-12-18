# Block + CARA: Development Guide

## Project Overview

This document provides comprehensive context for developing the **Block** LLM serving system and its extension **CARA** (Co-optimizing Super Heterogeneous LLM Serving through Model Routing and Predictive Batch Scheduling).

---

## 1. Block: Balance Loader of LLM Serving

### What is Block?

**Block** is a research prototype exploring *predictive, performance-aware scheduling* for distributed large-language-model (LLM) inference. It builds on top of Microsoft's [Vidur](https://github.com/microsoft/vidur) simulator and adds:

- **Side-car Predictor Service**: Forecasts per-instance leading metrics with Vidur at runtime
- **Global Scheduler**: Uses predictions (or live metrics) to route requests across instances
- **Length Estimator**: Light-weight model to predict response tokens for unseen prompts

### Block Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Global Scheduler                      │
│  (Routes requests based on predictive metrics)           │
└────────────┬────────────────────────────┬────────────────┘
             │                            │
    ┌────────▼────────┐          ┌───────▼────────┐
    │  Instance 1     │          │  Instance 2    │
    │  ┌──────────┐   │          │  ┌──────────┐  │
    │  │ vLLM/etc │   │          │  │ vLLM/etc │  │
    │  └──────────┘   │          │  └──────────┘  │
    │  ┌──────────┐   │          │  ┌──────────┐  │
    │  │Predictor │   │          │  │Predictor │  │
    │  └──────────┘   │          │  └──────────┘  │
    └─────────────────┘          └────────────────┘
```

### Key Components

1. **Predictor** (`block/predictor/`)
   - Co-locates with every inference node
   - Collects live stats or spins up Vidur simulation on-demand
   - Answers "What if I got one more request?"

2. **Global Scheduler** (`block/global_scheduler/`)
   - Receives requests
   - Queries Predictors
   - Applies scheduling policy (Block, LLumnix, round-robin, etc.)

3. **Query Length Tagger** (`block/length_estimation/`)
   - RoBERTa-based regressor
   - Predicts response-token count for unseen (model, prompt) pairs
   - Feeds scheduler with input-aware cost estimates

### Repository Layout

```
block/
 ├── predictor/             # Side-car prediction service
 ├── global_scheduler/      # Request router and scheduler
 │   ├── cara/              # CARA extension (new!)
 │   └── ...
 ├── length_estimation/     # Token-length regressor
 ├── benchmark/             # Load-generator
 │   └── cara/              # CARA benchmarking (new!)
 ├── config/                # Cluster configurations
 ├── exp/                   # Experiment scripts
 │   └── cara/              # CARA deployment scripts (new!)
 └── data/                  # Datasets (ShareGPT, BurstGPT, ArXiv-Summ)

vidur/                      # Modified Vidur simulator
 ├── scheduler/
 │   ├── sarathi_replica_scheduler.py
 │   ├── simulate_predict_replica_scheduler.py
 │   └── ...
 └── ...
```

---

## 2. CARA: Co-optimizing Heterogeneous LLM Serving

### Research Motivation

Modern LLM serving faces challenges:
- Must host **multiple model variants** (e.g., 7B, 13B, 70B)
- Support **heterogeneous frameworks** (vLLM, Ollama, etc.)
- Deploy across **diverse GPU clusters** with different performance/cost profiles
- Current systems address parts in isolation (routing OR scheduling, not both)

**CARA's Key Insight**: Missing control layer is **global batch orchestration** - the ability to form and schedule batches across the entire cluster, jointly considering request heterogeneity, model quality, and infrastructure constraints.

### CARA System Design

#### 2.1 Unified API with Request-Specific Objectives (RSOs)

CARA introduces a new API that allows users to specify per-request objectives:

```json
[
  {
    "request_id": "string",
    "prompts": "string",
    "request_specific_objective": {
      "latency_ms": 1000,
      "model_quality_min": 0.8,
      "token_budget": 100,
      "relax_order": ["latency", "quality", "budget"]
    },
    "sampling_params": {}
  }
]
```

**RSO Fields**:
- `latency_ms`: End-to-end latency constraint
- `model_quality_min`: Minimum normalized model quality score
- `token_budget`: Maximum tokens allowed (cost constraint)
- `relax_order`: Order to relax constraints if unsatisfiable

#### 2.2 Scheduling Model (MILP Formulation)

CARA formulates scheduling as a Mixed-Integer Linear Program (MILP):

**Decision Variable**:
- `x_{r,i} ∈ {0,1}`: Binary variable (1 if request r assigned to instance i)

**Objective Function**:
```
minimize:
  Σ x_{r,i} · (w_lat·P_{r,i} + w_cost·C_{r,i} - w_qual·Q_{r,i})
  + w_m · max_r(Σ x_{r,i}·P_{r,i})  # Minimize makespan
  + w_u · Σ_i(memory_utilization)    # Balance load

subject to:
  Σ_i x_{r,i} = 1                     # Each request assigned exactly once
  Σ_i x_{r,i}·P_{r,i} ≤ RSO_lat(r)   # Latency constraint
  Σ_i x_{r,i}·Q_{r,i} ≥ RSO_qual(r)  # Quality constraint
  Σ_i x_{r,i}·C_{r,i} ≤ RSO_budget(r) # Budget constraint
  Σ_r x_{r,i}·M_{r,i} ≤ Cap_i         # Instance capacity
```

**Prediction Parameters**:
- `P_{r,i}`: Predicted latency for request r on instance i
- `C_{r,i}`: Predicted cost
- `Q_{r,i}`: Predicted quality score
- `M_{r,i}`: Required memory (KV cache)

#### 2.3 CARA Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Global CARA Scheduler                        │
│  ┌──────────────┐        ┌─────────────────┐                 │
│  │   Batch      │        │ Model Estimator │                 │
│  │ Orchestrator │◄──────►│  (Quality &     │                 │
│  │  (Greedy     │        │   Length Pred)  │                 │
│  │  Mini-Batch) │        └─────────────────┘                 │
│  └──────┬───────┘                                             │
└─────────┼─────────────────────────────────────────────────────┘
          │
    ┌─────┴──────────────────────────────────┐
    │                                        │
┌───▼──────────────┐            ┌───────────▼──────────┐
│  vLLM Instance   │            │  Ollama Instance     │
│  ┌────────────┐  │            │  ┌────────────┐      │
│  │  vLLM      │  │            │  │  Ollama    │      │
│  │  Backend   │  │            │  │  Backend   │      │
│  └────────────┘  │            │  └────────────┘      │
│  ┌────────────┐  │            │  ┌────────────┐      │
│  │Performance │  │            │  │Performance │      │
│  │ Predictor  │  │            │  │ Predictor  │      │
│  │ (sidecar)  │  │            │  │ (sidecar)  │      │
│  └────────────┘  │            │  └────────────┘      │
└──────────────────┘            └─────────────────────┘
```

**Key Components**:

1. **Global Scheduler** (`block/global_scheduler/cara/cara_serve.py`)
   - Central orchestrator
   - Dynamic adaptive batching
   - Greedy mini-batch scheduling algorithm
   - Constraint checking with confidence bounds

2. **Model Estimator** (Centralized)
   - Predicts output length in tokens
   - Predicts normalized quality score
   - One prediction per (request, model) pair

3. **Performance Predictor** (Per-instance sidecar)
   - Uses sequence-to-value regression (LSTM/Transformer/Mamba)
   - Learns from online data collection
   - Forward-compatible (adapts to system changes)
   - Input: Request sequence (request_id, prompt_tokens, generated_tokens, predicted_total_tokens)
   - Output: Predicted latency P_{r,i}

4. **Instance Services**
   - **vLLM Instance** (`block/global_scheduler/cara/cara_instance/vllm_instance.py`)
   - **Ollama Instance** (`block/global_scheduler/cara/cara_instance/ollama_instance.py`)
   - Expose `/status` API for predictor
   - Execute requests via `/v1/completions` endpoint
   - Report server-side E2E latency

#### 2.4 Greedy Mini-Batch Scheduling Algorithm

```python
# Pseudocode from CARA paper Algorithm 1
def Schedule(B, C, I, Score):
    """
    B: Full batch of requests
    C: Mini-batch size
    I: Set of instances
    Score: Objective function
    """
    P = NewEmptyPlacement()  # instance -> Requests Set

    for M in Partition(B, C):  # Process in mini-batches
        best_score = -∞
        best_assignment = None

        # Try all permutations for this mini-batch
        for A in Permutation(I, M):
            score = EvaluatePlacement(A, P, Score)
            if score > best_score:
                best_score = score
                best_assignment = A

        # Commit best assignment
        P = ExtendPlacement(P, best_assignment)
        Dispatch(best_assignment)

    return P
```

**Complexity**:
- Full batch: O(|I|^|B|) - exponential (infeasible)
- Greedy mini-batch: O(|I|^|C| × |B|/|C|) - tractable
- Analogous to greedy sampling in LLM decoding

**Extensions**:
- Beam search variant (Appendix in paper)
- ML-based MILP approximators (future work)

#### 2.5 Dynamic Adaptive Batching

Key insight: Batching timeout cost depends on system load.

**Strategy**:
- **Low load** (workers free): Small batch size, short timeout → minimize latency
- **High load** (workers busy): Large batch size, longer timeout → maximize throughput
- Rationale: When workers are busy, requests would queue anyway, so batching delay is masked

---

## 3. Current CARA Implementation Status

### 3.1 Implemented Components

#### Global Scheduler
- **File**: `block/global_scheduler/cara/cara_serve.py`
- **Features**:
  - FastAPI server on port 8200
  - `/v1/completions` endpoint (vLLM-compatible)
  - Random and round-robin scheduling strategies
  - Supports heterogeneous backends (vLLM + Ollama)
  - Converts HuggingFace model names to Ollama tags

#### Instance Implementations

**Base Class**: `block/global_scheduler/cara/cara_instance/Instance.py`
```python
class Instance(ABC):
    """Abstract base for model instances"""

    @abstractmethod
    async def query_backend(self, payload: dict, headers: dict = None):
        """Execute inference request"""
        pass

    async def query_predictor(self, request_id, num_context_tokens, ...):
        """Query performance predictor"""
        pass

    async def query_instance(self, payload, predicted_num_decode_tokens):
        """Main entry point: predictor + backend"""
        pass
```

**vLLM Instance**: `block/global_scheduler/cara/cara_instance/vllm_instance.py`
- OpenAI-compatible `/v1/completions` API
- Streaming response handler for SSE
- JSON validation for chunk completeness
- **Returns**: `{generated_text, ttft, itl, output_tokens, success, error, model, server_latency}`

**Ollama Instance**: `block/global_scheduler/cara/cara_instance/ollama_instance.py`
- Native Ollama `/api/generate` endpoint
- Newline-delimited JSON streaming
- Temperature=0, repeat_penalty=1.0
- **Returns**: Same schema as vLLM instance

**Key Feature**: Both instances report `server_latency` (server-side E2E) to enable scheduling overhead calculation.

#### Benchmark Implementation

**Files**:
- `block/benchmark/cara/benchmark_serving.py` - Main benchmark (copied from vLLM)
- `block/benchmark/cara/cara_end_point_func.py` - CARA-specific request function

**Key Modifications**:

1. **RequestFuncOutput Dataclass** (extended from vLLM):
```python
@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0              # Client-side E2E
    output_tokens: int = 0
    ttft: float = 0.0                 # Server-side
    itl: list[float] = field(default_factory=list)  # Server-side
    tpot: float = 0.0
    prompt_len: int = 0
    error: str = ""
    start_time: float = 0.0
    model: str = ""
    request_id: str = ""
    scheduling_overhead: float = 0.0  # NEW: client E2E - server E2E
```

2. **Scheduling Overhead Calculation**:
```python
# Client-side E2E measurement
output.latency = time.perf_counter() - st

# Get server-side E2E from backend response
server_latency = response_map.get("server_latency", 0.0)

# Calculate overhead = network + CARA routing time
output.scheduling_overhead = output.latency - server_latency
```

3. **Benchmark Metrics Reporting**:
   - Standard metrics: TTFT, ITL, TPOT, E2EL
   - **CARA-specific**: Scheduling overhead (mean, median, std, percentiles)
   - Only reported when `--backend cara`

4. **Integration**:
   - Merged `CARA_ASYNC_REQUEST_FUNCS` with vLLM's `ASYNC_REQUEST_FUNCS`
   - Default backend: `cara`
   - Default port: `8200`

#### Deployment Scripts

**File**: `block/exp/cara/deploy_cara.py`

**Features**:
- Parses host file to group nodes by type (e.g., d8545, d7525)
- Allocates nodes based on model config `node_type` specification
- Supports selective deployment (deploy only specified models)
- Cleans up existing processes before deployment
- **vLLM deployment**:
  - Multi-GPU support (auto-detects GPU count)
  - Custom HF cache support
  - Configurable parameters (gpu-memory-utilization, max-model-len, enforce-eager)
- **Ollama deployment**:
  - Listens on 0.0.0.0:11434
  - Parallel request processing (configurable via --ollama-num-parallel)
  - Synchronous model pull + warmup inference
- Generates deployment config JSON for CARA scheduler

**File**: `block/exp/cara/run_cara_e2e.sh`

**E2E Workflow**:
1. Deploy backend instances (optional: skip with REDEPLOY=false)
2. Wait for backends to initialize (60s)
3. Start CARA scheduler server on target host
4. Run benchmark tests with random dataset
5. Save results with timestamp

**Configuration**:
- 50 random prompts (128 input tokens, 64 output tokens)
- Request rate: infinite (send all at once)
- Results saved to `experiment_output/cara_test_results/`

### 3.2 Configuration Files

**Model Config Template**: `block/config/cara/model_config_template.json`
```json
{
  "Qwen-2.5-3B": {
    "hf_model_name": "Qwen/Qwen2.5-3B",
    "backend": "vllm",
    "precision": "fp16",
    "node_type": {
      "d8545": {
        "count": 2,
        "gpu-memory-utilization": 0.95,
        "use_hf_cache": true
      }
    }
  },
  "Qwen-2.5-7B": {
    "hf_model_name": "Qwen/Qwen2.5-7B",
    "backend": "ollama",
    "node_type": {
      "d7525": 1
    }
  }
}
```

**Host Config**: `block/config/host_configs.json`
```json
{
  "node0.example.com": {
    "ip_address": "192.168.1.1",
    "predictor_ports": [8001, 8002]
  }
}
```

**Deployment Config** (generated): `block/config/cara/model_deployment.json`
```json
{
  "Qwen-2.5-3B": {
    "hf_model_name": "Qwen/Qwen2.5-3B",
    "backend": "vllm",
    "precision": "fp16",
    "node_hosts": ["user@node0.example.com", "user@node1.example.com"]
  }
}
```

### 3.3 What's NOT Yet Implemented

The following are described in the CARA paper but not yet implemented:

1. **MILP Solver Integration**
   - Current: Random/round-robin scheduling
   - Needed: Greedy mini-batch algorithm (Algorithm 1 from paper)
   - Needed: Objective function evaluation with weights

2. **Performance Predictor (Sidecar)**
   - Current: Placeholder in Instance base class
   - Needed: Sequence regression model (LSTM/Transformer/Mamba)
   - Needed: `/status` API on inference instances
   - Needed: Online data collection and model retraining

3. **Model Estimator**
   - Current: None
   - Needed: Output length prediction
   - Needed: Quality score prediction
   - Needed: Centralized microservice

4. **RSO Support**
   - Current: None
   - Needed: RSO parsing in request payload
   - Needed: Constraint checking with confidence bounds
   - Needed: Auto-relax policy

5. **Dynamic Adaptive Batching**
   - Current: None
   - Needed: Load-aware batch sizing
   - Needed: Adaptive timeout based on instance availability

6. **Beam Search Scheduler**
   - Current: None
   - Needed: Extension of greedy algorithm (Appendix in paper)

---

## 4. Development Guidelines

### 4.1 Adding New Scheduling Policies

To add a new scheduling policy to CARA:

1. **Update `cara_serve.py`**:
```python
@app.post("/v1/completions")
async def completion(request: Request) -> Response:
    # ...
    if scheduling == "your_new_policy":
        instance = your_selection_logic(instances, request_json)
    # ...
```

2. **Add CLI argument**:
```python
parser.add_argument("--scheduling", type=str, default="random",
                    help="Scheduling strategy: random, round_robin, your_new_policy")
```

### 4.2 Adding New Backend Types

To support a new inference framework (e.g., TGI, SGLang):

1. **Create instance class**:
```python
# block/global_scheduler/cara/cara_instance/your_backend_instance.py
from block.global_scheduler.cara.cara_instance.Instance import Instance

class YourBackendInstance(Instance):
    async def query_backend(self, payload: dict, headers: dict = None):
        # Adapt payload to your backend's API
        # Stream response and extract metrics
        # Return: {generated_text, ttft, itl, output_tokens,
        #          success, error, model, server_latency}
        pass
```

2. **Register in `cara_serve.py`**:
```python
elif backend_type == "your_backend":
    from block.global_scheduler.cara.cara_instance.your_backend_instance import YourBackendInstance
    instance = YourBackendInstance(...)
    instances.append(instance)
```

3. **Update `deploy_cara.py`**:
```python
def get_your_backend_commands(model_path: str, ...):
    # Return list of SSH commands to deploy
    pass

# In main():
elif backend == "your_backend":
    cmds = get_your_backend_commands(...)
    run_ssh_cmd(host, cmds, f"Deploying YourBackend ({model_key})")
```

### 4.3 Metrics Consistency Rules

**Critical**: Always maintain these metric definitions:

- **Client-side E2E Latency** (`output.latency`):
  - Measured at benchmark client
  - `time.perf_counter()` from request send to response complete
  - User-perceived latency
  - **PRIMARY metric for user SLOs**

- **Server-side E2E Latency** (`server_latency`):
  - Measured at backend instance
  - `time.perf_counter()` from request receive to response complete
  - Backend processing time only
  - Returned in instance `query_backend()` response

- **Scheduling Overhead** (`scheduling_overhead`):
  - Calculated at benchmark client
  - `client_latency - server_latency`
  - Captures network + CARA routing overhead
  - **CARA-specific metric**

- **TTFT** (Time To First Token):
  - Server-side measurement
  - From request receive to first token generated
  - Streaming latency indicator

- **ITL** (Inter-Token Latency):
  - Server-side measurement
  - List of time deltas between consecutive tokens
  - Decoding phase performance

- **TPOT** (Time Per Output Token):
  - `mean(ITL)`
  - Average next-token latency

### 4.4 Testing Workflow

**Local Testing** (single host):
```bash
# 1. Start CARA server locally
python -m block.global_scheduler.cara.cara_serve \
  --host 127.0.0.1 \
  --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --scheduling random

# 2. Run benchmark
export PYTHONPATH="$HOME/vllm:$PYTHONPATH"  # If using local vLLM build
python block/benchmark/cara/benchmark_serving.py \
  --backend cara \
  --base-url http://127.0.0.1:8200 \
  --dataset-name random \
  --num-prompts 10
```

**Distributed Testing** (multi-host):
```bash
# Run E2E script from local desktop
./block/exp/cara/run_cara_e2e.sh

# Or run with forced redeployment
./block/exp/cara/run_cara_e2e.sh true
```

### 4.5 Debugging Tips

**Check CARA server logs**:
```bash
ssh user@target-host 'tail -f Block/experiment_output/logs/cara_server.log'
```

**Check backend instance logs**:
```bash
# vLLM
ssh user@backend-host 'tail -f ~/vllm/vllm_server.log'

# Ollama
ssh user@backend-host 'tail -f ~/ollama/ollama_server.log'
ssh user@backend-host 'tail -f ~/ollama/ollama_pull.log'
ssh user@backend-host 'tail -f ~/ollama/ollama_warmup.log'
```

**Common issues**:

1. **ModuleNotFoundError: vllm.tokenizers**
   - Problem: Benchmark needs vLLM in PYTHONPATH
   - Solution: `export PYTHONPATH="$HOME/vllm:$PYTHONPATH"`
   - Or: Use older import `from vllm.transformers_utils.tokenizer import get_tokenizer`

2. **Ollama connection refused**
   - Problem: Ollama not listening on 0.0.0.0
   - Solution: Ensure `OLLAMA_HOST=0.0.0.0:11434` in deployment script

3. **Port already in use**
   - Problem: Previous instance not cleaned up
   - Solution: Deployment script includes cleanup commands (pkill, fuser -k)

---

## 5. Future Development Roadmap

### Phase 1: Core Scheduling (Next Steps)
- [ ] Implement greedy mini-batch scheduler (Algorithm 1)
- [ ] Add objective function evaluation with configurable weights
- [ ] Support RSO parsing in request payload
- [ ] Implement constraint checking with confidence bounds

### Phase 2: Predictive Components
- [ ] Implement Performance Predictor (sequence regression model)
- [ ] Add `/status` API to vLLM/Ollama instances
- [ ] Implement online data collection
- [ ] Add model retraining pipeline
- [ ] Implement Model Estimator microservice

### Phase 3: Advanced Features
- [ ] Dynamic adaptive batching
- [ ] Beam search scheduler variant
- [ ] Soft length control integration
- [ ] ML-based MILP approximators

### Phase 4: Extensions (From Paper Appendix)
- [ ] Dynamic instance provisioning (serverless LLM)
- [ ] Prefill-Decode disaggregation scheduling
- [ ] KV-cache reusing support
- [ ] Auto-scaling and failure recovery

---

## 6. Key Files Reference

### CARA Core
| File | Purpose | Status |
|------|---------|--------|
| `block/global_scheduler/cara/cara_serve.py` | Global scheduler FastAPI server | ✅ Implemented |
| `block/global_scheduler/cara/cara_instance/Instance.py` | Abstract base class for instances | ✅ Implemented |
| `block/global_scheduler/cara/cara_instance/vllm_instance.py` | vLLM backend adapter | ✅ Implemented |
| `block/global_scheduler/cara/cara_instance/ollama_instance.py` | Ollama backend adapter | ✅ Implemented |

### Benchmarking
| File | Purpose | Status |
|------|---------|--------|
| `block/benchmark/cara/benchmark_serving.py` | Main benchmark tool | ✅ Implemented |
| `block/benchmark/cara/cara_end_point_func.py` | CARA request function | ✅ Implemented |

### Deployment
| File | Purpose | Status |
|------|---------|--------|
| `block/exp/cara/deploy_cara.py` | Multi-host deployment script | ✅ Implemented |
| `block/exp/cara/run_cara_e2e.sh` | E2E workflow automation | ✅ Implemented |

### Configuration
| File | Purpose | Status |
|------|---------|--------|
| `block/config/cara/model_config_template.json` | Model deployment specification | ✅ Implemented |
| `block/config/cara/model_deployment.json` | Generated deployment config | ✅ Auto-generated |
| `block/config/host_configs.json` | Cluster host information | ✅ Implemented |
| `block/config/hosts` | Host list file | ✅ Implemented |

### Documentation
| File | Purpose |
|------|---------|
| `cara.tex` | Research paper (CARA proposal) |
| `readme.md` | Block project documentation |
| `claude.md` | This development guide |

---

## 7. Research Context

### Block Paper Contributions
- Predictive scheduling using Vidur simulation
- Side-car predictor architecture
- Length estimation for unseen prompts
- Comparison with LLumnix, round-robin baselines

### CARA Paper Contributions
- Unified routing + scheduling optimization
- Request-Specific Objectives (RSO) API
- Greedy mini-batch scheduling algorithm
- Dynamic adaptive batching
- Heterogeneous backend support (vLLM + Ollama + ...)
- Scheduling overhead as a first-class metric

### Key Differences: Block vs CARA

| Aspect | Block | CARA |
|--------|-------|------|
| **Scope** | Single backend type (vLLM) | Heterogeneous backends |
| **Scheduling** | Predictive load balancing | Global batch co-scheduling |
| **API** | Standard OpenAI | Extended with RSOs |
| **Optimization** | Minimize latency | Multi-objective MILP |
| **Batching** | Instance-local | Global cross-instance |
| **Metrics** | TTFT, ITL, E2EL | + Scheduling overhead |

---

## 8. Citation

**Block**:
```bibtex
@misc{blockllm,
  title   = {Block: Balance Loader of Language-Model Instances with Context and Knowledge},
  author  = {Anonymous},
}
```

**CARA**:
```bibtex
@misc{cara2025,
  title   = {Cara: Co-optimizing Super Heterogeneous LLM Serving through
             Model Routing and Predictive Batch Scheduling},
  author  = {Anonymous Authors},
}
```

---

## 9. License

Both Block and CARA are released under the MIT license.

---

**Last Updated**: 2025-12-11
