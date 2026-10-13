# GEMINI FLASH 3.8 IMPLEMENTATION PLAN & SPECIFICATION PROMPT
## Multi-Topology Scaling (GÉANT & Abilene) and Federated Optimizer Upgrade (FedAvg → FedAdam)

> **Target Model:** Gemini Flash 3.8 (High)  
> **Target Subsystem:** Traffic Engineering Prototype (`prototype/`)  
> **Objective:** Extend the Federated Learning + Reinforcement Learning Traffic Controller from single-topology NSFNET to multi-topology benchmark suites (**GÉANT** for large-scale training, **Abilene** for real-trace validation) while upgrading the FL aggregation engine from standard **FedAvg** to adaptive **FedAdam** with server-side momentum, preserving zero-leakage privacy across domains.

---

## 1. Executive Summary & Architectural Scope

The current production system operates on **NSFNET** ($|V|=14, |E|=42$ directed links) with:
1. **Domain-decentralized FL**: 3 domains ($A, B, C$), local LSTM predictors trained strictly on domain-visible link utilization sequences ($T=12, H=3$).
2. **PPO Centralized Controller**: Continuous state observation vector $\mathcal{S} \in \mathbb{R}^{2|E| + N_{\text{OD}}}$, action space $\mathcal{A} \in \{0, 1, 2\}^{N_{\text{OD}}}$ selecting one of $K=3$ pre-computed Yen shortest paths for the top $N_{\text{OD}}$ managed OD pairs.
3. **Flower FedAvg Strategy**: Parameter averaging across clients with synchronous fallback.

### New Requirements to Implement:
1. **Topology & Dataset Generalization**:
   - Primary Training Benchmark: **GÉANT** ($|V|=22, |E|=72$ directed links, 3 domains, 462 OD pairs, top 30 managed) using **real SNDlib/TOTEM NetFlow traces** as primary dataset and calibrated **synthetic gravity generation** as deterministic fallback.
   - Real-Trace Evaluation Benchmark: **Abilene** ($|V|=12, |E|=30$ directed links, 132 OD pairs, parsed from real historical NetFlow matrices in `Abilene_Evaluation.csv`).
   - Dynamic Environment Scaling: Parameterize `rl/env.py`, `rl/train_rl.py`, and `rl/evaluate_rl.py` so topology path, TM path, domain mappings, and managed OD pair count are configurable via CLI arguments without hardcoding.
2. **FL Strategy Upgrade (FedAvg → FedAdam)**:
   - Implement **FedAdam** adaptive server optimizer (Reddi et al., ICLR 2021) in `fl/server.py`.
   - Update synchronous fallback `run_local_fedadam` in `fl/train_fl.py` to maintain exact mathematical equivalence when Flower/Ray orchestration is unavailable.
   - Maintain the **Domain-Link Privacy Invariant**: Clients NEVER exchange OD traffic matrices or link states outside their topological monitoring boundary.

---

## 2. Multi-Topology Specifications & Domain Partitions

### 2.1 NSFNET (Baseline Reference)
- Nodes: 14, Directed Links: 44 (22 bidirectional links in `nsfnet_topology.json`).
- Domains: 3 (Domain 0: {0,1,2,7,8}; Domain 1: {3,4,5,9}; Domain 2: {6,10,11,12,13}).
- Managed OD pairs: $N_{\text{OD}} = 20$.
- State Dimension: $44 \text{ (current)} + 44 \text{ (predicted)} + 20 \text{ (demands)} = 108$.
- Maximum Domain Link Feature Width: $\text{MAX\_F\_IN} = 24$.

### 2.2 GÉANT (Large-Scale European Research Network)
- Topology File: `data/geant_topology.json`.
- Nodes: **22 nodes** (representing European NREN PoPs: UK, FR, NL, DE, IT, ES, CH, AT, BE, SE, etc.).
- Links: **36 bidirectional edges = 72 directed links**. All capacities 10 Gbps (normalized to 10 Mbps for emulation), delay 2–15 ms proportional to Euclidean distance.
- Domains (3-way geographical partition):
  - **Domain 0 (Western / Northern Europe)**: 7 nodes (e.g., UK, IE, FR, BE, NL, DE, SE).
  - **Domain 1 (Central / Eastern Europe)**: 8 nodes (e.g., CH, AT, CZ, PL, HU, SK, SI, HR).
  - **Domain 2 (Southern / Mediterranean Europe)**: 7 nodes (e.g., IT, ES, PT, GR, TR, IL, CY).
- OD Pairs: $22 \times 21 = 462$ directed pairs.
- Managed OD Pairs: $N_{\text{OD}} = 30$ (accounting for ~65% of total network traffic).
- State Dimension: $72 \text{ (current)} + 72 \text{ (predicted)} + 30 \text{ (demands)} = 174$.
- Action Dimension: `spaces.MultiDiscrete([3] * 30)`.
- **Dataset Strategy (Dual Approach)**:
  - **Primary (Real Traces)**: Download & ingest historical SNDlib / TOTEM GÉANT traffic matrices (2005 15-minute NetFlow XML/gzip or CSV archives, normalized to Mbps).
  - **Fallback (Synthetic Gravity Model)**: Calibrated gravity model generator with diurnal sinusoidal wave patterns in `data/generate_tm.py --topo data/geant_topology.json`.

### 2.3 Abilene (Real NetFlow Trace Network)
- Topology File: `data/abilene_topology.json`.
- Nodes: **12 nodes** (ATLA, CHIN, DNVR, HSTN, IPLS, KSCY, LOSA, NYCM, SNVA, STTL, WASH, SUNN).
- Links: **15 bidirectional edges = 30 directed links**.
- Raw Dataset: `data/generate_nsfnets/Abilene_Evaluation.csv` containing 48,096 consecutive 5-minute NetFlow snapshots (132 OD columns: `OD_1-2` to `OD_12-11`, time range 2004-03-01 to 2004-08-15).
- Scale Conversion: NetFlow values are in kbps. Multiply by $10^3$ to obtain bps or normalize by global $99.9^{\text{th}}$ percentile peak.
- Domains (3-way regional partition):
  - **Domain 0 (West)**: {LOSA, SNVA, STTL, DNVR} (4 nodes).
  - **Domain 1 (Central)**: {KSCY, IPLS, CHIN, HSTN} (4 nodes).
  - **Domain 2 (East)**: {ATLA, WASH, NYCM, SUNN} (4 nodes).
- Managed OD Pairs: $N_{\text{OD}} = 15$.
- State Dimension: $30 \text{ (current)} + 30 \text{ (predicted)} + 15 \text{ (demands)} = 75$.

---

## 3. Mathematical Formulation: FedAdam Upgrade

Standard FedAvg performs vanilla weighted parameter averaging:
$$\Delta_t^k = w_{t, \tau}^k - w_t$$
$$\Delta_t = \sum_{k=1}^K \frac{n_k}{n} \Delta_t^k$$
$$w_{t+1} = w_t + \Delta_t$$

When traffic distributions across domains are non-IID (heterogeneous diurnal curves and cross-border bursts), FedAvg exhibits client drift. **FedAdam** replaces the server-side update with adaptive moment estimation:

1. **Pseudo-gradient computation**:
   $$\Delta_t = \sum_{k=1}^K \frac{n_k}{n} \left( w_{t, \tau}^k - w_t \right)$$
2. **First moment (momentum)**:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \Delta_t$$
3. **Second moment (uncentered variance)**:
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \Delta_t^2$$
4. **Server parameter update**:
   $$w_{t+1} = w_t + \eta \frac{m_t}{\sqrt{v_t} + \tau}$$

### Recommended Hyperparameters:
- Server Learning Rate ($\eta$): $10^{-3}$ ($0.001$)
- Client Local Learning Rate ($\eta_L$): $10^{-3}$ ($0.001$) with Adam
- Exponential Decay $\beta_1$: $0.9$
- Exponential Decay $\beta_2$: $0.99$
- Degree of Adaptivity / Numerical Stability ($\tau$): $10^{-5}$ (preventing division-by-zero without over-dampening)
- Local Epochs: $E = 5$, Batch Size: $B = 32$

---

## 4. Step-by-Step Implementation Instructions

### Step 1: Create Topology JSON Files
Create `data/geant_topology.json` and `data/abilene_topology.json` matching the schema in `data/nsfnet_topology.json`:
```json
{
  "nodes": [
    {"id": 0, "label": "ATLA", "city": "Atlanta, GA", "domain": "C", "population_weight": 5.0}
  ],
  "links": [
    {"source": 0, "target": 1, "capacity_bps": 10000000, "delay_ms": 4.5}
  ]
}
```

### Step 2: Implement Real Abilene Trace Ingestion Script
Create `data/load_abilene.py`:
- Ingest `data/generate_nsfnets/Abilene_Evaluation.csv`.
- Parse timestamp and 132 OD pair columns.
- Re-index node IDs (1–12 in CSV to 0–11 zero-indexed).
- Resample / verify no missing timestamps.
- Export as `data/generated/abilene_traffic_matrices.npy` of shape `(N_timesteps, 12, 12)` in Mbps.

### Step 3: GÉANT Traffic Dataset Pipeline (Primary Real Traces + Synthetic Fallback)
#### Step 3A: Real GÉANT Trace Downloader & Ingestion Script (`data/load_geant_real.py`)
- Source: SNDlib GÉANT benchmark archive / TOTEM project repository (462 OD pairs, 22 nodes, 15-minute sampling intervals).
- Download and parse raw GÉANT XML or matrix text files into a clean NumPy array of shape `(N_timesteps, 22, 22)`.
- Handle node name-to-index mapping matching `data/geant_topology.json`.
- Normalize traffic volumes from raw kbps/bps to simulation units (Mbps).
- Export to `data/generated/geant_real_traffic_matrices.npy`.

#### Step 3B: Synthetic TM Generator Fallback (`data/generate_tm.py`)
- Extend `data/generate_tm.py` to accept CLI argument `--topo data/geant_topology.json`.
- Dynamically extract node population weights and hop-count matrix for the 22-node European topology.
- Calibrate $\alpha$ using shortest path routing to achieve 85% peak link utilization.
- Output `data/generated/geant_synthetic_traffic_matrices.npy` of shape `(8064, 22, 22)`.
- This serves as a deterministic fallback if external dataset hosting is unreachable or for controlled diurnal stress tests.

### Step 4: Multi-Topology Preprocessing (`data/preprocess.py`)
Modify `data/preprocess.py` to support `--topology [nsfnet|geant|abilene]`:
- Load chosen topology JSON and corresponding TM `.npy` file.
- Compute link utilizations using Yen/Dijkstra shortest paths across all directed edges.
- Partition links into domain sub-graphs based on the `domain` node attribute.
- For each domain $d \in \{0, 1, 2\}$, compute:
  - `domain_{d}_edge_indices.json`: Directed link indices in global edge list.
  - Generate sliding window tensors $X \in \mathbb{R}^{N \times 12 \times F_d}$, $Y \in \mathbb{R}^{N \times 3 \times F_d}$.
  - Save train/val/test splits in `data/preprocessed_{topology}/`.

### Step 5: Upgrade Federated Learning Server & Training (`fl/server.py` & `fl/train_fl.py`)
1. **In `fl/server.py`**:
   - Subclass `fl.server.strategy.FedAdam`.
   - Implement `SaveModelStrategy` with `aggregate_fit` that updates server moments and saves `models/fl_global_model_{topology}.pt`.
   - Pass `initial_parameters` derived from a dummy model with `f_in = max_f_in`.
2. **In `fl/train_fl.py`**:
   - Add CLI arguments: `--topology`, `--data_dir`, `--model_path`, `--strategy [fedavg|fedadam]`.
   - Upgrade fallback function `run_local_fedadam`:
     Maintain momentum vectors $m$ and velocity vectors $v$ per parameter tensor across rounds.
     Apply Adam updates to global parameters using aggregated client pseudo-gradients.

### Step 6: Parameterize the Gym Environment (`rl/env.py`)
Refactor `NSFNETRoutingEnv` $\to$ `NetworkRoutingEnv`:
- Constructor signature:
  ```python
  def __init__(
      self,
      topo_path: str,
      tm_path: str,
      fl_model_path: Optional[str] = None,
      domain_data_dir: Optional[str] = None,
      num_managed_od: int = 20,
      surge_prob: float = 0.0
  )
  ```
- Dynamically determine:
  - `num_nodes`, `num_edges`.
  - Top $N_{\text{OD}} = \text{num\_managed\_od}$ pairs based on average demand volume.
  - `observation_space`: `Box(low=0.0, high=1.0, shape=(2 * num_edges + num_managed_od,))`.
  - `action_space`: `MultiDiscrete([K] * num_managed_od)`.
- Update `DomainPredictor` to read domain edge indices from `domain_data_dir` and handle dynamic link counts without hardcoded dimensions.

### Step 7: Parameterize Training & Evaluation Scripts (`rl/train_rl.py` & `rl/evaluate_rl.py`)
- In `rl/train_rl.py`:
  - Add `--topology`, `--topo_path`, `--tm_path`, `--fl_model_path`, `--num_managed_od`.
  - Adjust policy network initialization bias for Path 0 prior across the dynamic `num_managed_od`.
- In `rl/evaluate_rl.py`:
  - Support evaluating trained policies across topologies.
  - Implement zero-shot or fine-tuned transfer testing (e.g., policy trained on GÉANT evaluated on Abilene with OSPF fallback).

---

## 5. Verification & Acceptance Criteria

1. **Topology Schema Integrity**:
   - `python -c "import json; [json.load(open(f)) for f in ['data/nsfnet_topology.json', 'data/geant_topology.json', 'data/abilene_topology.json']]"` exits with code 0.
2. **Preprocessed Dimensions**:
   - GÉANT: $X_{\text{global}}$ has feature dimension 72. Max domain $F_{\text{in}} \le 30$.
   - Abilene: $X_{\text{global}}$ has feature dimension 30. Max domain $F_{\text{in}} \le 16$.
3. **FL Convergence**:
   - FedAdam reduces validation MSE by $\ge 12\%$ compared to FedAvg after 20 rounds under non-IID diurnal traffic.
4. **Backward Compatibility**:
   - Running `python -m rl.train_rl --timesteps 1000` with default parameters continues to train NSFNET flawlessly.

---
*End of Specification Prompt.*
