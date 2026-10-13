# MULTI-TOPOLOGY & FEDADAM TESTING AND VERIFICATION PLAN
## Rigorous Test Suite, Validation Benchmarks, and Verification Milestones

> **Document Location:** `prototype/prompts/TESTING_AND_VERIFICATION_PLAN.md`  
> **Target Subsystem:** Traffic Engineering Prototype (`prototype/`)  
> **Scope:** Verification of GÉANT (22 nodes, 72 directed links), Abilene (12 nodes, 30 directed links, real traces), FedAdam optimization, and generalized RL environment.

---

## 1. Overview & Test Architecture

This plan establishes a multi-tier testing framework ensuring the newly integrated topologies (GÉANT, Abilene) and the upgraded federated optimization engine (**FedAdam**) satisfy all functional, architectural, and performance constraints without regressing baseline NSFNET functionality.

```
+-----------------------------------------------------------------------------------+
|                            LEVEL 4: SYSTEM BENCHMARK                              |
|   Full Scenario Evaluations (T1-T6), Wilcoxon Significance, Cross-Topology Tests  |
+-----------------------------------------+-----------------------------------------+
                                          |
+-----------------------------------------v-----------------------------------------+
|                        LEVEL 3: TRAINING INTEGRATION                              |
|   FL FedAdam Convergence (Rounds 1-20), RL Policy Training (Smoke & Full 200k)     |
+-----------------------------------------+-----------------------------------------+
                                          |
+-----------------------------------------v-----------------------------------------+
|                       LEVEL 2: PIPELINE & PREPROCESSING                           |
|   Real-Trace Loading, Gravity Model Calibration, Sliding Windows, Feature Padding  |
+-----------------------------------------+-----------------------------------------+
                                          |
+-----------------------------------------v-----------------------------------------+
|                        LEVEL 1: UNIT & SCHEMA TESTS                               |
|   JSON Graph Schemas, Matrix Shapes, Isomorphism, Privacy & Local Bounds         |
+-----------------------------------------------------------------------------------+
```

---

## 2. Level 1: Unit & Schema Verification

### 1.1 Topology Graph Invariant Tests
Create `tests/test_multi_topology.py`:
- **GÉANT**:
  - $|V| = 22$, Directed links $|E| = 72$ ($36$ bidirectional pairs).
  - Graph is weakly and strongly connected (no isolated islands or sinks).
  - Exact 3 domains (`A`, `B`, `C` or `0`, `1`, `2`) covering all 22 nodes ($7 + 8 + 7$).
  - Every link specifies non-zero `capacity_bps` and positive `delay_ms`.
- **Abilene**:
  - $|V| = 12$, Directed links $|E| = 30$ ($15$ bidirectional pairs).
  - Graph is strongly connected.
  - Exact 3 domains ($4 + 4 + 4$).
  - Nodes match the 12 canonical router identifiers (ATLA, CHIN, DNVR, HSTN, IPLS, KSCY, LOSA, NYCM, SNVA, STTL, WASH, SUNN).
- **NSFNET Backward Compatibility**:
  - Existing `test_topology_validity` continues to pass with 14 nodes and 22 links (44 edges).

### 1.2 Data Integrity Tests
- **Abilene Real NetFlow Loader (`tests/test_abilene_data.py`)**:
  - Verify that `load_abilene.py` parses `data/generate_nsfnets/Abilene_Evaluation.csv` into shape `(N, 12, 12)`.
  - Assert zero `NaN`s, zero negative demand values, and valid time monotonic progression.
  - Confirm diagonal demands ($TM_{ii} = 0$) are zeroed.
- **GÉANT Real & Synthetic Loaders (`tests/test_geant_data.py`)**:
  - Real loader parses SNDlib matrix archives to shape `(N, 22, 22)`.
  - Fallback synthetic generator produces `(8064, 22, 22)` with peak link utilization calibrated to $85\% \pm 5\%$ under OSPF routing.

---

## 3. Level 2: Data Preprocessing & FL Domain Partition Tests

### 2.1 Domain Privacy Invariant Test
- In `tests/test_preprocess_multitopo.py`:
  - Run preprocessing across all topologies:
    ```bash
    python -m data.preprocess --topology nsfnet
    python -m data.preprocess --topology geant
    python -m data.preprocess --topology abilene
    ```
  - **Privacy Check**: Assert that for each domain $d \in \{0, 1, 2\}$, the feature matrix $X_{\text{domain}}$ contains *only* links where at least one endpoint belongs to domain $d$.
  - Assert that client loaders never receive raw global OD matrices or inter-domain routing flow tables.

### 2.2 Tensor Dimensions & Windowing Tests
Verify exported NumPy tensors in `data/preprocessed_{topology}/`:
- Lookback window $T = 12$, Prediction horizon $H = 3$.
- Shapes:
  - NSFNET: $X \in (N, 12, F_d)$, $Y \in (N, 3, F_d)$ with $F_d \le 24$.
  - GÉANT: $X \in (N, 12, F_d)$, $Y \in (N, 3, F_d)$ with $F_d \le 30$.
  - Abilene: $X \in (N, 12, F_d)$, $Y \in (N, 3, F_d)$ with $F_d \le 16$.
- Check train/val/test splits integrity without temporal leakage ($t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$).

---

## 4. Level 3: Federated Learning & FedAdam Optimization Tests

### 3.1 Optimizer Convergence Benchmark (FedAvg vs. FedAdam)
Implement benchmark test in `tests/test_fedadam.py`:
- Run 20 rounds of FL training on GÉANT (non-IID regional diurnal traffic):
  ```bash
  python -m fl.train_fl --topology geant --strategy fedavg --rounds 20 --model_save_path models/fl_geant_fedavg.pt
  python -m fl.train_fl --topology geant --strategy fedadam --rounds 20 --model_save_path models/fl_geant_fedadam.pt
  ```
- **Validation Metric**: Evaluate with `python -m fl.evaluate_fl --topology geant --model_path ...`:
  - Compute Mean Squared Error (MSE), MAE, and $R^2$ on unseen test partition.
  - **Success Criteria**: FedAdam must achieve lower validation loss than FedAvg ($\ge 10\%$ MSE reduction) under heterogeneous multi-domain traffic.
- Repeat verification for Abilene with real NetFlow traces.

### 3.2 Flower Strategy & Local Fallback Equivalence
- Run Flower simulation mode and fallback mode `run_local_fedadam` for 2 rounds on synthetic data.
- Confirm state dict weight shapes and update directions match.

---

## 5. Level 4: RL Environment & Agent Generalization Tests

### 4.1 Parameterized Environment Sanity (`tests/test_env_multitopo.py`)
- Initialize `NetworkRoutingEnv` with each topology:
  1. NSFNET: `obs_dim = 108`, `action_dim = 20`.
  2. GÉANT: `obs_dim = 174` ($72 + 72 + 30$), `action_dim = 30`.
  3. Abilene: `obs_dim = 75` ($30 + 30 + 15$), `action_dim = 15`.
- Execute 100 random steps on each environment:
  - Confirm observations stay strictly in $[0.0, 1.0]$.
  - Confirm rewards are bounded and non-NaN.
  - Test link failure injection (`env.disable_link()`) dynamically re-routes paths using Yen's $K$-shortest paths.

### 4.2 Smoke-Test Training Runs
Execute quick 5,000-timestep PPO training runs to guarantee gradient stability:
```bash
python -m rl.train_rl --topology geant --timesteps 5000 --n_envs 4
python -m rl.train_rl --topology abilene --timesteps 5000 --n_envs 4
```
- Verify absence of CUDA/CPU memory leaks, NaN losses, or action dimension mismatches.

---

## 6. Level 5: Full Benchmark & Academic Evaluation Suite

### 6.1 Multi-Scenario Benchmark Execution
Run comprehensive evaluation across all 5 standard scenarios:
- **T1: Normal Diurnal Traffic**
- **T2: Traffic Spike (3× top 3 OD pairs)**
- **T3: Gradual Load (+50% over episode)**
- **T4: Flash Crowd (5× top 5 OD pairs)**
- **T5: Critical Link Failure**

```bash
python -m rl.evaluate_rl --topology geant --seeds 42 43 44 45 46 --out_dir results/geant
python -m rl.evaluate_rl --topology abilene --seeds 42 43 44 45 46 --out_dir results/abilene
```

### 6.2 Cross-Topology Zero-Shot Transfer Test (Advanced)
- Load agent trained on GÉANT ($F_{\text{in}} = 72, N_{\text{OD}} = 30$).
- Evaluate on Abilene using pad-masking (zeroing unmapped inputs and ignoring extraneous action logits).
- Compare against native OSPF shortest-path on Abilene to assess domain transferability.

### 6.3 Plot & Report Generation
Execute demonstration visualizers:
```bash
python scripts/demo.py --topology geant --no_random --out_dir results/demo/geant
python scripts/demo.py --topology abilene --no_random --out_dir results/demo/abilene
```
Verify generated charts in `results/demo/`:
- Multi-scenario MLU timelines (`overview_all_scenarios_mlu.png`).
- Delay, packet loss, and reward comparative curves.
- Statistical summary tables with Wilcoxon $p$-values ($p < 0.05$ denoted with $*$).

---

## 7. Execution Checklist & Acceptance Gates

| Gate | Description | Command / Check | Pass Criteria |
| :---: | :--- | :--- | :--- |
| **G1** | Schema & Unit Tests | `python -m pytest tests/` | 100% tests pass |
| **G2** | Data Preprocessing | `python -m data.preprocess --topology [all]` | No NaNs, exact domain bounds |
| **G3** | FL FedAdam Gain | `python -m fl.evaluate_fl --topology geant` | MSE lower than FedAvg |
| **G4** | RL Training Stability | `python -m rl.train_rl --topology [all] --timesteps 200000` | Smooth reward curve, low MLU |
| **G5** | Scenario Robustness | `python -m rl.evaluate_rl --topology [all]` | Outperforms OSPF/ECMP under T2, T4, T5 |
| **G6** | Visual Telemetry | `python scripts/demo.py --topology [all]` | High-res PNG plots generated |

---
*End of Testing & Verification Plan.*
