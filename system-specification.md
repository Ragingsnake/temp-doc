# System Specification: FL + RL Proactive Traffic Engineering for SDN-Based ISP Networks

> [!IMPORTANT]
> This document is the **single source of truth** for all technical decisions. Reference individual sections by heading when prompting for paper-writing or coding tasks.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Network Topology — NSFNET](#2-network-topology--nsfnet)
3. [Data Pipeline](#3-data-pipeline)
4. [Federated Learning — Traffic Prediction](#4-federated-learning--traffic-prediction)
5. [Reinforcement Learning — Traffic Engineering (Full MDP)](#5-reinforcement-learning--traffic-engineering-full-mdp)
6. [Lightweight Network Simulator (Gym Environment)](#6-lightweight-network-simulator-gym-environment)
7. [SDN Emulation — Mininet + Ryu](#7-sdn-emulation--mininet--ryu)
8. [System Integration & Real-Time Data Flow](#8-system-integration--real-time-data-flow)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Infrastructure & Azure Student Budget](#10-infrastructure--azure-student-budget)
11. [Revised Timeline (15 weeks)](#11-revised-timeline-15-weeks)

---

## 1. System Overview

### 1.1 Problem Statement

Current ISP routing reacts **after** congestion occurs. This system uses predicted future traffic to make routing decisions **before** congestion happens.

### 1.2 Core Idea (One Sentence)

A Federated-Learning LSTM predicts link-level traffic for the next 15 minutes; a PPO Reinforcement Learning agent uses those predictions (plus current network state) to proactively select optimal routing paths per OD pair; the chosen paths are installed as OpenFlow rules via a Ryu SDN controller running on an NSFNET topology emulated in Mininet.

### 1.3 Architecture Components

| Component | Role | Technology |
|---|---|---|
| FL Client (×3–4) | Local LSTM training on domain-specific link data | PyTorch + Flower |
| FL Server | Aggregates client models via FedAvg | Flower |
| Traffic Predictor | Predicts future link utilization from the global LSTM | PyTorch (inference) |
| RL Agent | Selects routing paths per OD pair | Stable-Baselines3 (PPO) |
| Lightweight Sim | Fast training environment for RL | Custom OpenAI Gymnasium |
| SDN Controller | Installs flow rules on switches | Ryu (OpenFlow 1.3) |
| Network Emulator | Emulates NSFNET for validation & demo | Mininet + Open vSwitch |
| Dashboard | Visualizes topology, metrics, decisions | Grafana or Streamlit |

### 1.4 Two-Phase Design Rationale

> [!WARNING]
> Training RL directly inside Mininet is infeasible. Mininet emulates all 14 switches on a single CPU; adding PyTorch inference in the loop drops emulation throughput to ~1–5 steps/second. PPO requires 1–5 million steps to converge. That is **60–1400 hours** of wall-clock time. This project has ~\$100 of Azure credit.

**Solution — two phases:**

| Phase | Purpose | Environment | Speed |
|---|---|---|---|
| **Phase 1 — Training** | Train FL model, then train RL agent | Custom Gym env (pure Python math model) | ~2,000–5,000 steps/sec |
| **Phase 2 — Validation & Demo** | Verify trained models in realistic emulation | Mininet + Ryu | Real-time |

---

## 2. Network Topology — NSFNET

### 2.1 Topology Definition

NSFNET: **14 nodes**, **21 bidirectional links** (42 directed edges).

**Nodes** (indexed 0–13):

| Index | Label | City |
|---|---|---|
| 0 | WA | Seattle, WA |
| 1 | CA1 | Palo Alto, CA |
| 2 | CA2 | San Diego, CA |
| 3 | NE | Lincoln, NE |
| 4 | IL | Champaign, IL |
| 5 | MI | Ann Arbor, MI |
| 6 | GA | Atlanta, GA |
| 7 | TX | Houston, TX |
| 8 | UT | Salt Lake City, UT |
| 9 | CO | Boulder, CO |
| 10 | PA | Pittsburgh, PA |
| 11 | NY | Ithaca, NY |
| 12 | NJ | Princeton, NJ |
| 13 | DC | College Park, MD |

**Adjacency List** (bidirectional):

```
(0,1), (0,2), (0,7)
(1,2), (1,3)
(2,5)
(3,4), (3,8)
(4,5), (4,6)
(5,10), (5,13)
(6,7)
(7,9)
(8,9), (8,11), (8,12)
(9,10)
(10,11), (10,12)
(11,13)
(12,13)
```

> [!NOTE]
> Multiple NSFNET versions exist in the literature. The above is consistent with the one commonly used in SNDlib. If the exact edges differ in the supervisor's preferred reference, adjust accordingly — the spec's formulas are topology-agnostic (parameterized by |V|, |E|, adjacency).

### 2.2 Link Parameters (Emulation)

| Parameter | Value | Rationale |
|---|---|---|
| Link capacity (all links) | **10 Mbps** | Mininet on a single VM; 10 Mbps makes congestion observable without needing massive traffic generators. Scale results by ratio if reporting at higher capacities. |
| Propagation delay per link | **2–10 ms** (proportional to geographic distance) | Keeps end-to-end delay realistic in the 10–60 ms range across the US. |
| Queue size per port | **100 packets** (Open vSwitch default) | Standard; allows measurable packet loss under congestion. |

### 2.3 FL Domain Partitioning

Divide the 14 nodes into **3 domains** for Federated Learning. Each domain corresponds to one FL Client.

| Domain | Nodes | Approx. Region | # Internal Links |
|---|---|---|---|
| Domain A (West) | 0, 1, 2, 7, 8 | West Coast + TX/UT | 6 |
| Domain B (Central) | 3, 4, 5, 9 | Midwest | 5 |
| Domain C (East) | 6, 10, 11, 12, 13 | East Coast + GA | 7 |

- **Internal links** = links where both endpoints are in the same domain.
- **Cross-domain links** = links connecting nodes in different domains. These are monitored by both adjacent domains.
- Each FL Client trains on the utilization time series of its **internal + adjacent cross-domain links**.

### 2.4 OD Pairs & Path Computation

- **Total OD pairs** (directed): 14 × 13 = 182.
- **Managed OD pairs (N_OD)**: Select the **top 20** OD pairs by average traffic volume from the dataset. Only these OD pairs have their routing actively managed by the RL agent. Remaining OD pairs use OSPF shortest-path by default.
- **K-shortest paths**: For each managed OD pair, pre-compute **K = 3** loop-free shortest paths using **Yen's algorithm** (by hop count, breaking ties by total propagation delay).
- Store as: `PathTable[od_pair_index] → [path_0, path_1, path_2]` where each `path_i` is an ordered list of node indices.

---

## 3. Data Pipeline

### 3.1 Dataset Selection

**Primary: Synthetic Traffic Matrices (Gravity Model)**

Generate traffic matrices for NSFNET directly, avoiding the need to adapt a foreign topology.

**Gravity Model:**

$$TM_{ij} = \alpha \cdot \frac{pop_i \cdot pop_j}{dist_{ij}^\beta}$$

Where:
- $pop_i$ = weight of node $i$ (proportional to metropolitan population, or set equal for simplicity)
- $dist_{ij}$ = hop-count distance between nodes $i$ and $j$
- $\alpha$ = scaling factor (tuned so that peak traffic creates ~80–90% utilization on the busiest link)
- $\beta$ = distance decay exponent (typically 1.0–2.0; use **1.0**)

**Temporal Patterns:**

$$TM_{ij}(t) = TM_{ij}^{base} \cdot \left(1 + A_{daily} \cdot \sin\left(\frac{2\pi t}{T_{day}}\right) + A_{weekly} \cdot \sin\left(\frac{2\pi t}{T_{week}}\right) + \epsilon(t)\right)$$

Where:
- $T_{day}$ = 288 timesteps (24h at 5-min intervals)
- $T_{week}$ = 2016 timesteps
- $A_{daily}$ = 0.3 (±30% daily swing)
- $A_{weekly}$ = 0.1 (±10% weekly swing)
- $\epsilon(t) \sim \mathcal{N}(0, \sigma^2)$, $\sigma = 0.05$ (5% Gaussian noise)

**Generate**: 4 weeks of data = 4 × 7 × 288 = **8,064 timesteps** (each timestep = one 14×14 traffic matrix).

**Secondary Validation: Abilene / GÉANT** (Optional)

- Abilene (Internet2): 12 nodes, 15 links, 5-min intervals, ~6 months of TMs.
  - Source: Zhang et al., [http://www.cs.utexas.edu/~yzhang/research/AbileneTM/](http://www.cs.utexas.edu/~yzhang/research/AbileneTM/)
  - Usage: Pad/interpolate to 14 nodes or use a subset of NSFNET nodes to match. Useful for validating that results generalize beyond synthetic data.

### 3.2 Data Preprocessing

1. **Compute link utilization time series** from TMs:
   - For each timestep $t$, route all TM demands using shortest paths → compute aggregate load on each link → $u_e(t) = \text{load}_e(t) / C_e$.
2. **Normalize**: Utilization values are already in [0, 1]. Traffic demand values are normalized by dividing by the global maximum demand: $d_{norm} = d / d_{max}$.
3. **Windowing**: Create sliding windows of length $T = 12$ (1 hour lookback) with prediction horizon $H = 3$ (15 minutes ahead).
   - Input: $X_t = [u(t-T+1), u(t-T+2), \ldots, u(t)]$ — shape $(T, |E_{domain}|)$
   - Target: $Y_t = [u(t+1), u(t+2), u(t+3)]$ — shape $(H, |E_{domain}|)$

### 3.3 Data Splits

| Split | Timesteps | Duration | Usage |
|---|---|---|---|
| Train | 1 – 5,760 | Weeks 1–3.5 | FL training |
| Validation | 5,761 – 6,912 | Week 3.5–4.2 | Hyperparameter tuning, early stopping |
| Test | 6,913 – 8,064 | Week 4.2–4 | Final evaluation, RL environment traffic source |

### 3.4 FL Data Distribution

Each FL Client receives **only** the link utilization columns corresponding to its domain's links (see §2.3). The FL Server never sees raw utilization data — only model weight updates.

---

## 4. Federated Learning — Traffic Prediction

### 4.1 Local Model Architecture (LSTM)

Each FL Client trains the **same architecture** locally.

| Layer | Specification |
|---|---|
| Input | Shape: $(T, F_{in})$ where $T = 12$, $F_{in} = |E_{domain}|$ (varies 5–8 per domain) |
| LSTM Layer 1 | Hidden size: **64**, batch_first=True |
| LSTM Layer 2 | Hidden size: **64**, batch_first=True |
| Dropout | Rate: **0.2** (between LSTM layers) |
| Fully Connected | Input: 64, Output: $H \times F_{in}$ where $H = 3$ |
| Reshape | Output shape: $(H, F_{in})$ — predicted utilization for next 3 timesteps |

**Estimated parameters per client**: ~37,000 (for $F_{in} = 7$). Tiny model, CPU-friendly.

### 4.2 FL Protocol

| Parameter | Value |
|---|---|
| Framework | Flower (flwr) |
| Number of clients | **3** (one per domain) |
| Aggregation strategy | **FedAvg** (simpler, well-understood; switch to FedAdam if FedAvg shows slow convergence) |
| Communication rounds | **100** |
| Local epochs per round | **5** |
| Local batch size | **32** |
| Local optimizer | Adam, learning rate **1e-3** |
| Loss function | **MSE** (Mean Squared Error) |
| Client selection | All clients participate every round (only 3 clients, no sampling needed) |

### 4.3 Global Model Assembly

After FL training converges:
1. The FL Server holds the **global model** — an LSTM with the same architecture as local models.
2. For **inference**, the global model receives concatenated link features from **all** domains: $F_{in}^{global} = |E| = 42$ (all directed links).
3. **Adaptation**: To handle the change in input dimension from local ($F_{in} \approx 7$) to global ($F_{in} = 42$), use one of:
   - **(Recommended) Per-domain inference + concatenation**: Each domain runs the global model on its own links, predictions are concatenated. No input dimension mismatch.
   - **(Alternative) Fine-tune a global head**: After aggregation, fine-tune the FC layer on a small centralized validation set to output all 42 link predictions.

### 4.4 Prediction Output

At each decision epoch $t$, the predictor outputs:

$$\hat{u}_{t+1}, \hat{u}_{t+2}, \hat{u}_{t+3} \in \mathbb{R}^{|E|}$$

For the RL agent, we use the **next-step prediction** $\hat{u}_{t+1}$ as the primary input. The 2nd and 3rd step predictions can be used for multi-step lookahead in future extensions.

### 4.5 Training Time Estimate

- Dataset: ~5,760 training timesteps → ~5,748 windows (after T+H offset)
- Per round: 3 clients × 5 epochs × (5748/32) batches ≈ 2,700 gradient updates total
- 100 rounds: ~270,000 gradient updates
- **Estimated wall-clock**: **30–90 minutes** on a 4-vCPU VM (CPU only, no GPU needed).

---

## 5. Reinforcement Learning — Traffic Engineering (Full MDP)

### 5.1 MDP Formulation Summary

| MDP Element | Definition |
|---|---|
| **State** $s_t$ | Current link utilization + predicted future utilization + traffic demands |
| **Action** $a_t$ | Path selection index for each managed OD pair |
| **Reward** $r_t$ | Weighted combination penalizing congestion, delay, and packet loss |
| **Transition** $P(s_{t+1} \mid s_t, a_t)$ | Determined by traffic evolution (from dataset) + routing effect of action |
| **Discount** $\gamma$ | 0.99 |

### 5.2 State Space $\mathcal{S}$

The state vector $s_t \in \mathbb{R}^{D_s}$ is a concatenation of three sub-vectors:

$$s_t = \left[ \; \mathbf{u}_t \; \| \; \hat{\mathbf{u}}_{t+1} \; \| \; \mathbf{d}_t \; \right]$$

| Sub-vector | Symbol | Dimension | Description | Range |
|---|---|---|---|---|
| Current link utilization | $\mathbf{u}_t$ | $|E| = 42$ | Utilization of each directed link at time $t$ | $[0, 1]$ |
| Predicted link utilization | $\hat{\mathbf{u}}_{t+1}$ | $|E| = 42$ | FL model's predicted utilization for time $t+1$ | $[0, 1]$ (clipped) |
| Traffic demands | $\mathbf{d}_t$ | $N_{OD} = 20$ | Normalized demand for each managed OD pair | $[0, 1]$ |

**Total state dimension: $D_s = 42 + 42 + 20 = 104$**

> [!NOTE]
> For the **"RL without prediction" baseline** (see §9.2), remove $\hat{\mathbf{u}}_{t+1}$ from the state. The state becomes $s_t = [\mathbf{u}_t \| \mathbf{d}_t]$ with $D_s = 62$.

### 5.3 Action Space $\mathcal{A}$

**Type**: `MultiDiscrete([K] * N_OD)` = `MultiDiscrete([3, 3, 3, ..., 3])` (20 entries)

$$a_t = (a_t^1, a_t^2, \ldots, a_t^{N_{OD}})$$

Where $a_t^i \in \{0, 1, 2\}$ selects which of the K=3 pre-computed paths to use for OD pair $i$.

**Total number of joint actions**: $3^{20} \approx 3.5 \times 10^9$. PPO handles this via its policy network — it does **not** enumerate all actions.

**Why discrete path selection over continuous splitting ratios:**
1. Simpler implementation and interpretation.
2. SDN flow rules map naturally to a single path per flow (install one rule per switch on the chosen path).
3. Avoids the complexity of ECMP-like splitting in Mininet validation.
4. Academically sufficient for demonstrating proactive TE; splitting ratios can be a future extension.

### 5.4 Reward Function $r_t$

At each step, after action $a_t$ is applied and the network transitions to the new state, compute:

$$r_t = -\left(\alpha \cdot \text{MLU}_t + \beta \cdot \bar{D}_t + \gamma \cdot \bar{L}_t\right)$$

Where:

| Term | Formula | Description |
|---|---|---|
| $\text{MLU}_t$ | $\displaystyle\max_{e \in E} \; u_e(t)$ | Maximum Link Utilization. Primary congestion indicator. |
| $\bar{D}_t$ | $\displaystyle\frac{1}{N_{OD}} \sum_{i=1}^{N_{OD}} D_i(t)$ | Average end-to-end delay across managed OD pairs (normalized by $D_{max}$). |
| $\bar{L}_t$ | $\displaystyle\frac{1}{N_{OD}} \sum_{i=1}^{N_{OD}} L_i(t)$ | Average packet loss ratio across managed OD pairs. |

**Weighting coefficients:**

| Coefficient | Value | Rationale |
|---|---|---|
| $\alpha$ | **1.0** | MLU is the most important TE metric; directly measures congestion risk. |
| $\beta$ | **0.5** | Delay matters but is secondary to preventing congestion. |
| $\gamma$ | **2.0** | Packet loss is severe; heavily penalize any loss. |

> [!TIP]
> These weights are starting values. Tune via ablation study: fix two, sweep the third. Report sensitivity in the thesis.

**Per-link delay model** (used in both Gym env and for calculating $\bar{D}_t$):

$$d_e = d_e^{prop} + \frac{S}{C_e} + d_e^{queue}$$

Where:
- $d_e^{prop}$ = propagation delay (fixed, 2–10 ms per link)
- $S$ = average packet size (1500 bytes = 0.012 Mbit)
- $C_e$ = link capacity (10 Mbps → $S/C_e$ = 1.2 µs, negligible)
- $d_e^{queue}$ = queuing delay, modeled as M/M/1:

$$d_e^{queue} = \begin{cases} \frac{1}{\mu_e - \lambda_e} = \frac{u_e}{C_e \cdot (1 - u_e)} & \text{if } u_e < 0.95 \\ d_{max}^{queue} = 50\text{ ms} & \text{if } u_e \geq 0.95 \end{cases}$$

**End-to-end delay for OD pair $i$** routed on path $p_i$:

$$D_i(t) = \sum_{e \in p_i} d_e(t)$$

**Normalize** by $D_{max}$ (the theoretical maximum delay when all links are at $u=0.94$) to keep $\bar{D}_t \in [0, 1]$.

**Per-link packet loss model:**

$$L_e = \begin{cases} 0 & \text{if } u_e \leq 0.90 \\ \frac{u_e - 0.90}{0.10} & \text{if } 0.90 < u_e < 1.00 \\ 1.0 & \text{if } u_e \geq 1.00 \end{cases}$$

Linear ramp from 0% loss at 90% utilization to 100% loss at 100% utilization.

**OD pair loss**: $L_i = 1 - \prod_{e \in p_i}(1 - L_e)$ (probability of at least one link dropping the packet).

### 5.5 Policy & Value Network Architecture

Using **PPO** from Stable-Baselines3 with a custom `MultiInputPolicy` or `MlpPolicy`.

**Actor (Policy) Network:**

| Layer | Specification |
|---|---|
| Input | $D_s = 104$ |
| Hidden 1 | Dense(256), ReLU |
| Hidden 2 | Dense(128), ReLU |
| Output heads | 20 parallel heads, each Dense(3) with Softmax → action probabilities for one OD pair |

**Critic (Value) Network:**

| Layer | Specification |
|---|---|
| Input | $D_s = 104$ |
| Hidden 1 | Dense(256), ReLU |
| Hidden 2 | Dense(128), ReLU |
| Output | Dense(1) → scalar state value $V(s)$ |

**Total estimated parameters**: ~60,000–80,000. CPU-trainable.

### 5.6 PPO Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO (clip) |
| Learning rate | **3e-4** (with linear decay to 1e-5) |
| Clip range $\epsilon$ | **0.2** |
| GAE $\lambda$ | **0.95** |
| Discount $\gamma$ | **0.99** |
| Entropy coefficient | **0.01** |
| Value function coefficient | **0.5** |
| Max gradient norm | **0.5** |
| Number of environments (vectorized) | **8** (parallel Gym envs) |
| Steps per rollout per env | **2048** |
| Number of epochs per update | **10** |
| Minibatch size | **64** |
| Total training timesteps | **2,000,000** |

### 5.7 RL Training Time Estimate

- Gym env speed: ~3,000 steps/sec (8 vectorized envs on 4 vCPUs)
- 2M steps: ~667 seconds ≈ **11 minutes**
- Including PPO gradient updates: **15–30 minutes** total
- May need up to 5M steps if convergence is slow: **~1 hour**

### 5.8 Episode Structure

| Parameter | Value |
|---|---|
| Episode length | **100 steps** (= 100 × 30 sec = 50 min simulated time) |
| Decision interval | **30 seconds** (simulated). Every 30 seconds the agent observes and acts. |
| Traffic source | Sampled from training split of the synthetic TM dataset. Each episode starts at a random offset in the dataset. |
| Reset | Re-sample starting time offset; re-initialize all link utilizations to 0; inject traffic for first timestep. |
| Terminal condition | After 100 steps, or if all links exceed 95% utilization (network collapse). |

---

## 6. Lightweight Network Simulator (Gym Environment)

> [!IMPORTANT]
> This is the **most critical engineering component** of the project. The Gym env must faithfully approximate Mininet's behavior while running 1000× faster.

### 6.1 Gymnasium Interface

```
Class: NSFNETRoutingEnv(gymnasium.Env)

observation_space: Box(low=0, high=1, shape=(104,), dtype=float32)
action_space:      MultiDiscrete([3] * 20)
```

### 6.2 Environment State

Internal state variables (not all exposed to the agent):

| Variable | Type | Description |
|---|---|---|
| `topology` | NetworkX DiGraph | NSFNET graph with link capacities and propagation delays |
| `path_table` | dict[int, list[list[int]]] | Pre-computed K=3 paths per OD pair |
| `current_tm` | ndarray (14, 14) | Current traffic matrix |
| `link_loads` | ndarray (42,) | Aggregate traffic load on each directed link |
| `link_utils` | ndarray (42,) | link_loads / capacity |
| `time_index` | int | Current position in the traffic dataset |
| `fl_model` | nn.Module | Trained FL LSTM model (loaded, inference-only) |
| `history_buffer` | deque(maxlen=12) | Last 12 utilization snapshots for FL prediction |

### 6.3 Step Logic (per decision epoch)

```
step(action):
    1. Decode action: for each managed OD pair i, select path path_table[i][action[i]]
    2. Advance time_index by 1 → load current_tm from dataset
    3. Compute link_loads:
        - For each managed OD pair: route demand along selected path → add to link_loads
        - For each unmanaged OD pair: route demand along shortest path → add to link_loads
    4. Compute link_utils = link_loads / capacities (clipped to [0, 2] to allow overload)
    5. Update history_buffer with link_utils
    6. Run FL model on history_buffer → predicted_utils
    7. Compute delays and losses per link (formulas from §5.4)
    8. Compute reward (formula from §5.4)
    9. Build observation = concat(link_utils_clipped_0_1, predicted_utils_clipped_0_1, demands_normalized)
    10. Check done condition
    11. Return observation, reward, done, truncated, info
```

### 6.4 Fidelity Checks

After training, validate that the Gym env's outputs (delay, utilization, loss) match Mininet's within **±15%** on the same traffic scenario. If they diverge significantly:
- Calibrate the M/M/1 delay model parameters
- Adjust the packet loss ramp thresholds
- Add a link-load → queuing-delay lookup table fitted from Mininet measurements

---

## 7. SDN Emulation — Mininet + Ryu

### 7.1 Mininet Configuration

| Parameter | Value |
|---|---|
| Mininet version | 2.3.1 |
| Switch type | OVSKernelSwitch (Open vSwitch in kernel mode) |
| Controller | RemoteController (Ryu, port 6653) |
| Topology | Custom Python topo class matching §2.1 |
| Link bandwidth | 10 Mbps (TCLink with `bw=10`) |
| Link delay | 2–10 ms per link (TCLink with `delay='Xms'`) |
| Host IP scheme | `10.<src_node>.<dst_node>.1/24` or flat `10.0.0.X/8` |

### 7.2 Ryu Controller Application

Custom Ryu app with the following modules:

| Module | Responsibility |
|---|---|
| **TopologyDiscovery** | Discover switches and links via LLDP. Build adjacency graph. |
| **StatsCollector** | Poll port statistics every **5 seconds** via `OFPPortStatsRequest`. Compute per-link byte rates → utilization. |
| **FlowInstaller** | Given a path (list of switch DPIDs and output ports), install OpenFlow rules with: `match=(eth_type=0x0800, ipv4_src, ipv4_dst)`, `actions=[output:port]`, `priority=100`, `idle_timeout=60`. |
| **AIBridge** | Expose a REST API or ZMQ socket for the RL agent to: (1) GET current link stats, (2) POST routing decisions. |

### 7.3 Traffic Generation in Mininet

- Use **iperf3** in UDP mode from each source host to destination host.
- Traffic rate per OD pair is read from the current TM entry.
- A **TrafficOrchestrator** Python script runs in a loop:
  1. Every 30 seconds, read the next TM from the test dataset.
  2. Start/adjust iperf3 sessions to match new demands.
  3. Log actual throughput, delay (ping), and packet loss.

### 7.4 Validation Protocol

Run the **same 100-step test scenario** in both the Gym env and Mininet. Compare:
- Per-link utilization: should correlate with $R^2 > 0.85$
- End-to-end delay: absolute error < 15 ms
- Packet loss rate: directional agreement (both show loss / no loss on the same links)

---

## 8. System Integration & Real-Time Data Flow

### 8.1 Timescale Management

> [!IMPORTANT]
> FL and RL operate at different timescales. This is how they connect.

| Process | Timescale | Trigger |
|---|---|---|
| FL Training | **Offline** (before deployment) | Run once. Re-train periodically if traffic patterns shift (e.g., weekly). |
| FL Inference (prediction) | **Every 30 seconds** (at each decision epoch) | Triggered by RL agent before it acts. |
| RL Decision | **Every 30 seconds** | Collects state → gets prediction → selects action → sends to SDN Controller. |
| SDN Flow Rule Update | **On-demand** (~ms latency) | Triggered by RL decision. Only update rules for OD pairs whose path changed. |
| Stats Collection | **Every 5 seconds** | Ryu polls switches. The most recent poll before a decision epoch is used. |

### 8.2 Data Flow Diagram (Decision Epoch)

```
┌────────────────────────────────────────────────────────────────────┐
│                        Decision Epoch t                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. Ryu StatsCollector ──────► link_utils_t (42-dim vector)       │
│                                       │                            │
│  2. link_utils_t appended to ────► history_buffer (12 × 42)       │
│                                       │                            │
│  3. history_buffer ──► FL Model ──► predicted_utils_{t+1} (42-d)  │
│                                       │                            │
│  4. TrafficMonitor ──────────► demands_t (20-dim vector)           │
│              │                        │                            │
│              └───────────┬────────────┘                            │
│                          │                                         │
│  5. state_t = concat(link_utils_t, predicted_utils_{t+1}, demands)│
│                          │                                         │
│  6. state_t ──► PPO Agent ──► action_t (20-dim MultiDiscrete)     │
│                          │                                         │
│  7. Decode action_t ──► path selections for each OD pair          │
│                          │                                         │
│  8. For each OD pair whose path changed:                          │
│        FlowInstaller ──► install new OpenFlow rules                │
│                                                                    │
│  9. Log: state_t, action_t, reward_t, metrics                     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 8.3 Incremental Flow Updates

Do **not** reinstall all flow rules every 30 seconds. Only update rules for OD pairs where the selected path **changed** from the previous epoch. This minimizes control-plane overhead and avoids transient forwarding loops during rule installation.

**Rule installation order**: Install rules **from destination switch backward to source switch** (reverse path order) to prevent blackholing during transition.

---

## 9. Evaluation Framework

### 9.1 Traffic Prediction Metrics

| Metric | Formula | What It Measures |
|---|---|---|
| MAE | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | Average absolute prediction error |
| RMSE | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Root mean squared error (penalizes large errors) |
| MAPE | $\frac{100}{n}\sum_{i=1}^{n}\frac{\|y_i - \hat{y}_i\|}{\|y_i\| + \epsilon}$ | Percentage error (add $\epsilon=10^{-6}$ to avoid div-by-zero) |
| $R^2$ | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Explained variance (closer to 1 = better) |

**Report per-link and aggregated (mean over all links).**

### 9.2 Traffic Engineering Metrics

| Metric | Formula / Description | Target |
|---|---|---|
| **MLU** | $\max_{e \in E} u_e$ | Lower is better. < 0.70 is good. |
| **Average Link Utilization** | $\frac{1}{\|E\|}\sum_{e} u_e$ | Higher is better (efficient use). |
| **Jain's Fairness Index** | $J = \frac{(\sum_{e} u_e)^2}{\|E\| \cdot \sum_{e} u_e^2}$ | Closer to 1.0 = more balanced load. |
| **End-to-End Delay** | Average $D_i$ across OD pairs (ms) | Lower is better. |
| **Packet Loss Ratio** | Average $L_i$ across OD pairs (%) | Lower is better. 0% ideal. |
| **Throughput** | Total goodput delivered / total demand (%) | Closer to 100% = better. |
| **Convergence Time** | Number of RL training steps to reach 95% of final reward | Lower is better. |
| **Decision Latency** | Wall-clock time from state observation to action output (ms) | Must be < 1 second for real-time viability. |

### 9.3 Baselines

| # | Baseline | Description | What It Proves |
|---|---|---|---|
| B1 | **OSPF-SPF** | All traffic routed on shortest path (hop count). No ML. | Basic traditional routing benchmark. |
| B2 | **ECMP** | Traffic split equally across all equal-cost shortest paths. No ML. | Standard load-balancing benchmark. |
| B3 | **RL-Only (Reactive)** | PPO agent with state = $[\mathbf{u}_t \| \mathbf{d}_t]$ (no prediction). Same RL hyperparameters. | Proves the value of traffic prediction. |
| B4 | **Centralized LSTM + RL** | Same LSTM trained on all data centrally (no FL). Same RL. | Proves FL achieves comparable accuracy without sharing raw data. |

### 9.4 Test Scenarios

| # | Scenario | Description | What It Tests |
|---|---|---|---|
| T1 | **Normal Traffic** | Steady-state traffic following daily sinusoidal pattern. No anomalies. | Baseline performance under typical conditions. |
| T2 | **Traffic Spike** | At $t=30$, triple the demand for 3 specific OD pairs for 10 steps (5 min). | Proactive congestion avoidance (FL should predict the spike). |
| T3 | **Gradual Increase** | Linearly increase all demands by 50% over the episode (100 steps). | Adaptation to sustained load growth. |
| T4 | **Flash Crowd** | At $t=50$, 5 OD pairs suddenly go to 5× normal demand for 5 steps. | Stress test — can RL react even with imperfect prediction? |
| T5 | **Link Failure** | At $t=40$, remove link (4,5). All traffic must reroute. | Resilience (RL must reroute affected OD pairs). Note: FL model still predicts based on the original topology; RL must compensate. |

### 9.5 Statistical Rigor

- Run each test scenario **5 times** with different random seeds (for noise in TMs and RL stochasticity).
- Report **mean ± standard deviation** for all metrics.
- Use a **paired t-test** (or Wilcoxon signed-rank for non-normal distributions) to check if the proposed system's improvement over each baseline is statistically significant ($p < 0.05$).

---

## 10. Infrastructure & Azure Student Budget

### 10.1 Azure for Students

- **Credit**: $100 USD (no credit card required)
- **Validity**: 12 months
- **Key constraint**: No access to GPU VMs (NC/ND series require pay-as-you-go). **All work is CPU-only.**

### 10.2 Recommended VM

| Spec | Value |
|---|---|
| VM Size | **Standard_D4s_v3** |
| vCPUs | 4 |
| RAM | 16 GB |
| Temp storage | 32 GB SSD |
| Cost | ~$0.192 / hour |
| OS Image | Ubuntu 22.04 LTS |

**Budget math**:

| Task | Estimated Hours | Cost |
|---|---|---|
| FL Training (100 rounds) | 1–2 h | $0.38 |
| RL Training (2–5M steps) | 0.5–1 h | $0.19 |
| Mininet validation (5 scenarios × 5 seeds × 15 min) | ~6 h | $1.15 |
| Development, debugging, dashboard | ~40 h | $7.68 |
| **Total estimated** | **~50 h** | **~$9.60** |

> [!TIP]
> That's under 10% of the $100 budget. Plenty of room for re-runs, hyperparameter sweeps, and debugging. If more compute is needed, scale up to **Standard_D8s_v3** (8 vCPU, 32 GB, ~$0.384/h) for faster parallel Gym envs.

### 10.3 Software Stack

| Component | Package | Version |
|---|---|---|
| Python | python3 | 3.10+ |
| Deep Learning | pytorch | 2.x (CPU build) |
| FL Framework | flwr (Flower) | 1.x |
| RL Framework | stable-baselines3 | 2.x |
| Gym | gymnasium | 0.29+ |
| SDN Controller | ryu-manager | 4.34 |
| Network Emulator | mininet | 2.3.1 |
| Virtual Switch | openvswitch-switch | 2.17+ |
| Traffic Gen | iperf3 | 3.x |
| Graphs | networkx | 3.x |
| Data / Viz | numpy, pandas, matplotlib | latest |
| Dashboard | streamlit or grafana | latest |

### 10.4 Alternative: Local Machine

If Azure is inconvenient for interactive development, Mininet can run in a **VirtualBox/VMware Ubuntu VM** on a local machine (8 GB+ RAM recommended). Use Azure only for longer training runs.

---

## 11. Revised Timeline (15 weeks)

Adjustments from the original proposal to address integration complexity:

| Week | Phase | Tasks |
|---|---|---|
| 1 | Research | Survey FL+RL for TE papers. Finalize dataset choice. |
| 2 | Research | Deep-dive into PPO, Flower, Ryu APIs. Write related work section of thesis. |
| 3 | Data + Infra | Generate synthetic NSFNET TMs. Set up Azure VM. Install full software stack. |
| 4 | FL - Build | Implement LSTM model, FL Client/Server with Flower. |
| 5 | FL - Train | Train FL model. Evaluate prediction accuracy (MAE, RMSE, MAPE). Tune hyperparams. |
| 6 | RL Env | Build the custom Gymnasium environment (§6). Unit-test step logic. |
| 7 | RL - Build | Implement PPO agent with Stable-Baselines3. Define state/action/reward per this spec. |
| 8 | RL - Train | Train PPO agent in Gym env. Monitor reward curves. Tune hyperparameters. |
| 9 | Integration | Connect FL predictions into RL state. Re-train RL with FL in the loop. Run all baselines (B1–B4) in Gym env. |
| 10 | SDN - Build | Build Mininet topology + Ryu controller app (stats collector, flow installer, AI bridge). |
| 11 | SDN - Validate | Run trained FL+RL system inside Mininet. Fidelity check (§6.4). Debug discrepancies. |
| 12 | Evaluation | Execute all 5 test scenarios (§9.4) × 5 seeds. Collect all metrics. |
| 13 | Demo + Dashboard | Build Streamlit dashboard. Record demo video. |
| 14 | Thesis Writing | Write methodology, experiments, results chapters. Produce all tables and figures. |
| 15 | Finalize | Buffer week. Finalize thesis. Prepare defense slides. |

> [!CAUTION]
> **Week 9 (Integration) is the highest-risk week.** If FL predictions degrade RL performance (e.g., prediction errors mislead the agent), you may need to retrain RL with noise-augmented predictions. Budget extra time here.

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|---|---|
| $V$ | Set of nodes (routers), $\|V\| = 14$ |
| $E$ | Set of directed edges (links), $\|E\| = 42$ |
| $C_e$ | Capacity of link $e$ (bps) |
| $u_e(t)$ | Utilization of link $e$ at time $t$, $= \text{load}_e / C_e$ |
| $\hat{u}_e(t)$ | Predicted utilization of link $e$ at time $t$ |
| $TM_{ij}(t)$ | Traffic demand from node $i$ to node $j$ at time $t$ (bps) |
| $N_{OD}$ | Number of managed OD pairs (20) |
| $K$ | Number of candidate paths per OD pair (3) |
| $T$ | LSTM lookback window (12 timesteps = 60 min) |
| $H$ | LSTM prediction horizon (3 timesteps = 15 min) |
| $s_t$ | RL state at decision epoch $t$ |
| $a_t$ | RL action at decision epoch $t$ |
| $r_t$ | RL reward at decision epoch $t$ |
| $\pi_\theta(a \mid s)$ | PPO policy parameterized by $\theta$ |
| $V_\phi(s)$ | PPO value function parameterized by $\phi$ |
| MLU | Maximum Link Utilization |
| $D_i$ | End-to-end delay for OD pair $i$ |
| $L_i$ | Packet loss ratio for OD pair $i$ |
| $J$ | Jain's Fairness Index |

---

## Appendix B: Key Design Decisions & Rationale

| Decision | Chosen Option | Alternatives Considered | Rationale |
|---|---|---|---|
| RL Algorithm | PPO | DQN, SAC, DDPG | PPO handles MultiDiscrete actions natively, is stable, and has excellent SB3 support. DQN can't handle multi-dimensional discrete. SAC/DDPG are for continuous actions. |
| Action Space | Discrete path selection | Continuous splitting ratios | Simpler to implement, interpret, and map to SDN flow rules. Splitting ratios are a natural future extension. |
| FL Aggregation | FedAvg | FedAdam, FedProx | FedAvg is simplest and well-understood. Switch to FedAdam only if FedAvg shows instability with heterogeneous domains. |
| LSTM vs Transformer | LSTM | Transformer, GRU | LSTM is well-suited for time series with 12-step sequences. Transformer overhead is unjustified for this sequence length. GRU is viable but LSTM is more cited in the FL traffic prediction literature. |
| Training Env | Custom Gym | Mininet-in-the-loop, ns-3 | Mininet is 1000× too slow for RL training. ns-3 is even heavier. Custom Gym with M/M/1 delay model is fast and sufficiently faithful for path-selection decisions. |
| Link Capacity | 10 Mbps | 100 Mbps, 1 Gbps | 10 Mbps makes congestion observable with low iperf traffic on a single VM. Higher capacities need more aggressive traffic generation, stressing the VM's CPU. |
| K (paths) | 3 | 2, 4, 5 | K=3 balances diversity of routing options vs. action space size. K=2 is too restrictive; K≥4 increases action space exponentially ($K^{20}$). |
| N_OD (managed pairs) | 20 | All 182, 10, 50 | 20 is enough to demonstrate the system on the most impactful flows while keeping the action space tractable ($3^{20}$). |

---

## Appendix C: File & Directory Structure (Recommended)

```
project/
├── data/
│   ├── generate_tm.py          # Synthetic traffic matrix generator (§3.1)
│   ├── preprocess.py           # Windowing, normalization (§3.2)
│   └── nsfnet_topology.json    # Node/link definitions (§2.1)
├── fl/
│   ├── model.py                # LSTM architecture (§4.1)
│   ├── client.py               # Flower FL Client
│   ├── server.py               # Flower FL Server
│   └── train_fl.py             # FL training script
├── rl/
│   ├── env.py                  # Gymnasium NSFNETRoutingEnv (§6)
│   ├── train_rl.py             # PPO training script (§5.6)
│   ├── evaluate_rl.py          # Run baselines & test scenarios
│   └── callbacks.py            # Custom SB3 callbacks (logging, early stop)
├── sdn/
│   ├── topo.py                 # Mininet NSFNET topology (§7.1)
│   ├── ryu_app.py              # Ryu controller application (§7.2)
│   ├── traffic_gen.py          # iperf3 traffic orchestrator (§7.3)
│   └── validate.py             # Fidelity check: Gym vs Mininet (§6.4)
├── dashboard/
│   └── app.py                  # Streamlit dashboard (§7, demo)
├── models/                     # Saved model checkpoints
│   ├── fl_global_model.pt
│   └── ppo_agent.zip
├── results/                    # Experiment logs, CSVs, plots
├── thesis/                     # LaTeX source
└── requirements.txt
```
