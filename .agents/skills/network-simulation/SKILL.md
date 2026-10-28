---
name: network-simulation
description: >-
  NS-3 network simulation for Cognitive Radio Networks: CRN topology setup,
  MAC protocol optimization (Pure ALOHA, Slotted ALOHA, ILA-CSMA), end-to-end
  delay measurement, trace parsing, collision probability analysis, and Nash
  equilibrium evaluation. Use when building or running network simulations.
---

# Network Simulation Skill

## Scope
Covers NS-3 discrete-event simulation of the Cognitive Radio Network, including
MAC layer optimization and delay analytics before physical SDR deployment.

## Workflow

### Step 1: NS-3 Environment Setup
- Install NS-3 (version 3.38+ recommended)
- Configure CRN module or custom spectrum sensing module
- Set up point-to-point links with predefined channel delays and data rates

### Step 2: CRN Topology Builder
Implement in `/src/ns3_simulation/`:
- `topology_builder.py` — Define PU/SU node positions, mobility models
- `traffic_generator.py` — UDP Echo client/server, deterministic loads
  - 1000 packets, 1024 bytes each, 0.01s intervals
- `channel_model.py` — Propagation loss, fading integration

### Step 3: MAC Protocol Implementation
- `mac_pure_aloha.py` — Baseline random access
- `mac_slotted_aloha.py` — Synchronized slot access
- `mac_ila_csma.py` — ILA-CSMA hybrid sensing:
  - Stage 1: Channel Activity Detection (CAD)
  - Stage 2: RSSI comparison against AI-driven threshold
  - Interference screening (CSMA-HS) + Time-on-Air spacing (CSMA-AB)

### Step 4: End-to-End Delay Measurement
- Enable NS-3 ASCII tracing → `.tr` event log
- Implement trace parser (`trace_parser.py`):
  ```python
  send_times = {}
  recv_times = {}
  delays = []
  # Parse .tr file for send/receive events
  # Calculate per-packet delay
  average_delay = sum(delays) / len(delays)
  ```

### Step 5: Performance Metrics
- Packet Delivery Ratio (PDR)
- Collision probability vs offered load
- Energy consumed per successful packet
- Compare: ILA-CSMA vs CSMA/CA in 2000-node scenario
  - Target: +20% PDR improvement
  - Delay reduction: 18.5s → 8.2s

### Step 6: Nash Equilibrium Analysis
- Model SU spectrum sharing as a game
- Verify stable equilibrium where SUs don't encroach on PU rights

## Output Files
- `/src/ns3_simulation/` — All simulation code
- `/configs/ns3_config.yaml` — Simulation parameters
- `/results/logs/` — Trace files and parsed results
- `/results/plots/` — Delay distributions, PDR curves, collision graphs

## Verification
- End-to-end delay must be calculable from trace files
- ILA-CSMA must show measurable improvement over pure ALOHA
- Results reproducible across simulation runs with same seed
