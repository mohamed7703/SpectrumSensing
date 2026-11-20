---
name: network-simulation
description: Design and verify cognitive radio network experiments, Python discrete-event simulations, packet overlap and delay, trace analysis, and reproducible ns-3 comparisons when requested.
---

# Network simulation

- Start with the Python discrete-event implementation in `src/ns3_simulation/` for local protocol and timing checks. Record the simulator and implementation version in each run. An ns-3 experiment requires a pinned ns-3 version, executable scenario, configuration, and generated traces.
- Compare Pure ALOHA, Slotted ALOHA, and ILA-CSMA on matched offered traffic, packet sizes, primary-user activity, observation horizon, and channel conditions. Configure seeds and log run identifiers and RNG streams. Use matched inputs within each replication and independent run identifiers across replications. In ns-3, document both seed and run number.
- Verify primary-user activity that starts and ends mid-packet, symmetric secondary-user overlap, and exact boundary contact. Resolve delivery at packet completion. Count pending packets at the observation cutoff and preserve them when the horizon extends.
- Preserve time order when replaying sensing decisions into a MAC. Track the sensing-to-access delay, false alarms, and duration of missed detections. Treat the current overlap rule as a simplified interference proxy. Receiver protection claims require a validated receiver model or measurements.
- Reconcile send, delivery, drop, and pending counts. Report delivered-only delay alongside delivery, loss and pending rates, offered load, and interference exposure. Give results across independent runs with uncertainty. Report any energy result with its measurement or model boundary.

Use [network YAML](../../../experiments/configs/ns3_config.yaml), write new run metadata to [experiment logs](../../../experiments/logs/), and preserve traces with their result records. Run the existing MAC, trace parser, topology, and network recovery tests for timing and accounting changes.

References: [ns-3 repository guidance](https://github.com/nsnam/ns-3-dev-git/blob/c4f3a4a2897900e63d1b7c5f1953f099b8b1d17a/AGENTS.md) and [ns-3 random-variable manual](https://github.com/nsnam/ns-3-dev-git/blob/c4f3a4a2897900e63d1b7c5f1953f099b8b1d17a/doc/manual/source/random-variables.rst).
