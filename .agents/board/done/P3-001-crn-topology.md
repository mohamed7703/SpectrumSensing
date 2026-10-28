---
id: P3-001
title: "Implement CRN Simulation Architecture, Topology & Traffic Generator"
phase: 3 - Network Simulation
status: done
priority: high
depends_on: [P2-004]
tags: [simulation, ns3, topology, traffic, channel-model]
---

## Objective
Implement discrete-event simulation models for Cognitive Radio Networks, modeling Primary Users (PUs)
and Secondary Users (SUs), Point-to-Point/Wireless links, deterministic & Poisson traffic generators,
and propagation channel attenuation.

## Acceptance Criteria
- [x] Node and network topology builder in `/src/ns3_simulation/topology_builder.py`
- [x] Configurable traffic generators (UDP Echo, Poisson, bursty traffic) in `/src/ns3_simulation/traffic_generator.py`
- [x] Wireless propagation loss and path fading model in `/src/ns3_simulation/channel_model.py`
- [x] Network configuration file in `/configs/ns3_config.yaml`
- [x] Unit tests in `/tests/test_topology.py`
