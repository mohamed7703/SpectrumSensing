---
id: P1-002
title: "Implement Energy Detection Decision Statistic"
phase: 1 - Math Foundations
status: done
priority: critical
depends_on: [P1-001]
tags: [math, energy-detection, core]
---

## Objective
Implement the energy detection test statistic D(Y) with threshold comparison logic.

## Acceptance Criteria
- [x] Compute accumulated energy over L samples
- [x] Chi-square distribution under H0 (central, 2L DoF)
- [x] Non-central chi-square under H1
- [x] Threshold γ comparison returning H0/H1 decision
- [x] Output: `/src/math_models/energy_detection.py`
