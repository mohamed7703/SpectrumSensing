---
id: P1-003
title: "Implement Detection Metrics (Pd, Pfa) and ROC Curves"
phase: 1 - Math Foundations
status: done
priority: critical
depends_on: [P1-002]
tags: [math, metrics, roc, core]
---

## Objective
Implement Pd and Pfa calculations with ROC curve generation.

## Acceptance Criteria
- [x] Pd = P(r(n) ≥ γ | H1) — computed analytically and empirically
- [x] Pfa = P(Ĥ1 = H1 | H0) — computed analytically and empirically
- [x] ROC curve: Pd vs Pfa sweep across threshold γ
- [x] Compare against known theoretical energy detection ROC
- [x] Output: `/src/math_models/detection_metrics.py`
