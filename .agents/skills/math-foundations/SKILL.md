---
name: math-foundations
description: >-
  Mathematical modeling for spectrum sensing: binary hypothesis testing (H0/H1),
  energy detection statistics, Pd/Pfa derivations, ROC curve generation,
  threshold optimization, noise models (AWGN), and channel fading PDFs
  (Rayleigh, Rician, Nakagami-m). Use when implementing or verifying the
  theoretical foundation of the cognitive radio system.
---

# Mathematical Foundations Skill

## Scope
This skill covers all mathematical modeling required before any ML or hardware work.
It implements the equations from the paper (Abdelbaset et al., Sensors 2024) and the
deployment plan's theoretical framework section.

## Workflow

### Step 1: Binary Hypothesis Testing Module
Implement the core H0/H1 signal model:
- `H0: r(n) = w(n)` — spectrum idle (noise only)
- `H1: r(n) = h·s(n) + w(n)` — PU transmitting + noise
- Generate synthetic signals under both hypotheses
- Validate with known noise variance σ²_n and signal variance σ²_s

### Step 2: Energy Detection Decision Statistic
Implement test statistic D(Y) = accumulated energy over L samples:
- Under H0: central chi-square distribution with 2L DoF
- Under H1: non-central chi-square distribution
- Threshold γ comparison logic

### Step 3: Detection Metrics (Pd and Pfa)
Implement probability calculations:
- `Pd = P(r(n) ≥ γ | H1)` — correct detection probability
- `Pfa = P(Ĥ1 = H1 | H0)` — false alarm probability
- Operational ratios: Pd = correct_detections / total_PU_transmissions
- Pfa = false_alarms / total_idle_periods

### Step 4: ROC Curve Generation
- Sweep threshold γ across range
- Plot Pd vs Pfa for various SNR levels
- Compare against paper's benchmark results

### Step 5: Channel Fading Models
Implement fading PDFs:
- **Rayleigh**: `f(a) = (a/σ²) · exp(-a²/2σ²)` — no line-of-sight
- **Rician**: Includes Bessel function I₀ for dominant path
- **Nakagami-m**: Generalized fading model
- Validate: At SNR=0dB, accuracy drops below 70% under Rayleigh/Rician
- At SNR≥8dB, accuracy exceeds 95%

### Step 6: Noise Uncertainty and SNR Wall
- Implement the SNR wall phenomenon (-12 to -8 dB)
- Show where energy detection fails
- Motivate the transition to ML-based sensing

## Output Files
All implementations go to `/src/math_models/`:
- `hypothesis_testing.py` — H0/H1 signal generation and testing
- `energy_detection.py` — Test statistic and threshold logic
- `detection_metrics.py` — Pd, Pfa, ROC computation
- `channel_fading.py` — Rayleigh, Rician, Nakagami-m PDFs
- `noise_models.py` — AWGN generation, SNR wall analysis
- `__init__.py` — Module exports

## Verification
- Each module has tests in `/tests/test_math_models.py`
- ROC curves must match known theoretical values for energy detection
- Pd at SNR=10dB with AWGN must exceed 0.95 for reasonable threshold
