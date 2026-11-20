---
name: math-foundations
description: Derive and verify spectrum sensing detector laws, SNR and fading conventions, tied-score ROC operating points, false-alarm thresholds, and uncertainty in detection rates.
---

# Mathematical foundations

- Define H0 as noise only and H1 as signal plus noise. State whether noise variance is total complex power, how the detector aggregates samples, and where SNR is measured. For a flat channel, received signal power includes the factor `|h|²`.
- With independent circular complex Gaussian noise, scaled accumulated energy under H0 has a central chi-square law with twice the complex sample count as degrees of freedom. Under H1, a fixed-energy waveform produces a noncentral chi-square law, while an independent circular Gaussian signal changes the scale of the central law. State conditioning on the channel and verify fading and detector formulas under their assumptions.
- For unit nominal noise, actual noise in `[1/rho, rho]`, and fixed received signal power `S`, robust mean separation requires `S > rho - 1/rho`. At a symmetric 1 dB bound this boundary is approximately −3.33 dB. Use the [recovery report](../../../docs/recovery/RECOVERY_PROGRESS.md) for the assumptions and conservative sample bound. A mean-separation boundary establishes a property of this energy model.
- Compute Pd from detections over H1 and Pfa from false alarms over H0. Use the existing exact ROC that admits one tied score block at a time. Include both endpoints and record the deployed comparator. When INT8 ties skip the target operating point, report the attainable deterministic points. If a class is absent, report its conditional rate as undefined.
- Use an independent H0 calibration set and keep the final test sealed. Exact binomial intervals apply to independent Bernoulli trials. For adjacent or overlapping frames from shared sessions, report independent session counts and account for grouping in comparisons. If evidence is limited to too few independent sessions, report that limitation.

Use the [measurement contract](../../../docs/recovery/MEASUREMENT_CONTRACT.md) and existing metric, calibration, and foundation regression tests.

References: [K-Dense Statistical Analysis skill](https://github.com/K-Dense-AI/scientific-agent-skills/blob/ec59a29a03d5c40f73ab40878df04860bc00e876/skills/statistical-analysis/SKILL.md) and [SciPy binomial intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
