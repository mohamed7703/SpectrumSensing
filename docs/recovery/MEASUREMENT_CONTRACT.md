# Measurement contract, version 1

- Labels are nonempty real vectors containing exactly 0 or 1; integer casting occurs only after validation. Scores must be finite real values. Single-column arrays are accepted; matrices, nonfinite data and mismatched lengths are rejected.
- Confusion counts allow a single observed class. Pd and miss rate are NaN without H1; Pfa is NaN without H0. Accuracy remains defined for nonempty observations. Precision/F1 use zero when their own count denominator is zero, a convention rather than evidence of absent-class performance.
- ROC requires both classes. Each distinct descending score contributes one operating point after its entire tie block is accepted. The origin is associated with threshold +infinity. Raw scores are not limited to [0,1]. Returned arrays preserve threshold traversal, including repeated FPRs.
- AUC uses trapezoidal integration of that exact empirical curve and agrees with pairwise positive/negative ranking with half-credit for ties. Runtime is O(N log N), without the old O(N_positive*N_negative) comparison matrix.
- Legacy `num_thresholds` arguments are retained and validated, but do not subsample scientific operating points or alter AUC. Plotting may downsample separately.
- EER is the linearly interpolated FPR=FNR crossing on the exact empirical curve. It can describe a randomized threshold mixture rather than one attainable fixed threshold. Pd at a predeclared Pfa remains the sensing endpoint.
- Training metrics additionally require probability scores in [0,1] for BCE. Single-class training/evaluation batches return NaN AUC rather than invented 0.5.
- `binomial_rate_interval` returns exact Clopper-Pearson bounds, including one-sided alternatives. It assumes independent Bernoulli trials; it does not account for correlated frames, shared captures or model-training variation.
- Software-test success and scientific acceptance are separate fields. Correct metric calculations do not establish correct data generation, real neural computation, realistic interference, hardware performance or publication readiness.
