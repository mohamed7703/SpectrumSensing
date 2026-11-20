---
name: calibration-novelty-review
description: Review prior work and design pre-USRP novelty checks for INT8 spectrum detector calibration under receiver gain changes. Use for literature searches, novelty comparisons, threshold feasibility, and calibration-budget experiments.
---

# Calibration novelty review

## Compare with prior work

1. State the proposed rule, competing method, and candidate contribution. For this study, ask whether gain-aware INT8 calibration uses fewer confirmed noise-only frames or improves weak-signal detection at a fixed false-alarm target.
2. Search communications venues and a complementary index or preprint server. Record the query, source, date, and filters. Include quantized spectrum sensing, constant false-alarm rate, gain or noise uncertainty, score ties, and sample-efficient threshold calibration. Follow references and later papers citing the closest work. Group preprints and published versions of the same study.
3. Start with the [INT8 edge sensing study](https://doi.org/10.1109/LCOMM.2025.3582680) and the [impairment-aware benchmark](https://doi.org/10.1007/s13369-026-11704-1). Read available full text and record the section, figure, or table behind each comparison. Mark details unresolved when access or reporting is incomplete.
4. Use one row per study and record scoring precision, calibration rule and noise sample count, gain and noise changes, capture/session splits, false-alarm target, weak-signal detection, hardware, and source location. Distinguish prior work on the method from prior work on the experimental finding. State the search limits alongside the candidate gap.

## Check feasibility before USRP

- Sweep distinct scores in the [frozen synthetic replay](../../../results/recovery/R005-runtime-stress/), including the all-detect and no-detect endpoints. Keep tied INT8 scores together and use the archived comparator when replaying its counts. Use the sweep as a diagnostic and evaluate the selected rule on fresh data.
- Define the false-alarm criterion and acceptable detection loss before comparing methods. Use disjoint groups for training, model selection, calibration, and testing. Count all noise-only observations used to fit each method, including previously collected per-gain samples. Compare with a fixed nominal INT8 threshold, independent INT8 recalibration, and a simple gain-aware energy detector on the same held-out trials.
- Repeat across model seeds and independent sessions. Vary gain and noise separately and test gain settings held out from calibration. Report counts, paired differences, and uncertainty at the independent session level where possible. Use the [measurement contract](../../../docs/recovery/MEASUREMENT_CONTRACT.md) for metric definitions.
- Report the closest overlap and remaining difference in one sentence, followed by the smallest experiment that could challenge it. If no deterministic threshold attains both targets, examine the detector's score resolution and separation. If ordinary recalibration matches the proposed rule, reconsider the method claim.

The claim-by-claim comparison and documented search draw on [ARIS Novelty Check](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/ca70fe2c2f87e8c8e70729acad4f472ef19472d1/skills/skills-codex/novelty-check/SKILL.md) and [K-Dense Literature Review](https://github.com/K-Dense-AI/scientific-agent-skills/blob/ec59a29a03d5c40f73ab40878df04860bc00e876/skills/literature-review/SKILL.md).
