# INT8 receiver calibration study

**Research question.** Can a gain-aware threshold reduce the number of confirmed noise-only frames needed for calibration, or improve detection at a specified false-alarm limit, compared with ordinary independent INT8 recalibration?

## Draft pre-USRP protocol

The first pre-USRP study is complete. The [decision brief](SUPERVISOR_BRIEF.md) summarizes the frozen-score sweep, fresh benchmark, closest prior work, and [NI USRP-2920 handoff](usrp-2920/README.md). The [frozen primary analysis](FROZEN_PROTOCOL.md) records the choices used for this run.

1. Complete the [prior-work comparison](../../literature/novelty-comparison.md), focusing on quantization, gain transfer, calibration budgets, and independent receiver sessions.
2. Sweep every attainable deterministic threshold in the [frozen synthetic pilot](../../results/recovery/R005-runtime-stress/). For each gain and SNR, record false alarms and detections as counts. Use this sweep to diagnose score resolution and separation.
3. Before evaluating a proposed rule on fresh data, specify the false-alarm criterion, weak-signal SNRs, acceptable detection loss, total calibration budgets, and held-out gain settings. A 5% false-alarm target is a starting proposal. Specify how sampling uncertainty and capture-session dependence will enter the decision.
4. Compare the fixed nominal INT8 threshold, ordinary independent recalibration, the proposed rule, and a gain-aware energy detector on matched test conditions. Record model and split identifiers and charge each method for all noise-only calibration samples it uses.
5. Use the pre-USRP result to decide what the hardware pilot needs to resolve. For a cross-day transfer test, collect known noise-only and signal-present sessions on different days and evaluate the locked threshold on the independent session.

See the [publication brief](PUBLICATION_BRIEF.md) for context, the [interactive replay](../presentations/receiver-calibration-workflow/Receiver_Calibration_Workflow.html) for the pilot data, and the [measurement contract](../recovery/MEASUREMENT_CONTRACT.md) for metric definitions.
