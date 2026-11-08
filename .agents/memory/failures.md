# Failures & Dead-End Log
> Consult this file before attempting any approach. Avoid repeating past mistakes.

## Format
```
- **[Date]**: Brief description of what was attempted
  - *Failure Reason:* Why it didn't work
  - *Resolution/Rule:* What to do instead
```

## Log
- **2026-11-05, R002/R003:** 16 foundation, 4 initial physical, 10 initial network, 7 data, 1 native graph, 2 export-wrapper and 3 one-sided quantization regressions failed before their fixes. Preserve transcripts in the sibling audit folder; new measurements live under results/recovery.
- **2026-11-05, environment:** Python 3.10 cannot install Keras 3.13.2; use compatible 3.12.4 with TF 2.20.0 for new runs. Preserve compatibility failures of historical models; never silently remove non-null quantization metadata. Network timeouts resolved with longer pip timeouts and trusted Windows CA roots, not disabled TLS.
- **2026-11-05, export:** TF/Keras from_keras_model attempted a missing private _get_save_spec. Use public ConcreteFunction tracing of the existing trained graph and test actual interpreter predictions. Failed first R003 attempt remains recorded. PowerShell 2>&1 can mark informational native stderr as NativeCommandError; capture subprocess return codes explicitly for future logged runs.
- **2026-11-05, checkpoint test:** Independently resumed training=True calls draw different dropout masks. Verify saved optimizer variables and deterministic-gradient updates; do not claim bitwise RNG continuation or widen numeric tolerances to hide different random computations.
- **2026-11-05, scientific audit:** 205 original tests passed despite exact ROC counterexamples, shared feature-head inference under neural model names, placeholder exports, mock miscalibration and incomplete event accounting. Passing tests do not establish the scientific claims. Preserve audit fixtures; add regressions before fixes; track claims separately.
- **2026-11-05, R001:** audit regressions failed 32/41 before correction; fixed by exact score traversal and validation rather than widening tolerances or changing expected scientific results.
