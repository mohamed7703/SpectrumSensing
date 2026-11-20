---
name: manuscript-evidence
description: Assemble evidence for the spectrum sensing manuscript, prepare PaperOrchestra inputs, and verify draft claims, figures, and citations against research records.
---

# Manuscript evidence

1. Start from the [claim ledger](../../../docs/recovery/CLAIM_LEDGER.md), [measurement contract](../../../docs/recovery/MEASUREMENT_CONTRACT.md), and the run that supports the claim. For each result, record the draft claim, run identifier and configuration, model and data identifiers, test conditions, counts, uncertainty, and source artifact. Identify the evidence setting, such as synthetic replay, mock receiver, or independent USRP capture.
2. Check manuscript numbers against machine-readable records or recalculate from saved scores. Trace each figure to its data and plotting script. For detection at a specified false-alarm target, report how the threshold was set and whether held-out testing met the criterion. Keep planned experiments in the protocol until results are available.
3. Verify citation metadata and read the part of each paper that supports the claim. For comparisons, check the cited paper's method, test setting, and result. Mark comparisons that depend on inaccessible full text.
4. For PaperOrchestra, assemble the research question, candidate contribution, prior-work comparison, methods, verified experiment summaries, figures or plot inputs, and known gaps in the authorized materials folder. The [authors' PaperOrchestra repository](https://github.com/google-research/paper-orchestra/blob/d9bae997e793a015ed38bb1605be9eb58c00284f/README.md) documents the raw-materials, LaTeX-template, and output directory arguments. If asked to run the generator, use the approved local installation and confirm which materials may be shared with any configured external model.
5. Audit the generated draft against the claim-to-evidence mapping. Resolve unsupported claims, missing figures, incorrect citations, and scope changes before treating the draft as ready for author review.

The claim-to-evidence mapping and draft audit draw on [ARIS Paper Writing](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/ca70fe2c2f87e8c8e70729acad4f472ef19472d1/skills/skills-codex/paper-writing/SKILL.md).
