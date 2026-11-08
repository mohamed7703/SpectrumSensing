# R006: Refresh the canonical repository documentation

Date: 2026-11-08. The user authorized replacing/changing old repository files after the recovery branch was pushed.

Replaced the root README with the current evidence status, scientific limitations, offline presentation link, pinned installation and supported experiment commands. Preserved the previous README under the historical audit. Added a local-run ignore rule to keep new generated runs distinct from reviewed evidence.

Promotion uses a normal fast-forward from the reviewed recovery branch to main when ancestry allows it, preserving Git history and avoiding a force push. The Python implementation is unchanged from the 325-case pre-push validation. Documentation links and command-line entry points are checked separately.
