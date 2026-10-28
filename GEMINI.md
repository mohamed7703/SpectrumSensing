# Project Rules — Cognitive Radio Spectrum Sensing

## Session Start Protocol
1. Read `.agents/memory/state.md` — know the current phase and active milestone
2. Read `.agents/memory/failures.md` — avoid repeating dead ends
3. Check `.agents/board/in-progress/` — resume any active tasks
4. Check `.agents/board/backlog/` — pick the next unblocked task

## Code Rules
- All Python files must include module-level docstrings describing their purpose
- Use type hints on all function signatures
- Never hardcode hyperparameters — use `/configs/*.yaml`
- Every new module needs tests in `/tests/`
- Mathematical formulas must be documented with LaTeX in docstrings

## Phase Transition Rules
- A phase is COMPLETE only when all its board tasks are in `done/`
- Do not start a phase's implementation before its predecessor is complete
- Exception: research and design work for future phases is always allowed

## Experiment Rules
- Every training run or significant simulation must be logged in `results/logs/`
- Use the experiment template from `results/logs/TEMPLATE.md`
- Save plots to `results/plots/` with descriptive filenames

## State Update Rules
- Update `.agents/memory/state.md` after every milestone completion
- Move task cards from `backlog/` → `in-progress/` → `done/` as work progresses
- Log any architectural decision in `.agents/memory/decisions.md`
