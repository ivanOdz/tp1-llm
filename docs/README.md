# Design documentation (no code yet)

This folder captures **implementation decisions** for TP1 (73.69 LLM, 2026) before any notebook or source is written.

| Doc | Purpose |
| --- | --- |
| [ASSIGNMENT_MAPPING.md](ASSIGNMENT_MAPPING.md) | PDF exercises (all three) → artifacts, grading focus, delivery checklist |
| [EDA.md](EDA.md) | Measured facts from `supermarket_products.csv` (10k rows); filter degeneracy, **the behavioural cue in text**, metric noise floor |
| [DESIGN.md](DESIGN.md) | End-to-end technical design (features, model, train, eval, baselines, limitations) |
| [NOTEBOOK_SPEC.md](NOTEBOOK_SPEC.md) | Sequential notebook outline (cells, no code) |
| [OPEN_DECISIONS.md](OPEN_DECISIONS.md) | D1–D24: what is resolved, what still needs a ruling, and the locked list |

Architecture Decision Records live in [`adr/`](../adr/). Read those when you want the **why**, not the full pipeline. Start with [adr/0009](../adr/0009-text-behavioural-cue.md) — it is the decision that reshaped the experimental design.

**Current phase:** documentation only, post-audit. The design has been reconciled against the CSV and the PDF; open items are D24 (Exercise 3 owner) and the detail-level defaults in D2, D8, D9, D13. Do not generate `.ipynb` until a later prompt asks for a plan, then code.
