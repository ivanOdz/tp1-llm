# Design documentation (no code yet)

This folder captures **implementation decisions** for TP1 (73.69 LLM, 2026) before any notebook or source is written.

| Doc | Purpose |
| --- | --- |
| [ASSIGNMENT_MAPPING.md](ASSIGNMENT_MAPPING.md) | PDF exercises → artifacts and grading focus |
| [DATA_PROFILE.md](DATA_PROFILE.md) | Measured facts from `supermarket_products.csv` (10k rows) |
| [DESIGN.md](DESIGN.md) | End-to-end technical design (features, model, train, eval) |
| [NOTEBOOK_SPEC.md](NOTEBOOK_SPEC.md) | Sequential Colab notebook outline (cells, no code) |
| [OPEN_DECISIONS.md](OPEN_DECISIONS.md) | Items to change before implementation |

Architecture Decision Records live in [`adr/`](../adr/). Read those when you want the **why**, not the full pipeline.

**Current phase:** documentation only. Do not generate `.ipynb` or `src/` until the design is reviewed and a later prompt asks for a plan, then code.
