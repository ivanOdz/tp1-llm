# ADR 0008 — Delivery is a Colab notebook (package deferred)

- **Status:** Accepted
- **Date:** 2026-08-26

## Context

The repo README already sketched a `src/` layout (parquet, experiment folders). The assignment execution environment in the latest brief is **Google Colab**, with a **single sequential notebook**.

## Decision

**v1 implementation** (when requested): one Jupyter notebook following [NOTEBOOK_SPEC.md](../docs/NOTEBOOK_SPEC.md). Design lives in `docs/` + `adr/`. The modular package and parquet pipeline stay **out of scope** until explicitly requested.

## Consequences

- Graders can run Colab top-to-bottom.
- Some duplication of helper functions in cells is acceptable; keep helpers typed and sectioned.
- The old README tree is historical intent, not the current build plan.

## Alternatives

- Library + thin notebook: cleaner engineering, worse one-click Colab story unless we also vendor the package.
