# ADR 0008 — Delivery is a Jupyter notebook (package deferred)

- **Status:** Accepted
- **Date:** 2026-08-26

## Context

An earlier revision of the repo sketched a `src/` layout with a parquet pipeline and `experiments/` folders. Delivery for the practical part is instead a **single sequential Jupyter notebook**, run locally (GPU used if available, CPU adequate at this scale).

The PDF does not prescribe the artifact form — §4 requires only a repository with `README.md`, the commit hash, and the presentation. The notebook is our choice.

## Decision

**v1 implementation** (when requested): one Jupyter notebook following [NOTEBOOK_SPEC.md](../docs/NOTEBOOK_SPEC.md). Design lives in `docs/` + `adr/`. The modular package and parquet pipeline stay **out of scope** until explicitly requested.

## Consequences

- Graders (and teammates) can run the notebook top-to-bottom.
- Some duplication of helper functions in cells is acceptable; keep helpers typed and sectioned.
- Dependencies are pinned in [`requirements.txt`](../requirements.txt) rather than installed from inside the notebook (D21).
- The zero-byte `src/`, `notebooks/0*.ipynb` and `experiments/final_cv/` placeholders were deleted — the last of these encoded a k-fold design that D16 rules out.

## Alternatives

- Library + thin notebook: cleaner engineering, more setup for a one-shot run unless the package is vendored or installed.
