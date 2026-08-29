# ADR 0005 — Grouped train / valid / test by `query_id`

- **Status:** Accepted
- **Date:** 2026-08-26
- **Updated:** 2026-08-29 — stratification adopted (D15); seed repeats added (D16)

## Context

The PDF suggests a train/valid/test split. Filters are constant within `query_id`; several rows share the same search. A **row-wise** random split would place sibling impressions on both sides of the split. The model could score a test item using patterns of the same query seen in training (especially `relative_price_position` and category).

## Decision

1. Split the **set of `query_id`s** (default 70% / 15% / 15%), **stratified on whether the query has at least one purchase** (D15).
2. Assign every row to the split of its query.
3. Fit preprocessors on train rows only.
4. Early stopping on **valid**; **test** once per arm for the report.
5. After the split, drop `query_id` from model inputs.

No k-fold. One grouped train / valid / test partition is the evaluation design; dispersion comes from **5 seed repeats** of that same protocol (D16).

## Consequences

- Valid/test metrics estimate performance on **new searches**.
- Without stratification, held-out prevalence varies 0.118–0.153 across seeds. Since PR-AUC's baseline *is* the prevalence, that made valid and test PR-AUC non-comparable — hence stratifying (D15), which closes the gap this ADR previously left open.
- A single split cannot resolve PR-AUC differences below ~0.05, which is why seed repeats and reported dispersion are mandatory rather than optional.

## Alternatives

- Row-wise stratified split: rejected (query leakage).
- Time-based split: timestamps are per impression, not a single query time; ordering within a query is not a clean global timeline for production SERPs.
