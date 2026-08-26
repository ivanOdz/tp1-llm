# ADR 0005 — Grouped train / valid / test by `query_id`

- **Status:** Accepted
- **Date:** 2026-08-26

## Context

The PDF suggests a train/valid/test split. Filters are constant within `query_id`; several rows share the same search. A **row-wise** random split would place sibling impressions on both sides of the split. The model could score a test item using patterns of the same query seen in training (especially filter-derived distances and category).

## Decision

1. Split the **set of `query_id`s** (default 70% / 15% / 15%, seed 42).
2. Assign every row to the split of its query.
3. Fit preprocessors on train rows only.
4. Early stopping on **valid**; **test** once for the report.
5. After the split, drop `query_id` from model inputs.

No k-fold. One grouped train / valid / test partition is the evaluation design.

## Consequences

- Valid/test metrics estimate performance on **new searches**.
- Class balance can differ slightly across splits (mitigation: D15 stratification).

## Alternatives

- Row-wise stratified split: rejected (query leakage).
- Time-based split: timestamps are per impression, not a single query time; ordering within a query is not a clean global timeline for production SERPs.
