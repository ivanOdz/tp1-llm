# ADR 0001 — Target is impression-level `bought`

- **Status:** Accepted
- **Date:** 2026-08-26

## Context

The PDF defines BTR as purchases divided by impressions on the search-results page, and lists `bought` as the purchase flag. Evaluation is PR-AUC and ROC-AUC without a threshold.

## Decision

Model **binary classification** of `bought` at **impression (row) level**. The network outputs one logit per row. Predicted probabilities are interpreted as that product’s contribution to BTR under the observed query context.

## Consequences

- Standard `BCEWithLogitsLoss` and sklearn AUCs apply directly.
- We do not aggregate to query-level rates for the loss.
- We do not optimize a listwise ranking loss (possible extension, not in scope).

## Alternatives

- Query-level regression of empirical BTR: few queries, discards which item was bought.
- Multiclass / cart-then-buy funnel: changes the metric the PDF asked for.
