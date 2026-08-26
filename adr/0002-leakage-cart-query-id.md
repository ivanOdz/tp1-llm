# ADR 0002 — Drop `cart` and `query_id` as features

- **Status:** Accepted
- **Date:** 2026-08-26

## Context

The assignment prompt asks to drop `cart` and `query_id` to avoid leakage and overfitting. The CSV confirms a stronger fact: **every purchase has `cart=True`**, and **no purchase has `cart=False`**. `query_id` is a high-cardinality search key (2 012 values).

## Decision

- **`cart`:** never an input. It is a downstream action, not an attribute of the impression at render time. Using it would predict “buy given already in cart,” which is not BTR.
- **`query_id`:** never an input. It would let the model memorize searches. A copy is kept **only** to build grouped splits, then discarded.

## Consequences

- Honest impression-level BTR model.
- Split code depends on `query_id` internally ([adr/0005](0005-grouped-splits.md)).

## Alternatives

- Keep `cart` as a “funnel” feature: rejected (leakage).
- Hash `query_id` into an embedding: still leaks identity of the search and does not generalize to new queries.
