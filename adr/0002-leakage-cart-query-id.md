# ADR 0002 — Drop `cart` and `query_id` as features

- **Status:** Accepted
- **Date:** 2026-08-26

## Context

The PDF lists `cart` and `query_id` as dataset columns but gives no instruction about either, so excluding them is **our decision, derived from the data**, not a stated requirement. The CSV makes the case directly: **every purchase has `cart=True`**, and **no purchase has `cart=False`**, i.e. \(P(bought \mid cart=\text{false}) = 0\). `query_id` is a high-cardinality search key (2 012 values).

## Decision

- **`cart`:** never an input. It is a downstream action, not an attribute of the impression at render time. Using it would predict “buy given already in cart,” which is not BTR.
- **`query_id`:** never an input. It would let the model memorize searches. A copy is kept **only** to build grouped splits, then discarded.

## Consequences

- Honest impression-level BTR model.
- Split code depends on `query_id` internally ([adr/0005](0005-grouped-splits.md)).
- The same render-time test is what flags the behavioural cue in `title`/`description` as a target-derived proxy ([adr/0009](0009-text-behavioural-cue.md)). That cue passes the render-time test where `cart` fails it, which is why it is kept and ablated rather than dropped.

## Alternatives

- Keep `cart` as a “funnel” feature: rejected (leakage).
- Hash `query_id` into an embedding: still leaks identity of the search and does not generalize to new queries.
