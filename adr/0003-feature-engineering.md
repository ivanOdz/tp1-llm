# ADR 0003 — Interaction features instead of raw filters

- **Status:** Accepted (engineering). Match-flag **retention** is open (D1).
- **Date:** 2026-08-26

## Context

Raw filters (`filter_category`, `filter_storage_type`, `filter_price_*`) describe the query, not the item. Feeding them as-is either duplicates item fields or encourages the model to relearn “item vs filter” implicitly. The requested design is **explicit interactions**.

On this dataset, category and storage **always** match the filters, and price **always** lies inside the filter band ([DATA_PROFILE.md](../docs/DATA_PROFILE.md)).

## Decision

1. Build `is_category_match`, `is_storage_match`, `price_distance_min`, `price_distance_max`.
2. Drop all original `filter_*` columns afterwards.
3. Default (D1): **drop the two match flags** after documenting they have zero variance, so the MLP does not receive constants.
4. **Keep both price distances** (they vary and encode position in the budget band).
5. Parse `dimensions_in` → `volume`; drop the string.
6. Cyclical encodings for hour and day-of-week; drop `timestamp`.

## Consequences

- No duplicate one-hots for category/storage via filters.
- Notebook still demonstrates the interaction idea (compute → variance check → drop).
- `volume` is a single continuous size feature.

## Alternatives

- Keep raw filters + item fields: collinear with category/storage; worse pedagogy.
- Keep constant match flags: extra two dimensions with no information.
- Skip match flags entirely: slightly less aligned with the written assignment prompt.
