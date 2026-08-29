# ADR 0003 — Interaction features instead of raw filters

- **Status:** Accepted (engineering). Match-flag **retention** is open (D1).
- **Date:** 2026-08-26
- **Updated:** 2026-08-29 — price band → `relative_price_position` only

## Context

Raw filters (`filter_category`, `filter_storage_type`, `filter_price_*`) describe the query, not the item. Feeding them as-is either duplicates item fields or encourages the model to relearn “item vs filter” implicitly. The requested design is **explicit interactions**.

On this dataset, category and storage **always** match the filters, and price **always** lies inside the filter band ([EDA.md](../docs/EDA.md)).

## Decision

1. Build `is_category_match`, `is_storage_match`, and `relative_price_position = (price - filter_price_min) / (filter_price_max - filter_price_min)`. Zero-width band → **`0.5`**, asserted to occur 0 times on this CSV.
2. Drop all original `filter_*` columns afterwards.
3. Default (D1): **drop the two match flags** after documenting they have zero variance, so the MLP does not receive constants.
4. **Keep only `relative_price_position`** as the filter-derived price signal. Do **not** feed `price_distance_min` / `price_distance_max`.
5. Parse `dimensions_in` → `volume`; drop the string. Apply `log1p` before scaling (D17).
6. Cyclical encodings for hour and day-of-week; drop `timestamp`.

## Consequences

- No duplicate one-hots for category/storage via filters.
- Notebook still demonstrates the interaction idea (compute match flags → variance check → drop).
- One bounded feature encodes price position in the selected band without absolute distance units.
- `volume` is a single continuous size feature.

## Alternatives

- Keep raw filters + item fields: collinear with category/storage; worse pedagogy.
- Keep constant match flags: extra two dimensions with no information.
- Skip match flags entirely: loses the chance to demonstrate the interaction idea and its variance check, which is part of the Exercise 1 preprocessing story.
- Keep both absolute distances instead of (or with) relative position: collinear with band width, and the useful signal is *where* price sits in the band, not how far in dollars.

## Note on measured value

The whole engineered tabular block reaches only ROC-AUC 0.551 under gradient boosting ([EDA.md](../docs/EDA.md) §Text). These features are correct and well-justified engineering, but their downstream contribution on this dataset is near-noise. Do **not** judge the engineering by arm-level metrics.
