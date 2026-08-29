# ADR 0004 — Preprocessing: tokenizer, scale, one-hot

- **Status:** Accepted (D2, D8, D13 open for details; D10, D14, D17, D22 resolved by measurement)
- **Date:** 2026-08-26
- **Updated:** 2026-08-29 — `max_length` 96 → 80, `log1p` on `volume`, provenance corrected

## Context

Exercise 1 requires a per-feature preprocessing story. Mixed types: short English-like catalog text, continuous measurements, low-cardinality categoricals (`brand` = 15 levels).

The continuous set is **derived from the data**, not enumerated by the PDF: the PDF lists all 22 columns without designating types. Four survive as continuous after `package_size` is dropped as redundant and `dimensions_in` is parsed into `volume`.

## Decision

| Block | Transform | Fit on |
| --- | --- | --- |
| Text (`title` + `description` + `ingredients`, joined by `sep_token`) | HF tokenizer, `padding="max_length"`, `truncation=True`, `max_length=80` → `input_ids`, `attention_mask` | Pretrained tokenizer (no vocab fit) |
| Four continuous columns: `price`, `net_weight_oz`, `nutrition_score`, `log1p(volume)` | Median impute + `StandardScaler` | **Train only** |
| `category`, `storage_type`, `unit_of_measure`, `country_of_origin`, `allergens`, `brand` | One-hot (`handle_unknown="ignore"`) | **Train only** |
| Cyclical sin/cos + `relative_price_position` | None (already bounded) | — |
| `allergens` NaN | Explicit level `"None"` before one-hot | — |

`package_size` is not encoded (redundant).

Embeddings for `input_ids` are **learned**, not copied from BERT.

## Consequences

- Tabular vector is a dense `float32` row of width **62** on this CSV; text is a pair of long tensors of length 80.
- Valid/test use `transform` only.
- `n_tabular` is **read from the fitted encoder**, never hardcoded (needed for MLP `in_features`).
- `max_length=80` truncates zero rows (measured WordPiece max 74) while avoiding the 42% padding waste of 96.
- `log1p` on `volume` keeps every scaled numeric inside \(|z| < 5\).

## Alternatives

- Ordinal / target encoding for `brand`: unnecessary at cardinality 15; one-hot is what the PDF suggests investigating.
- Character-level tokenizer: longer sequences, weaker morphology.
- Load pretrained BERT encoder: fights the “implement a Transformer” requirement.
