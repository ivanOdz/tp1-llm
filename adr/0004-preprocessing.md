# ADR 0004 — Preprocessing: tokenizer, scale, one-hot

- **Status:** Accepted (with D2, D8, D10, D13, D14 open for details)
- **Date:** 2026-08-26

## Context

Exercise 1 requires a per-feature preprocessing story. Mixed types: short English-like catalog text, continuous measurements, low-cardinality categoricals (`brand` ≈ 15).

## Decision

| Block | Transform | Fit on |
| --- | --- | --- |
| Text (`title` + `description` + `ingredients`) | HF tokenizer, pad/truncate → `input_ids`, `attention_mask` | Pretrained tokenizer (no vocab fit) |
| Six continuous columns listed in the assignment prompt | Median impute + `StandardScaler` | **Train only** |
| `category`, `storage_type`, `unit_of_measure`, `country_of_origin`, `allergens`, `brand` | One-hot (`handle_unknown="ignore"`) | **Train only** |
| Cyclical sin/cos | None | — |
| `allergens` NaN | Explicit level `"None"` before one-hot | — |

`package_size` is not encoded (redundant).

Embeddings for `input_ids` are **learned**, not copied from BERT.

## Consequences

- Tabular vector is a dense `float32` row; text is a pair of long tensors.
- Valid/test use `transform` only.
- `n_tabular` is known after fitting the encoder (needed for MLP `in_features`).

## Alternatives

- Ordinal / target encoding for `brand`: unnecessary at cardinality 15; one-hot is what the PDF suggests investigating.
- Character-level tokenizer: longer sequences, weaker morphology.
- Load pretrained BERT encoder: fights the “implement a Transformer” requirement.
