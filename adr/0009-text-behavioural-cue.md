# ADR 0009 — Behavioural cue in product text: keep, and ablate against a stripped arm

- **Status:** Accepted
- **Date:** 2026-08-29
- **Supersedes:** the "signal may be weak" expectation in [EDA.md](../docs/EDA.md) §Text

## Context

[ADR 0006](0006-hybrid-transformer-mlp.md) assumed the tabular block carries most of the signal and that templated descriptions would contribute little, with the tabular-only ablation as the way to "find out". Measurement on a grouped 70/15/15 split (seed 42) inverts that assumption:

| Model | ROC-AUC | PR-AUC | Baseline |
| --- | ---: | ---: | ---: |
| Logistic regression, full 62-dim tabular block | 0.517 | 0.141 | 0.137 |
| Gradient boosting, full 62-dim tabular block | 0.551 | 0.154 | 0.137 |
| TF-IDF on concatenated text | 0.955 | 0.648 | 0.132 |
| Title parenthetical alone, mapped to its train purchase rate | 0.960 | 0.670 | 0.132 |

The dominant predictor is a behavioural phrase embedded in `title` and restated in `description`. The title parenthetical takes 20 values that split into three regimes:

| Cue group | n | Purchase rate |
| --- | ---: | ---: |
| Customer Favorite, Best Seller, Top Rated, #1 Pick | 1 931 | 0.63 – 0.68 |
| Well Reviewed, Shopper Favorite, Highly Rated, Popular Choice | 1 973 | 0.02 – 0.04 |
| Remaining 12 (Clearance Listing, Rarely Reordered, Standard Listing, New Listing, …) | 6 096 | 0.00 |

Highest-weight TF-IDF n-grams are `most repurchased`, `frequently reordered`, `returning customers`, `customer pick`; lowest are `less repurchased`, `rarely reordered`, `limited`. Removing the title parenthetical and the final description sentence still leaves ROC-AUC 0.953, so the cue is redundantly encoded across both fields and cannot be removed by deleting one span.

This is aggregate historical repurchase behaviour written into the catalog string. [ADR 0002](0002-leakage-cart-query-id.md) rejects `cart` because it is "a downstream action, not an attribute of the impression at render time". The cue is genuinely present in the rendered title, so it is not leakage in the `cart` sense — but it is a target-derived proxy, and a model that reads it is doing token lookup rather than language understanding.

## Decision

**Keep the cue, and make the comparison against a cue-stripped arm the project's primary ablation.**

Three arms, identical splits, seeds, tabular tensor, and training loop:

| Arm | Text input | Purpose |
| --- | --- | --- |
| A — hybrid (full text) | `title + description + ingredients` as-is | Upper bound; measures cue visibility |
| B — hybrid (cue-stripped) | Same, with behavioural phrases removed | Measures what the encoder learns from product language proper |
| C — tabular-only | none (`use_text_encoder=False`) | Measures the non-text feature block |

Cue stripping is a documented, regex-based removal applied identically to all splits (it is not a fitted transform, so it carries no leakage risk):

1. Drop the trailing parenthetical from `title`.
2. Drop description sentences matching the behavioural template set (repurchase, reorder, feedback, rating, and listing-status phrasings).
3. Log the retained-token delta and assert the top TF-IDF coefficients of a probe model no longer contain behavioural n-grams.

Report all three arms with non-neural baselines (majority-rate, logistic regression, gradient boosting on the same tabular block) in one table.

## Consequences

- The presentation's headline finding becomes **"the target is encoded in the catalog text, and here is how we detected and quantified it"** — a leakage-analysis result, reported deliberately rather than discovered as an unexplained gap in a comparison table.
- Arm A vs Arm C is no longer evidence that "attention on product language helps". That claim, if made at all, rests on Arm B vs Arm C.
- Arm B is expected to land near the tabular baseline. A flat result there is a valid, reportable outcome and must **not** be treated as underfitting (see [ADR 0007](0007-training-ablation.md) and the revised diagnostic in [DESIGN.md](../docs/DESIGN.md) §7).
- Exercise 2's "comparación de alternativas de los distintos módulos" is satisfied empirically by three arms plus baselines, not by prose alone.
- Cost: three training runs instead of two, plus the stripping regex and its verification probe.

## Alternatives

- **Keep the cue only (two arms, as originally specified):** the ablation reports a large gap whose cause is misattributed to architecture. Rejected.
- **Strip the cue only:** honest, but discards the most interesting finding in the dataset and leaves a near-null result with nothing to contrast it against. Rejected.
- **Treat the cue as leakage and drop the text branch entirely:** would remove the Transformer, which the assignment requires. Rejected.
- **Extract the cue as an explicit categorical feature and one-hot it:** cleaner modelling, but it moves the signal into the tabular block and makes the text encoder redundant — the opposite of the assignment's pedagogical target. Worth one row in the baseline table, not an arm.
