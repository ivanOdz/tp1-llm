# Architecture Decision Records

Format: short markdown, one decision per file. Status is `Accepted` unless noted.

| ID | Title |
| --- | --- |
| [0001](0001-target-bought-btr.md) | Target is impression-level `bought` |
| [0002](0002-leakage-cart-query-id.md) | Drop `cart` and `query_id` as features |
| [0003](0003-feature-engineering.md) | Interactions; `relative_price_position` only |
| [0004](0004-preprocessing.md) | Tokenizer, scaler, one-hot (fit on train) |
| [0005](0005-grouped-splits.md) | Train/valid/test grouped by `query_id` |
| [0006](0006-hybrid-transformer-mlp.md) | Light Transformer on text + MLP |
| [0007](0007-training-ablation.md) | PR-AUC early stopping and tabular ablation |
| [0008](0008-delivery-notebook.md) | Notebook delivery; `src/` package deferred |
| [0009](0009-text-behavioural-cue.md) | Behavioural cue in product text: keep, ablate against a stripped arm |
