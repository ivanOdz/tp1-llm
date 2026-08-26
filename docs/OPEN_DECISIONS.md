# Open decisions

Edit this file before asking for the implementation plan. Defaults are what [DESIGN.md](DESIGN.md) currently assumes.

Legend: **Recommended** is the default if you say nothing.

| ID | Topic | Recommended | Alternatives | Notes |
| --- | --- | --- | --- | --- |
| D1 | Zero-variance match flags | Compute, then **drop** `is_category_match` and `is_storage_match` after logging that variance is 0 | Keep them anyway; skip creating them | They are always 1 on this CSV. Keeping them adds noise in the MLP. |
| D2 | Tokenizer | `bert-base-uncased` AutoTokenizer, **random** `nn.Embedding` | Train a small WordPiece on train text; whitespace / custom vocab | HF tokenizer is reproducible and matches the `transformers` install. Do not load BERT encoder weights. |
| D3 | Split ratios | **70 / 15 / 15** of `query_id` | 80/10/10 | ~302 queries in valid/test; ~200 positives — enough for PR-AUC. |
| D4 | Encoder size | `d_model=64`, 4 heads, 2 layers, FFN 128 | 32/2/2 (even cheaper); 96/4/2 or 3 layers as stretch | PDF: start with `d_model < 100`. |
| D5 | Positional encoding | **Sinusoidal** (Vaswani) | Learned `nn.Embedding(max_length, d_model)` | Sinusoidal is easier to explain in the presentation. |
| D6 | Extra ablation | **Hybrid vs tabular-only** only | Also text-only MLP | Text-only is nice if hybrid ≪ tabular (isolates harmful text). |
| D7 | Batch size | **64** | 32 if GPU memory is tight; 128 on a larger GPU | Sequence length 96 is small; 64 should be fine. |
| D8 | `nutrition_score == 0` | **Leave as 0** | Missing indicator + impute | 1 244 zeros; unknown if sentinel. |
| D9 | Class imbalance in loss | **No `pos_weight` first** | `pos_weight = n_neg/n_pos` on train | PR-AUC + early stopping may be enough; `pos_weight` can be a one-line experiment. |
| D10 | Max token length | **96** | 64 (tight); 128 (safe) | Whitespace max is 50; WordPiece can exceed 64. |
| D11 | Delivery | **One Colab notebook** | Notebook + `src/` package | Assignment env is Colab. Package is deferred (old README tree). |
| D12 | Move tensors to GPU | **In the training loop, per batch** | Inside `Dataset.__getitem__` | User asked for device mapping; batch-level is the correct interpretation. |
| D13 | Scale cyclical / binary | **Do not scale** sin/cos (or match flags) | Put them through `StandardScaler` | Already bounded. User listed scaler columns explicitly; follow that list. |
| D14 | Text concat separator | Tokenizer `sep_token` between fields | Plain space | SEP makes field boundaries visible to attention. |
| D15 | Split stratification | **Ungrouped random on query ids** (seeded) | Stratify queries by “has at least one purchase” | 52.6% of queries have zero buys; stratification would stabilize valid PR-AUC. |

## Locked (do not reopen without a reason)

- Target is `bought`, not `cart`.
- `cart` is not a feature.
- `query_id` is not a model input.
- Raw `filter_*` columns are not fed to the model after engineering.
- Single grouped train / valid / test split. **No k-fold.**
- Loss is `BCEWithLogitsLoss`.
- Reported metrics include ROC-AUC and PR-AUC.
- Early stopping monitors **validation PR-AUC**.
- Transformer is **from-scratch encoder**, not a pretrained LLM backbone.
- Preprocessors fit on **train only**.
