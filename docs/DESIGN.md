# End-to-end design

Status: **proposed, no code**. Change via [OPEN_DECISIONS.md](OPEN_DECISIONS.md); lock via ADRs.

Related: [DATA_PROFILE.md](DATA_PROFILE.md), [NOTEBOOK_SPEC.md](NOTEBOOK_SPEC.md).

---

## 1. Problem formulation

**Task:** binary classification of `bought` for each search impression.

**Business metric:** BTR = purchases / impressions. At row level, the model estimates \(P(bought=1 \mid x)\). Ranking quality of those probabilities is evaluated with ROC-AUC and PR-AUC. No decision threshold.

**Why not regression of query-level BTR?** The PDF defines BTR globally but the table is impression-level, and evaluation is specified as classification AUCs. Query-level rates would throw away ranking among items in the same SERP.

---

## 2. Columns: keep, drop, derive

### Drop before modeling

| Column | Reason |
| --- | --- |
| `cart` | Target leakage (no purchase without cart; see data profile) |
| `query_id` | Identifier; **not** a feature. Keep a copy **only** until splits are done |
| `filter_category`, `filter_storage_type`, `filter_price_min`, `filter_price_max` | After interaction features are built |
| `dimensions_in` | After `volume` |
| `timestamp` | After cyclical encodings |
| `package_size` | Redundant with `net_weight_oz` + `unit_of_measure` |

### Derived features

| Feature | Definition |
| --- | --- |
| `is_category_match` | \(1\) if `category == filter_category` else \(0\) |
| `is_storage_match` | \(1\) if `storage_type == filter_storage_type` else \(0\) |
| `price_distance_max` | `filter_price_max - price` |
| `price_distance_min` | `price - filter_price_min` |
| `volume` | \(L \times W \times H\) from `dimensions_in` |
| `hour_sin`, `hour_cos` | \(\sin(2\pi h/24),\ \cos(2\pi h/24)\) |
| `day_sin`, `day_cos` | \(\sin(2\pi d/7),\ \cos(2\pi d/7)\) with `d = day_of_week` (Mon=0) |
| `text` | Concatenation of `title`, `description`, `ingredients` with the tokenizer’s separator |

**Degeneracy:** match flags are constant `1` on this CSV. Default: **compute them, log the variance, drop zero-variance columns** so the notebook still shows the intended engineering. Price distances stay. See D1.

### Target

`y = bought.astype(float)` in \(\{0, 1\}\).

---

## 3. Preprocessing (fit on train only)

Leakage rule: **every fitted transform** (imputer, scaler, one-hot, optional custom vocab) is fit on the **training split** and applied to valid/test.

### 3.1 Text

- Concatenate the three string fields.
- Tokenizer: Hugging Face `AutoTokenizer` from `bert-base-uncased` (D2).
- Padding and truncation to `max_length = 96` (whitespace length ≤ 50; WordPiece can split further).
- Outputs: `input_ids`, `attention_mask` (`int64`).
- **Do not** load BERT weights. Embeddings are `nn.Embedding` trained with the rest of the model.

### 3.2 Numerical (median impute + `StandardScaler`)

`price`, `net_weight_oz`, `nutrition_score`, `volume`, `price_distance_max`, `price_distance_min`.

No missing values today; the pipeline stays for robustness.

### 3.3 Binary / already bounded (no scaler)

If kept: `is_category_match`, `is_storage_match`.

Always: `hour_sin`, `hour_cos`, `day_sin`, `day_cos` (already in \([-1, 1]\)).

### 3.4 Categorical (one-hot)

`category`, `storage_type`, `unit_of_measure`, `country_of_origin`, `allergens` (NaN → `"None"`), `brand`.

`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

### 3.5 Tabular vector

Concatenate, in a **fixed documented order**:

1. Scaled numeric (6)
2. Cyclical (4)
3. Optional match flags (0 or 2)
4. One-hot block

Approximate width: ~6 + 4 + 53 one-hots ≈ **63** (plus 2 if matches kept). Call this `n_tabular`.

---

## 4. Splits

- **Unit of splitting:** `query_id`, not rows.
- **Ratios (default):** 70% / 15% / 15% of queries → train / valid / test (D3).
- **Seed:** 42.
- After grouping, drop `query_id` from feature tables.
- Test is touched **once** for the reported numbers; architecture and early stopping use train/valid only.

Rationale: [adr/0005](../adr/0005-grouped-splits.md).

---

## 5. PyTorch data

### Dataset contract

Each item:

- `input_ids`: `LongTensor [L]`
- `attention_mask`: `LongTensor [L]` (1 = token, 0 = pad)
- `tabular`: `FloatTensor [n_tabular]`
- `label`: `FloatTensor []` (scalar)

### Device

**Dataset and DataLoader stay on CPU.** Move **batches** to `device` in the train/eval loop (`pin_memory=True` when CUDA). Putting tensors on GPU inside `__getitem__` breaks `num_workers` and is harder to debug. The user request to “map tensors to device” is satisfied at batch time.

### Loaders

- `batch_size` default 64 (D7).
- `shuffle=True` only on train.
- Collate: default stacking is enough (already padded by the tokenizer).

---

## 6. Model: hybrid Transformer + MLP

Constraint from the PDF: start **small** (`d_model < 100`), cheap to train on Colab.

### Default hyperparameters (D4)

| Symbol | Value |
| --- | --- |
| `d_model` | 64 |
| `nhead` | 4 |
| `num_encoder_layers` | 2 |
| `dim_feedforward` | 128 |
| `dropout` (encoder) | 0.1 |
| `max_length` | 96 |
| MLP hidden | 128 → 64 |
| MLP dropout | 0.2 |

### Forward (hybrid)

1. `tok_emb = Embedding(vocab_size, d_model)(input_ids) * sqrt(d_model)` (optional scale, Transformer-paper style).
2. Add **sinusoidal** positional encoding (length `max_length`). Pedagogical choice: matches Vaswani; learned PE is the alternative (D5).
3. `nn.TransformerEncoder` with `batch_first=True`.
4. Padding: convert HF `attention_mask` to `src_key_padding_mask` (`True` where pad).
5. **Masked** global average pooling over the sequence → `text_vec` `[B, d_model]`.
6. `h = concat(text_vec, tabular)`.
7. MLP: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → **1 logit**.

No sigmoid in the model. `BCEWithLogitsLoss` applies it internally.

### Ablation: tabular-only

Flag `use_text_encoder: bool` (name TBD). If false:

- Skip embedding/encoder/pooling.
- MLP input size = `n_tabular` only.
- Same loss, splits, and training loop.

This is the required ablation axis ([adr/0007](../adr/0007-training-ablation.md)). Stretch: text-only (D6).

### Why the Transformer sits on text

Tabular fields are already dense or one-hot. Self-attention is justified on **variable-length product language** (title, description, ingredients). A pretrained LLM encoder would hide the architecture the course wants to grade.

---

## 7. Training and evaluation

| Item | Choice |
| --- | --- |
| Loss | `BCEWithLogitsLoss` (optional `pos_weight` for 13% positives — D9) |
| Optimizer | AdamW, `lr=1e-3`, `weight_decay=1e-2` |
| Epochs | max 30 |
| Early stopping | monitor **validation PR-AUC**, mode max, patience 5, restore best weights |
| Metrics each epoch | train loss; valid ROC-AUC, PR-AUC; optional train AUCs |
| Test | once, best checkpoint |
| Logging | print + lists/dicts for later plots (loss, both AUCs vs epoch) |

PR-AUC via `sklearn.metrics.average_precision_score`. ROC-AUC via `sklearn.metrics.roc_auc_score`. Scores from **logits passed through sigmoid** (or equivalently `predict_proba` from logits).

Overfitting/underfitting: plot train vs valid PR-AUC; early stopping is the control. If both stay near the positive-class baseline (~0.13 PR-AUC), the model is underfitting (then scale `d_model` / epochs, not the reverse).

---

## 8. Reproducibility and Colab

- `device = cuda if available else cpu`.
- Seed NumPy, Python `random`, and PyTorch (and `cuda.manual_seed_all` if GPU).
- First cell: `pip install` torch, pandas, scikit-learn, matplotlib, seaborn, transformers.
- Data: upload CSV or mount Drive; path constant at the top of the notebook.
- Checkpoint: `best_model.pt` in the Colab session (and optional Drive copy).

---

## 9. What this design deliberately is not

- Not a ranking loss (ListNet, pairwise) even though BTR is a SERP metric. PDF asks for PR/ROC-AUC on `bought`.
- Not a `src/` package in the first implementation (see [adr/0008](../adr/0008-delivery-colab-notebook.md)). The tree sketched in the repo README is **deferred**.
