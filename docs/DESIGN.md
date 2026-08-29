# End-to-end design

Status: **proposed, no code**. Change via [OPEN_DECISIONS.md](OPEN_DECISIONS.md); lock via ADRs.

Related: [EDA.md](EDA.md), [NOTEBOOK_SPEC.md](NOTEBOOK_SPEC.md).

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
| `cart` | Target leakage (no purchase without cart; see [EDA.md](EDA.md)) |
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
| `relative_price_position` | \((price - filter_price_min) / (filter_price_max - filter_price_min)\); if band width is \(0\), return **`0.5`** (band centre) |
| `volume` | \(L \times W \times H\) from `dimensions_in`, then `log1p` at scaling time (§3.2) |
| `hour_sin`, `hour_cos` | \(\sin(2\pi h/24),\ \cos(2\pi h/24)\) |
| `day_sin`, `day_cos` | \(\sin(2\pi d/7),\ \cos(2\pi d/7)\) with `d = day_of_week` (Mon=0) |
| `text` | Concatenation of `title`, `description`, `ingredients` with the tokenizer’s separator |

**Zero-width bands:** there are **0** such rows in this CSV, so the guard is defensive only. It must still be defined rather than left to produce `NaN`: width \(= 0\) → `0.5`, and the notebook asserts the guard fires zero times. Any non-zero count is a data-quality alarm, not a silent fallback.

**Degeneracy:** match flags are constant `1` on this CSV. Default: **compute them, log the variance, drop zero-variance columns** so the notebook still shows the intended engineering. **Do not** keep `price_distance_min` / `price_distance_max`; only `relative_price_position` enters the model. See D1.

### Target

`y = bought.astype(float)` in \(\{0, 1\}\).

---

## 3. Preprocessing (fit on train only)

Leakage rule: **every fitted transform** (imputer, scaler, one-hot, optional custom vocab) is fit on the **training split** and applied to valid/test.

### 3.1 Text

- Concatenate the three string fields with the tokenizer's `sep_token` between them (D14).
- Tokenizer: Hugging Face `AutoTokenizer` from `bert-base-uncased` (D2), `add_special_tokens=True`.
- `padding="max_length"`, `truncation=True`, **`max_length = 80`** (D10). Measured WordPiece length including `[CLS]`/`[SEP]` and the two field separators: mean 55.6, p95 64, **max 74**. 80 truncates **zero** rows with modest headroom; 96 wastes 42% of encoder positions on padding, 64 would truncate 4.54% of rows.
- Outputs: `input_ids`, `attention_mask` (`int64`).
- **Do not** load BERT weights. Embeddings are `nn.Embedding` trained with the rest of the model.
- **Arm B** applies the cue-stripping regex to `title` and `description` *before* concatenation ([adr/0009](../adr/0009-text-behavioural-cue.md)). Stripping is a fixed rule, not a fitted transform, so it is applied identically to train/valid/test.

### 3.2 Numerical (median impute + `StandardScaler`)

`price`, `net_weight_oz`, `nutrition_score`, `log1p(volume)`.

`volume` is skewed 2.97 and reaches \(|z| = 9.61\) after scaling, with 50 rows beyond \(|z| > 5\); `log1p` first gives skew 0.44 and max \(|z| = 3.05\). The other three are transformed as-is (max \(|z| \le 4.7\)). See D17.

No missing values today; the imputer stays for robustness.

### 3.3 Bounded (no scaler)

Always: `relative_price_position` (in \([0, 1]\) when price is in-band), `hour_sin`, `hour_cos`, `day_sin`, `day_cos` (already in \([-1, 1]\)).

If kept: `is_category_match`, `is_storage_match`.

### 3.4 Categorical (one-hot)

`category`, `storage_type`, `unit_of_measure`, `country_of_origin`, `allergens` (NaN → `"None"`), `brand`.

`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

### 3.5 Tabular vector

Concatenate, in a **fixed documented order**:

1. Scaled numeric (4)
2. Bounded: `relative_price_position` + cyclical (5)
3. Optional match flags (0 or 2)
4. One-hot block

Approximate width: ~4 + 5 + 53 one-hots ≈ **62** (plus 2 if matches kept). Call this `n_tabular`.

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

Constraint from the PDF: start **small** (`d_model < 100`) and keep compute non-limiting, then increase complexity within available compute. The scale-up is part of the graded experiment, not an optional extra (see §7.2 and [NOTEBOOK_SPEC.md](NOTEBOOK_SPEC.md) cell group 11).

### Default hyperparameters (D4)

| Symbol | Value |
| --- | --- |
| `d_model` | 64 |
| `nhead` | 4 |
| `num_encoder_layers` | 2 |
| `dim_feedforward` | 128 |
| `dropout` (encoder) | 0.1 |
| `max_length` | 80 |
| MLP hidden | 128 → 64 |
| MLP dropout | 0.2 |

### Forward (hybrid)

1. `tok_emb = Embedding(vocab_size, d_model)(input_ids) * sqrt(d_model)` (optional scale, Transformer-paper style).
2. Add **sinusoidal** positional encoding (length `max_length`). Pedagogical choice: matches Vaswani; learned PE is the alternative (D5).
3. `nn.TransformerEncoder` with `batch_first=True`.
4. Padding: convert HF `attention_mask` to `src_key_padding_mask` (`True` where pad).
5. **Masked** global average pooling over the sequence → `text_vec` `[B, d_model]`. Pooling averages over **all non-pad positions, including `[CLS]`, `[SEP]` and the two field separators**. They are content-bearing here (the separators mark field boundaries, which is D14's whole point) and excluding them would mean maintaining a second mask. State this explicitly in the notebook — it is the kind of detail a grader asks about.
6. `h = concat(text_vec, tabular)`.
7. MLP: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → **1 logit**.

Explicit widths, so `in_features` is never guessed:

| Arm | MLP input | Hidden | Output |
| --- | ---: | --- | ---: |
| A, B (hybrid) | `d_model + n_tabular` = 64 + 62 = **126** | 128 → 64 | 1 |
| C (tabular-only) | `n_tabular` = **62** | 128 → 64 | 1 |

`n_tabular = 62` is confirmed by construction on this CSV (4 numeric + 5 bounded + 53 one-hots, match flags dropped per D1). It must still be **read from the fitted encoder at runtime**, not hardcoded, because `OneHotEncoder` is fit on train only.

No sigmoid in the model. `BCEWithLogitsLoss` applies it internally.

### Ablation: three arms

Controlled by two constructor arguments — `use_text_encoder: bool` and, at the data level, `strip_behavioural_cue: bool`:

| Arm | `use_text_encoder` | `strip_behavioural_cue` | Question it answers |
| --- | --- | --- | --- |
| A | `True` | `False` | Upper bound with the cue visible |
| B | `True` | `True` | Does the encoder learn anything from product language proper? |
| C | `False` | — | What do the non-text features carry? |

When `use_text_encoder=False`: skip embedding/PE/encoder/pooling, MLP input is `n_tabular` (62), everything else identical — same splits, seeds, loss, optimizer, schedule and early stopping.

**A vs C** measures cue visibility. **B vs C** is the only pair that speaks to the Transformer's value on language. Rationale: [adr/0009](../adr/0009-text-behavioural-cue.md), [adr/0007](../adr/0007-training-ablation.md). D6 (text-only arm) remains an optional fourth.

### Why the Transformer sits on text

Tabular fields are already dense or one-hot. Self-attention is justified on **variable-length product language** (title, description, ingredients). A pretrained LLM encoder would hide the architecture the course wants to grade.

---

## 7. Training and evaluation

| Item | Choice |
| --- | --- |
| Loss | `BCEWithLogitsLoss` (optional `pos_weight` for 13% positives — D9) |
| Optimizer | AdamW, `lr=3e-4`, `weight_decay=1e-2` (D18) |
| LR schedule | Linear warmup over the first 10% of steps, then constant (D18) |
| Gradient clipping | `clip_grad_norm_(max_norm=1.0)` (D18) |
| Epochs | max 30 |
| Early stopping | monitor **validation PR-AUC**, mode max, patience 5, restore best weights |
| Metrics each epoch | train loss; **train and valid** ROC-AUC and PR-AUC (train curves are required by the §7 diagnostic, not optional) |
| Test | once per arm, best checkpoint |
| Logging | print + history dict for later plots (loss, both AUCs vs epoch) |

**Why not `lr=1e-3`:** `nn.TransformerEncoderLayer` defaults to post-LN, which is the configuration known to need warmup to train stably. Either lower the LR with warmup (chosen) or set `norm_first=True`. Arm C (no encoder) is insensitive to this and keeps the same settings so the arms stay comparable.

PR-AUC via `sklearn.metrics.average_precision_score`. ROC-AUC via `sklearn.metrics.roc_auc_score`. Scores from **logits passed through sigmoid** (or equivalently `predict_proba` from logits).

### Overfitting / underfitting / no-signal

Plot train **and** valid PR-AUC; early stopping is the control. Read the two curves jointly — the three diagnoses are distinguished by the **train** curve, not by the valid curve alone:

| Train PR-AUC | Valid PR-AUC | Diagnosis | Action |
| --- | --- | --- | --- |
| Low (~prevalence) | Low (~prevalence) | **Underfitting or no signal** — ambiguous on its own | Compare against the baselines in §7.1. If gradient boosting on the same inputs is also at prevalence, the inputs carry no signal and scaling the model is the **wrong** action. |
| High | Low | **Overfitting** | Regularize: dropout, weight decay, smaller `d_model`, earlier stop. |
| Rising at the patience boundary | Rising | **Undertrained** | Raise max epochs / patience, then scale `d_model`. |

This matters concretely here: Arm C (tabular-only) and probably Arm B (cue-stripped text) will sit near the 0.13 prevalence baseline because their inputs are near-noise ([EDA.md](EDA.md) §Text), **not** because the network is too small. Scaling `d_model` in that situation burns compute and produces a misleading "we tried harder" narrative.

### 7.1 Reference baselines (required, not optional)

Fit on the same grouped split and reported in the same table as the neural arms:

| Baseline | Purpose |
| --- | --- |
| Predict the train prevalence for every row | Defines the PR-AUC floor (≈ 0.13) |
| Logistic regression on the tabular block | Linear ceiling for non-text features (measured: ROC-AUC 0.517) |
| Gradient boosting on the tabular block | Strong non-linear tabular ceiling (measured: ROC-AUC 0.551) |
| Title cue as a one-hot categorical → logistic regression | Shows how much of Arm A is one token (measured: ROC-AUC 0.960) |

Without these, "does the Transformer help?" has no reference point and the PDF's request to compare module alternatives is unmet.

### 7.2 Run repeats and reported dispersion

A single grouped split cannot resolve PR-AUC differences below **~0.05** ([EDA.md](EDA.md) §Metric noise floor). Every arm is therefore run over **5 seeds** on the same split protocol, and the comparison table reports **mean ± sd**. This is not k-fold — the train/valid/test protocol is unchanged — so it does not reopen the locked split decision (D16).

### 7.3 Secondary per-query diagnostic

On queries with at least one purchase, report **mean per-query average precision** and **recall@1**. These are diagnostics, not the selection metric — early stopping still uses global valid PR-AUC (locked). They exist so the results section can speak to the PDF's framing ("identificar los mejores productos y promocionarlos") rather than only to pooled classification.

---

## 8. Reproducibility (local Jupyter)

Execution target is **local Jupyter**, consistent with [NOTEBOOK_SPEC.md](NOTEBOOK_SPEC.md), [adr/0008](../adr/0008-delivery-notebook.md) and D11. GPU is used if present, otherwise CPU (this dataset trains on CPU in minutes).

- `device = cuda if available else cpu`.
- Seed NumPy, Python `random`, and PyTorch (and `cuda.manual_seed_all` if GPU) from a single `SEED` constant.
- Dependencies come from the pinned [`requirements.txt`](../requirements.txt) (`pip install -r requirements.txt`), **not** an inline `pip install` cell. The notebook asserts the imported versions instead of installing them.
- Data: `DATA_PATH` constant at the top of the notebook, pointing at `data/raw/supermarket_products.csv`. No upload or Drive mount.
- Tokenizer: `bert-base-uncased` is downloaded from the Hugging Face Hub on first run and cached. Offline reruns need `HF_HOME` pointed at that cache; the notebook prints the resolved cache path so a grader can reproduce without network.
- Checkpoints: `best_arm_a.pt`, `best_arm_b.pt`, `best_arm_c.pt` in the working directory, gitignored. Filenames must match [NOTEBOOK_SPEC.md](NOTEBOOK_SPEC.md) §Runtime file layout.

---

## 9. What this design deliberately is not

- Not a ranking loss (ListNet, pairwise) even though BTR is a SERP metric. PDF asks for PR/ROC-AUC on `bought`.
- Not a `src/` package in the first implementation (see [adr/0008](../adr/0008-delivery-notebook.md)). The modular tree and parquet pipeline are **deferred**; their placeholder directories were removed from the repo.
- Not a claim that self-attention on product language drives performance. Arm A's headline number is largely one behavioural token; the language claim, if made at all, rests on Arm B vs Arm C ([adr/0009](../adr/0009-text-behavioural-cue.md)).

## 10. Known limitations to state in the presentation

- **The dataset encodes the target in the catalog text.** Arm A's ~0.96 ROC-AUC is a leakage-detection result, not an architecture result.
- **The non-text feature block is near-noise** (gradient boosting: 0.551 ROC-AUC). Feature engineering quality cannot be judged from downstream metrics on this data.
- **Evaluation is a single grouped split with seed repeats**, not k-fold. Reported dispersion is across seeds, not across data partitions.
- **BTR is a per-SERP rate, but the primary metrics pool globally.** A per-query diagnostic is reported alongside (§7.3) because 52.6% of queries have zero purchases and global ROC-AUC is therefore driven partly by between-query propensity rather than within-SERP ordering.
