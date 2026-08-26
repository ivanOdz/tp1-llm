# Notebook spec (cells, no code)

Artifact to generate later: `notebooks/btr_transformer_colab.ipynb` (name can change). Environment: Google Colab, GPU recommended.

Each section below is one or more **sequential cells**. Implementation must follow [DESIGN.md](DESIGN.md) unless an [open decision](OPEN_DECISIONS.md) was changed.

Do not implement this file yet.

---

## Cell group 0 — Title and design pointers (markdown)

- Course, problem (BTR / `bought`), hybrid Transformer + MLP.
- Links to `docs/` and `adr/` in the repo (for local clones). In Colab, a short restatement of leakage + grouped split is enough so the notebook stands alone for graders.

## Cell group 1 — Environment and data loading

1. Markdown: hardware, dependencies.
2. Code: `pip install` torch, pandas, scikit-learn, matplotlib, seaborn, transformers.
3. Code: imports, `device`, seeds.
4. Markdown: expected CSV schema.
5. Code: `DATA_PATH`, `pd.read_csv`, `shape` / `dtypes` / missingness / `bought` rate / `cart × bought` / `query_id` counts.
6. Markdown: 3–4 sentence EDA conclusions (leakage, imbalance, degenerate filters) citing the printed tables.

## Cell group 2 — Feature engineering

1. Markdown: what is dropped and why (`cart`, `query_id` as feature, `package_size`).
2. Code: parse `dimensions_in` → `volume`; drop `dimensions_in`.
3. Code: match flags + price distances; drop raw `filter_*`.
4. Code: assert or log variance of match flags; drop zero-variance columns if D1 = drop.
5. Code: `timestamp` → UTC datetime → `hour`, `day_of_week` → four sin/cos columns; drop `timestamp`, `hour`, `day_of_week`.
6. Code: `allergens` fill `None`; `text` concatenation; `y` from `bought`.
7. Code: preview of engineered frame (columns, `head`, `volume` describe).

Keep `query_id` in the frame **until after the split**.

## Cell group 3 — Train / valid / test split

1. Markdown: grouped split rationale (no query in two sets).
2. Code: split **queries** 70/15/15, seed 42; map rows; print row counts, query counts, positive rates per split.
3. Code: drop `query_id` from the three frames.

## Cell group 4 — Preprocessing pipeline

1. Markdown: fit on train only.
2. Code: constants listing numeric / cyclical / categorical columns (and leftover binaries).
3. Code: `SimpleImputer` + `StandardScaler` on numeric; `OneHotEncoder` on categoricals; concatenate with unscaled cyclical (and matches if kept).
4. Code: tokenizer + `max_length`; function `encode_text(series) → input_ids, attention_mask`.
5. Code: print `n_tabular`, tokenizer vocab size, a decoded sample.

## Cell group 5 — Dataset and DataLoader

1. Markdown: contract (text tensors + tabular + label); CPU dataset, batch `.to(device)` later.
2. Code: typed `ImpressionDataset(Dataset)`.
3. Code: three datasets + three DataLoaders; one-batch smoke test (`shapes`, `device` after a manual move).

## Cell group 6 — Model

1. Markdown: diagram in words (embed → PE → encoder → masked GAP → concat tabular → MLP → logit). Ablation flag. Hyperparameter table.
2. Code: sinusoidal PE module (or function).
3. Code: `BTRHybridModel(nn.Module)` with `use_text_encoder: bool`.
4. Code: instantiate hybrid + tabular-only; print parameter counts.

## Cell group 7 — Train / eval helpers

1. Markdown: loss, metrics, early stopping on valid PR-AUC.
2. Code: `logits_to_proba`, `batch_to_device`, `compute_aucs(y_true, y_score)`.
3. Code: `train_one_epoch`, `@torch.no_grad() evaluate`.
4. Code: `fit(...)` with early stopping, history dict, best-state restore.

## Cell group 8 — Run A: hybrid (Transformer on)

1. Markdown: this is the main system.
2. Code: train; plots (loss, valid ROC-AUC, valid PR-AUC vs epoch); test metrics of the restored checkpoint.

## Cell group 9 — Run B: ablation (Transformer off)

1. Markdown: same splits, same tabular tensor, MLP-only.
2. Code: train; test metrics.
3. Code: comparison table (valid/test ROC-AUC, PR-AUC, params, epochs run).

## Cell group 10 — Discussion (markdown)

- Did text help? Overfit vs underfit from the curves.
- Degenerate match features: what we observed.
- What we would scale next (`d_model`, layers) if valid PR-AUC still climbing at patience boundary.

## Cell group 11 — Optional stretch (commented or skipped)

Only if time: `d_model=96` or 3 layers, **same splits**, one extra row in the comparison table. Do not touch test until that run is chosen as a candidate (or accept that stretch runs also report test if labeled as exploratory).

---

## Colab file layout (runtime)

Not committed unless we later export:

- `best_hybrid.pt` / `best_tabular.pt`
- matplotlib figures inline

Repo layout after implementation (planned, not created now):

- `notebooks/btr_transformer_colab.ipynb`
- `data/raw/supermarket_products.csv` (gitignored if large)
- these `docs/` and `adr/` files
