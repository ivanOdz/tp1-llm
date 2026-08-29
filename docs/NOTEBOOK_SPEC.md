# Notebook spec

Artifact: [`notebooks/btr_transformer.ipynb`](../notebooks/btr_transformer.ipynb). Groups 0–7 are implemented; 8–12 are a placeholder. Environment: **local Jupyter**, GPU used if available, CPU adequate at this scale ([DESIGN.md](DESIGN.md) §8).

Each section below is one or more **sequential cells**. Implementation must follow [DESIGN.md](DESIGN.md) unless an [open decision](OPEN_DECISIONS.md) was changed.

---

## Cell group 0 — Title and design pointers (markdown)

- Course, problem (BTR / `bought`), hybrid Transformer + MLP.
- Links to `docs/` and `adr/` in the repo. A short restatement of leakage + grouped split is enough so the notebook stands alone for graders.

## Cell group 1 — Environment and data loading

1. Markdown: hardware, and `pip install -r requirements.txt` as the setup step (run in the shell, not the notebook).
2. Code: version assertions for torch, pandas, scikit-learn, transformers — fail loudly on mismatch rather than installing.
3. Code: imports, `device`, single `SEED` constant, seed all RNGs; print the resolved Hugging Face cache path.
4. Markdown: expected CSV schema.
5. Code: `DATA_PATH`, `pd.read_csv`, `shape` / `dtypes` / missingness / `bought` rate / `cart × bought` / `query_id` counts.
6. Code: **behavioural-cue audit** — extract the `title` parenthetical, cross-tabulate against `bought`, and fit the cue-only and TF-IDF probes on a grouped split. This is the single most important EDA output ([EDA.md](EDA.md) §Text).
7. Markdown: EDA conclusions — `cart` leakage, imbalance, degenerate filters, **and the behavioural cue** — citing the printed tables.

## Cell group 2 — Feature engineering

1. Markdown: what is dropped and why (`cart`, `query_id` as feature, `package_size`).
2. Code: parse `dimensions_in` → `volume`; drop `dimensions_in`.
3. Code: match flags + `relative_price_position` (zero-width band → `0.5`, and assert the guard fires 0 times); drop raw `filter_*` (no absolute price distances).
4. Code: assert or log variance of match flags; drop zero-variance columns if D1 = drop.
5. Code: `timestamp` → UTC datetime → `hour`, `day_of_week` → four sin/cos columns; drop `timestamp`, `hour`, `day_of_week`.
6. Code: `allergens` fill `None`; `y` from `bought`.
7. Code: build **both** text columns — `text_full` (Arm A) and `text_stripped` (Arm B, cue removed) — using the tokenizer `sep_token` between fields; print a before/after sample pair.
8. Code: preview of engineered frame (columns, `head`, `volume` describe, `log1p(volume)` skew before/after).

Keep `query_id` in the frame **until after the split**.

## Cell group 3 — Train / valid / test split

1. Markdown: grouped split rationale (no query in two sets) + why queries are stratified by "has ≥1 purchase" (D15).
2. Code: split **queries** 70/15/15, **stratified on has-≥1-purchase**, seed from `SEED`; map rows; print row counts, query counts, positive rates per split, and assert no `query_id` appears in two splits.
3. Code: drop `query_id` from the three frames.

## Cell group 4 — Preprocessing pipeline

1. Markdown: fit on train only.
2. Code: constants listing numeric / bounded (`relative_price_position` + cyclical) / categorical columns (and leftover binaries).
3. Code: `log1p` on `volume`; `SimpleImputer` + `StandardScaler` on the four numerics; `OneHotEncoder` on categoricals; concatenate with unscaled bounded features (and matches if kept).
4. Code: tokenizer + `max_length=80`; function `encode_text(series) → input_ids, attention_mask`. Print the measured WordPiece length distribution to justify 80 and assert zero truncation.
5. Code: print `n_tabular` (expect 62, read from the fitted encoder), tokenizer vocab size, a decoded sample showing the `[SEP]` field boundaries.

## Cell group 5 — Dataset and DataLoader

1. Markdown: contract (text tensors + tabular + label); CPU dataset, batch `.to(device)` later.
2. Code: typed `ImpressionDataset(Dataset)`.
3. Code: a dataset/loader factory over (split, text variant) so Arms A and B share one code path; one-batch smoke test (`shapes`, `device` after a manual move).

## Cell group 6 — Model

1. Markdown: diagram in words (embed → PE → encoder → masked GAP over all non-pad positions → concat tabular → MLP → logit). Ablation flag. Hyperparameter table with explicit MLP input widths (126 hybrid / 62 tabular-only).
2. Code: sinusoidal PE module (or function).
3. Code: `BTRHybridModel(nn.Module)` with `use_text_encoder: bool`.
4. Code: instantiate hybrid + tabular-only; print parameter counts and note that the embedding table dominates (~1.95M of the hybrid total).

## Cell group 7 — Train / eval helpers

1. Markdown: loss, metrics, early stopping on valid PR-AUC, warmup + gradient clipping and why ([DESIGN.md](DESIGN.md) §7).
2. Code: `logits_to_proba`, `batch_to_device`, `compute_aucs(y_true, y_score)`, `per_query_metrics(...)`.
3. Code: `train_one_epoch` (with warmup schedule and `clip_grad_norm_`), `@torch.no_grad() evaluate`.
4. Code: `fit(...)` with early stopping, history dict (train **and** valid curves), best-state restore.
5. Code: `run_arm(...)` wrapper that loops 5 seeds and returns mean ± sd.

## Cell group 8 — Reference baselines

1. Markdown: why non-neural baselines are required before any neural claim ([DESIGN.md](DESIGN.md) §7.1).
2. Code: prevalence predictor, logistic regression and gradient boosting on the tabular block, and the title-cue one-hot model. Same grouped split.
3. Code: baseline table (ROC-AUC, PR-AUC, prevalence floor per split).

## Cell group 9 — Three arms

Each arm runs over **5 seeds** ([DESIGN.md](DESIGN.md) §7.2); report mean ± sd.

1. Markdown: Arm A (full text) / Arm B (cue-stripped) / Arm C (tabular-only) and the question each answers.
2. Code: cue-stripping function + verification probe — assert the top TF-IDF coefficients of a probe model no longer contain behavioural n-grams ([adr/0009](../adr/0009-text-behavioural-cue.md)).
3. Code: Arm A — train; plots of **train and valid** loss, ROC-AUC, PR-AUC vs epoch.
4. Code: Arm B — train; same plots.
5. Code: Arm C — train; same plots.
6. Code: test metrics for each arm's best checkpoint, evaluated once.
7. Code: unified comparison table — baselines + three arms: valid/test ROC-AUC and PR-AUC (mean ± sd), per-query AP and recall@1, parameter count, best epoch.

## Cell group 10 — Discussion (markdown)

- **A vs C** measures cue visibility, not architecture value. Say so explicitly.
- **B vs C** is the only pair that speaks to self-attention on product language. Interpret against the §7 diagnostic table: near-prevalence on both curves with a gradient-boosting baseline also near prevalence means *no signal*, not underfitting.
- Degenerate match features: what we observed.
- Metric dispersion across seeds vs the size of the deltas being claimed.
- Limitations from [DESIGN.md](DESIGN.md) §10.

## Cell group 11 — Architecture scale-up (required, not optional)

The PDF's design question 2 asks for a small base architecture *and then* increasing complexity within available compute. This is graded, so it runs.

1. Markdown: what is being scaled and why, with the compute budget.
2. Code: at least two scale-up configurations on **Arm A and Arm B**, same splits and seeds — e.g. `d_model=96` and `num_encoder_layers=3`.
3. Code: extend the comparison table with valid metrics for each configuration.

**Test-set rule (no exceptions):** scale-up runs are selected on **valid** only. Test is evaluated once per arm, for the single configuration chosen as that arm's final model. Exploratory configurations report valid metrics and leave the `test` column empty. This matches [DESIGN.md](DESIGN.md) §4 — there is no exploratory-labelling exemption.

## Cell group 12 — Exercise 3: personalization (markdown)

Theoretical answer for the required third exercise ([ASSIGNMENT_MAPPING.md](ASSIGNMENT_MAPPING.md) §Exercise 3). Notebook carries a short version; the slide is the deliverable. Must state that this CSV has **no user identifier**, so the answer is a design sketch, not an experiment.

---

## Runtime file layout

Not committed (gitignored):

- `best_arm_a.pt`, `best_arm_b.pt`, `best_arm_c.pt` — must match [DESIGN.md](DESIGN.md) §8
- matplotlib figures inline

Repo layout:

- `notebooks/btr_transformer.ipynb` (groups 0–7 implemented)
- `data/raw/supermarket_products.csv` (gitignored if large)
- these `docs/` and `adr/` files

## Implementation notes (groups 0–7)

Ordering corrections so the notebook is executable; not new design decisions.

- Tokenizer is loaded in group 1, not 4.4. Group 2.7 needs `tokenizer.sep_token` to build `text_full` / `text_stripped`.
- Cue-stripping is defined in group 2, not 9.2. The TF-IDF verification probe stays in group 9.
- `split_queries(...)` is defined as a helper in group 1 and called again in group 3. The cue audit (1.6) needs a grouped split to fit its probes.
- `query_id` is dropped from the feature frames in 3.3 as specified, but an aligned `np.ndarray` of query ids is retained per split so `per_query_metrics` (7.2) is implementable.
- Cue stripping is an exact set-membership deletion (19 description templates + title parenthetical), not a fuzzy regex, because the CSV supports it.
