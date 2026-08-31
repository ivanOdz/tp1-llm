# Notebook spec

Artifact: [`notebooks/btr_transformer.ipynb`](../notebooks/btr_transformer.ipynb). Groups 0–14 are implemented and executed. Environment: **local Jupyter**, GPU used if available, CPU adequate at this scale ([DESIGN.md](DESIGN.md) §8).

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
2. Code: six baselines on the locked split — prevalence constant, logistic regression and gradient boosting on the tabular block, title-cue one-hot → logistic regression, TF-IDF on full text, TF-IDF on cue-stripped text. All fitted transforms fit on train only.
3. Output: baseline table with valid/test ROC-AUC, PR-AUC and the per-query diagnostics.

The last two rows are the non-neural analogues of arms A and B, so the neural arms have a bag-of-words reference and not only a tabular one.

## Cell group 9 — The three arms at the base configuration

1. Markdown: Arm A (full text) / B (cue-stripped) / C (tabular-only) and the question each answers.
2. Code: cue-stripping verification probe — top ± TF-IDF n-grams before and after stripping; assert no behavioural n-gram survives in the top-30 ([adr/0009](../adr/0009-text-behavioural-cue.md) step 3).
3. Code: run the three arms at `T2-base` + `M1-base` over `ARM_SEEDS`, valid metrics only, into the shared `RESULTS` cache.
4. Code: 3×3 panel of **train and valid** loss / ROC-AUC / PR-AUC vs epoch, with the prevalence line drawn on the PR-AUC panels.

Test is **not** touched here.

## Cell group 10 — Architecture grid (3 Transformer × 2 MLP)

Required by D19; shaped by [adr/0010](../adr/0010-architecture-grid.md).

1. Markdown: the two factors, the configurations, and the valid-only selection rule.
2. Code: full factorial over `T1-small` / `T2-base` / `T3-large` × `M1-base` / `M2-wide` on arms A and B; arm C over the MLP axis alone. The base cell is reused from group 9 rather than retrained.
3. Code: grid table (params, valid ROC/PR mean ± sd, best epoch, seconds).
4. Code: per-arm heatmap of valid PR-AUC, plus **main effects** — marginal means per factor compared against the mean within-configuration seed sd, so a factor that does not clear seed noise is reported as unresolved.

## Cell group 11 — Final configuration and the single test evaluation

1. Markdown: selection rule and the A→B cross-evaluation.
2. Code: pick each arm's highest mean valid PR-AUC configuration, retrain over `ARM_SEEDS`, evaluate test **once** (guarded by `_TEST_EVALUATED`), and save `best_arm_{a,b,c}.pt`.
3. Code: Arm A's final checkpoint additionally scored on cue-stripped test text — the distribution-shift probe, at no extra training cost — reported against the tabular ceiling and the prevalence floor.

## Cell group 12 — Unified comparison table

1. Markdown: what the table contains and why the per-query diagnostics are there (D23).
2. Code: baselines + final arms in one frame; write `outputs/results.json`.
3. Code: effect sizes against dispersion — A vs C and B vs C deltas compared against 2× the seed sd and against the EDA split-level noise floor.

## Cell group 13 — Discussion (markdown)

- **A vs C** measures cue visibility, not architecture value. Say so explicitly.
- **B vs C** is the only pair that speaks to self-attention on product language. Interpret against the §7 diagnostic: near-prevalence on both curves *with a gradient-boosting baseline also near prevalence* means no signal, not underfitting.
- What the architecture grid showed on each axis, and whether either cleared seed noise.
- What the A→B collapse implies about what the encoder actually learned.
- Degenerate match features; metric dispersion vs the size of the deltas claimed; limitations from [DESIGN.md](DESIGN.md) §10.

## Cell group 14 — Exercise 3: personalization (markdown)

Theoretical answer for the required third exercise ([ASSIGNMENT_MAPPING.md](ASSIGNMENT_MAPPING.md) §Exercise 3). Must state that this CSV has **no user identifier**, so the answer is a design sketch, not an experiment. Covers: what BTR becomes, data required, the two places personalization enters this architecture, the new splitting/leakage risk, cold start, and what the behavioural cue implies. The slide is the deliverable; the notebook carries the long form.

---

## Runtime file layout

Not committed (gitignored):

- `best_arm_a.pt`, `best_arm_b.pt`, `best_arm_c.pt` — must match [DESIGN.md](DESIGN.md) §8
- matplotlib figures inline

Repo layout:

- `notebooks/btr_transformer.ipynb` (groups 0–14)
- `data/raw/supermarket_products.csv` (gitignored if large)
- these `docs/` and `adr/` files

## Implementation notes

Ordering corrections so the notebook is executable; not new design decisions.

- Tokenizer is loaded in group 1, not 4.4. Group 2.7 needs `tokenizer.sep_token` to build `text_full` / `text_stripped`.
- Cue-stripping is defined in group 2, not 9.2. The TF-IDF verification probe stays in group 9.
- `split_queries(...)` is defined as a helper in group 1 and called again in group 3. The cue audit (1.6) needs a grouped split to fit its probes.
- `query_id` is dropped from the feature frames in 3.3 as specified, but an aligned `np.ndarray` of query ids is retained per split so `per_query_metrics` (7.2) is implementable.
- Cue stripping is an exact set-membership deletion (19 description templates + title parenthetical), not a fuzzy regex, because the CSV supports it.
- `BTR_FAST=1` shrinks the run to 1 seed / 2 epochs for an end-to-end smoke test. It changes no design decision and warns that its numbers are not reportable.
- `run_arm` carries the `_TEST_EVALUATED` guard: a second test evaluation for the same arm name raises rather than silently reporting a second look.
- The A→B cross-evaluation reuses Arm A's final checkpoint instead of training a fourth model, so it adds no compute and no extra test exposure.
