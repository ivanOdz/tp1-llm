# Open decisions

Edit this file before asking for the implementation plan. Defaults are what [DESIGN.md](DESIGN.md) currently assumes.

Legend: **Recommended** is the default if you say nothing. **Resolved** means the decision was settled by measurement or by an explicit ruling and is recorded in the locked list below.

| ID | Topic | Recommended | Alternatives | Notes |
| --- | --- | --- | --- | --- |
| D1 | Zero-variance match flags | Compute, then **drop** `is_category_match` and `is_storage_match` after logging that variance is 0 | Keep them anyway; skip creating them | They are always 1 on this CSV. Keeping them adds noise in the MLP. |
| D2 | Tokenizer | `bert-base-uncased` AutoTokenizer, **random** `nn.Embedding` | Train a small WordPiece on train text; whitespace / custom vocab | HF tokenizer is reproducible and matches the `transformers` install. Do not load BERT encoder weights. |
| D3 | Split ratios | **70 / 15 / 15** of `query_id` | 80/10/10 | ~302 queries in valid/test; ~195 positive rows — enough for PR-AUC, but see D16 on dispersion. |
| D4 | Encoder size | `d_model=64`, 4 heads, 2 layers, FFN 128 as the **base**; scale-up is required, not optional | 32/2/2 (cheaper base); 96/4/2 and 3 layers as the scale-up rung | PDF: start with `d_model < 100`, *then* increase complexity. See D19. |
| D5 | Positional encoding | **Sinusoidal** (Vaswani) | Learned `nn.Embedding(max_length, d_model)` | Sinusoidal is easier to explain in the presentation. Optional empirical comparison at fixed inputs would strengthen the "module alternatives" requirement. |
| D6 | Extra ablation | Superseded by the three-arm design (D20) | Also a text-only MLP as a fourth arm | See [adr/0009](../adr/0009-text-behavioural-cue.md). |
| D7 | Batch size | **64** | 32 if GPU memory is tight; 128 on a larger GPU | Sequence length 80 is small; 64 is fine. |
| D8 | `nutrition_score == 0` | **Leave as 0** | Missing indicator + impute | 1 244 zeros; unknown if sentinel. Low stakes — the whole tabular block is near-noise. |
| D9 | Class imbalance in loss | **No `pos_weight` first** | `pos_weight = n_neg/n_pos` on train | PR-AUC + early stopping may be enough; `pos_weight` can be a one-line experiment. |
| D10 | Max token length | **Resolved: 80** | — | Measured WordPiece max is **74** (mean 55.6, p95 64) including `[CLS]`/`[SEP]` and two field separators. 80 truncates zero rows; 96 wastes 42% of positions; 64 truncates 4.54%. |
| D11 | Delivery | **One Jupyter notebook, run locally** | Notebook + `src/` package | Package is deferred. See D21 for the environment ruling. |
| D12 | Move tensors to GPU | **In the training loop, per batch** | Inside `Dataset.__getitem__` | Batch-level is the correct interpretation. |
| D13 | Scale cyclical / binary | **Do not scale** sin/cos, `relative_price_position` (or match flags) | Put them through `StandardScaler` | Already bounded. Scaler columns are the four continuous numerics only. |
| D14 | Text concat separator | Tokenizer `sep_token` between fields | Plain space | SEP makes field boundaries visible to attention. Separator tokens **do** participate in mean pooling (D22). |
| D15 | Split stratification | **Resolved: stratify** query ids by "has ≥1 purchase", then assign rows by query | Unstratified random assignment of query ids | Previously worded "ungrouped random on query ids", which read as a row-wise split and contradicted [adr/0005](../adr/0005-grouped-splits.md). The split is **always grouped**; the open question was only stratification. Held-out prevalence varies 0.118–0.153 unstratified, and PR-AUC's baseline *is* the prevalence, so valid/test comparisons were not apples-to-apples. |
| D16 | Run repeats and dispersion | **Resolved: 5 seeds per arm, report mean ± sd** | Single run | Measured noise floor is ±0.016 PR-AUC on a 15% holdout; differences below ~0.05 are unresolvable from one run. Seed repeats keep the same train/valid/test protocol, so the "no k-fold" lock stands. |
| D17 | `volume` transform | **Resolved: `log1p` before impute + scale** | Raw, or quantile transform | Raw: skew 2.97, max \(\|z\|\) 9.61, 50 rows beyond \(\|z\|>5\). `log1p`: skew 0.44, max \(\|z\|\) 3.05. Other three numerics unchanged. |
| D18 | Optimizer stability | **Resolved: `lr=3e-4`, linear warmup over first 10% of steps, `clip_grad_norm_(1.0)`** | `lr=1e-3` with `norm_first=True`; `lr=1e-3` bare | `nn.TransformerEncoderLayer` is post-LN by default, the configuration that needs warmup. Bare `lr=1e-3` on post-LN is a known instability. |
| D19 | Architecture scale-up | **Resolved: in scope, required** | Optional stretch | PDF design question 2 asks for a small base *then* increasing complexity within available compute. Selection on valid only. |
| D20 | Behavioural cue in text | **Resolved: keep it, and add a cue-stripped arm** | Keep only; strip only; extract as a one-hot feature | The cue predicts `bought` at ROC-AUC 0.960 alone vs 0.551 for the whole tabular block. [adr/0009](../adr/0009-text-behavioural-cue.md). |
| D21 | Execution environment | **Resolved: local Jupyter**, pinned `requirements.txt`, no inline `pip install` | Colab; both with a guarded install cell | `DESIGN.md` §8 previously specified Colab against three other documents. |
| D22 | Special tokens in mean pooling | **Resolved: include all non-pad positions** (`[CLS]`, `[SEP]`, field separators) | Mask out special tokens | Separators are content-bearing under D14; excluding them needs a second mask for no clear gain. Must be stated explicitly in the notebook. |
| D23 | Per-query diagnostic | **Resolved: report mean per-query AP and recall@1** on queries with ≥1 purchase | Global metrics only | Diagnostic only — early stopping still uses global valid PR-AUC (locked). BTR is a per-SERP rate and 52.6% of queries have zero purchases. |
| D24 | Exercise 3 owner | **Resolved: drafted in notebook group 14** | — | Design sketch written (no user identifier in this CSV). Still needs a **slide** made from it. |
| D25 | What the scale-up varies | **Resolved: full 3 x 2 factorial** — encoder size (32/64/96) x MLP width (128-64 / 256-128) | Single base-to-large ladder; encoder axis only | A ladder confounds the two capacity axes; arm C has only the MLP axis. Dropout held at 0.2 so the head axis is width-only. [adr/0010](../adr/0010-architecture-grid.md). |

## Locked (do not reopen without a reason)

- Target is `bought`, not `cart`.
- `cart` is not a feature.
- `query_id` is not a model input.
- Raw `filter_*` columns are not fed to the model after engineering.
- Price-band signal is **`relative_price_position` only** (no `price_distance_*`); zero-width band → `0.5`, asserted to fire 0 times.
- Single grouped train / valid / test split, **stratified on has-≥1-purchase**. **No k-fold.** Dispersion comes from seed repeats (D16).
- Loss is `BCEWithLogitsLoss`.
- Reported metrics include ROC-AUC and PR-AUC, plus non-neural reference baselines ([DESIGN.md](DESIGN.md) §7.1).
- Early stopping monitors **validation PR-AUC**.
- Transformer is **from-scratch encoder**, not a pretrained LLM backbone.
- Preprocessors fit on **train only**. Cue stripping is a fixed rule, not a fitted transform.
- Test is evaluated **once per arm**, on that arm's final configuration. No exploratory exemption.
- Three arms: A (full text), B (cue-stripped text), C (tabular-only). Claims about self-attention on language rest on **B vs C**, never A vs C.
- Architecture grid reports **valid only**; one configuration per arm is promoted to the single test evaluation, enforced by a guard in `run_arm`.
