# BTR-Transformer

University TP1 (73.69 Large Language Models, 2026): predict supermarket e-commerce **Buy Through Rate** via impression-level `bought`, using a **small Transformer encoder trained from scratch** on product text plus an MLP on tabular features.

**Phase:** notebook groups 0–14 implemented and executed end to end — EDA, baselines, the three-arm ablation, a 3 × 2 architecture grid, the single test evaluation, and the written analysis.

## Read first

- [docs/README.md](docs/README.md) — index
- [docs/EDA.md](docs/EDA.md) — measured data facts, **including the behavioural cue that carries almost all the signal**
- [docs/DESIGN.md](docs/DESIGN.md) — end-to-end pipeline
- [docs/OPEN_DECISIONS.md](docs/OPEN_DECISIONS.md) — what is resolved and what still needs a ruling
- [docs/NOTEBOOK_SPEC.md](docs/NOTEBOOK_SPEC.md) — notebook cell plan
- [adr/](adr/) — why each decision was made

Assignment PDF: [docs/DeepLearningTP0.pdf](docs/DeepLearningTP0.pdf) (titled *Trabajo Práctico 1* despite the filename).

## Data

- File: `data/raw/supermarket_products.csv` (gitignored)
- 10 000 impressions, 2 012 queries, 13.01% `bought=True`
- Only missing values: `allergens` (44.55%)

## The headline finding

The dataset **encodes the target in the catalog text**. A one-hot of a single behavioural phrase in `title` — "Best Seller", "most repurchased", "rarely reordered" — predicts `bought` at test ROC-AUC **0.957**, while the entire engineered tabular block reaches **0.595** under gradient boosting.

The full hybrid reaches **0.9668 ± 0.0029**, i.e. **+0.0098 over one token**. And its own checkpoint, re-scored with the cue removed at inference, falls to **0.5044 ± 0.0156 — chance** — without falling back on the tabular features it is still given.

The project reports this deliberately rather than presenting it as an architecture result. See [adr/0009](adr/0009-text-behavioural-cue.md) and notebook group 13.

## Artifact

[`notebooks/btr_transformer.ipynb`](notebooks/btr_transformer.ipynb) — setup → EDA with cue audit → feature engineering → stratified grouped splits → preprocessing → hybrid model → training helpers → six reference baselines → three ablation arms → 3 × 2 architecture grid → single test evaluation → analysis → Exercise 3.

Run it with `BTR_FAST=1` to validate the whole thing end to end in ~4 minutes at a smoke budget (1 seed, 2 epochs); its numbers are deliberately not reportable.

Three arms, identical splits and seeds:

| Arm | Text input | Question |
| --- | --- | --- |
| A | full text | Upper bound with the cue visible |
| B | cue-stripped text | Does the encoder learn from product language proper? |
| C | none (tabular-only) | What do the non-text features carry? |

**A vs C** measures cue visibility. **B vs C** is the only pair that speaks to self-attention on language.

## Architecture grid

A full 3 × 2 factorial over the two capacity axes, so encoder size and head width can be read independently instead of confounded in one "bigger model" run ([adr/0010](adr/0010-architecture-grid.md)):

| Transformer | `d_model` / heads / layers / FFN | | MLP head | Hidden |
| --- | --- | --- | --- | --- |
| `T1-small` | 32 / 2 / 2 / 64 | | `M1-base` | 128 → 64 |
| `T2-base` | 64 / 4 / 2 / 128 | | `M2-wide` | 256 → 128 |
| `T3-large` | 96 / 4 / 3 / 192 | | | |

The grid reports **validation only**. One configuration per arm is promoted to the single test evaluation, enforced in code by a guard that raises on a second test look.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Deferred

A modular `src/` package and a parquet pipeline are out of scope for v1. See [adr/0008](adr/0008-delivery-notebook.md).
