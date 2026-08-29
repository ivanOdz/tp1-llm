# BTR-Transformer

University TP1 (73.69 Large Language Models, 2026): predict supermarket e-commerce **Buy Through Rate** via impression-level `bought`, using a **small Transformer encoder trained from scratch** on product text plus an MLP on tabular features.

**Phase:** design documentation, audited and reconciled. No notebook or training code yet.

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

The dataset **encodes the target in the catalog text**. A single behavioural phrase in `title` — "Best Seller", "most repurchased", "rarely reordered" — predicts `bought` at ROC-AUC **0.960** on a grouped holdout, while the entire engineered tabular block reaches only **0.551** under gradient boosting.

The project reports this deliberately rather than presenting it as an architecture result. See [adr/0009](adr/0009-text-behavioural-cue.md).

## Planned artifact

`notebooks/btr_transformer.ipynb` — setup → EDA with cue audit → feature engineering → stratified grouped splits → preprocessing → hybrid model → baselines → three ablation arms → scale-up → discussion.

Three arms, identical splits and seeds:

| Arm | Text input | Question |
| --- | --- | --- |
| A | full text | Upper bound with the cue visible |
| B | cue-stripped text | Does the encoder learn from product language proper? |
| C | none (tabular-only) | What do the non-text features carry? |

**A vs C** measures cue visibility. **B vs C** is the only pair that speaks to self-attention on language.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Deferred

A modular `src/` package and a parquet pipeline are out of scope for v1. See [adr/0008](adr/0008-delivery-notebook.md).
