# BTR-Transformer

University TP1 (73.69 LLM, 2026): predict supermarket e-commerce **Buy Through Rate** via impression-level `bought`, with a **small Transformer encoder** on product text plus an MLP on tabular features.

**Phase:** design documentation only. No notebook or training code yet.

## Read first

- [docs/README.md](docs/README.md) — index
- [docs/DESIGN.md](docs/DESIGN.md) — pipeline
- [docs/OPEN_DECISIONS.md](docs/OPEN_DECISIONS.md) — change these before implementation
- [docs/NOTEBOOK_SPEC.md](docs/NOTEBOOK_SPEC.md) — future Colab cell plan
- [adr/](adr/) — why each decision was made

Assignment PDF: [docs/DeepLearningTP0.pdf](docs/DeepLearningTP0.pdf).

## Data

- File: `data/raw/supermarket_products.csv` (gitignored)
- 10 000 impressions, 2 012 queries, 13.01% `bought=True`

## Planned artifact (not written)

A single Google Colab notebook implementing setup → feature engineering → grouped splits → preprocess → hybrid model → train/eval → tabular ablation.

## Deferred

The previous `src/` + parquet tree is **not** the v1 delivery. See [adr/0008](adr/0008-delivery-colab-notebook.md).
