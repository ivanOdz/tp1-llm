# Assignment mapping

Source: [`DeepLearningTP0.pdf`](DeepLearningTP0.pdf) — *Trabajo Práctico 1, 73.69 Large Language Models, 2026*.

## Goal (PDF §1)

Solve a concrete prediction problem **and** implement a Transformer so the architecture is actually understood. Transfer learning with a frozen BERT encoder would miss the pedagogical target. The Transformer in this project is a **small encoder trained from scratch** on product text.

## Problem (PDF §2)

Predict **Buy Through Rate (BTR)** for supermarket e-commerce: purchased products / impressed products on a search-results page. Operationally this is **row-level binary classification** of `bought` given an impression (one catalog product shown for a search).

Predicted probability \(\hat{p}(bought=1 \mid impression)\) is the product-level BTR contribution. The PDF says a decision threshold is **not necessary** ("no es necesario complejizar el análisis con una definición de umbral"), so none is chosen.

## Exercise 1 — Formulation and EDA

Must justify:

1. What is predicted (target).
2. Data characteristics (support, distributions, quality).
3. Which features are used.
4. Preprocessing per feature (encoding, scaling, tokenization).

Covered in [EDA.md](EDA.md) and [DESIGN.md](DESIGN.md) §§1–3. The notebook will later include a compact EDA section that **cites these facts**, not a second undocumented analysis.

## Exercise 2 — System

Must include **at least one Transformer**, placed and justified. Design questions from the PDF:

| PDF question | Decision (summary) | Detail |
| --- | --- | --- |
| How to partition data | Train / valid / test, **grouped by `query_id`**, stratified on has-≥1-purchase | [adr/0005](../adr/0005-grouped-splits.md), D15 |
| How to experiment on architecture | Small hybrid base (`d_model=64 < 100`), **then a required scale-up rung** selected on valid | [adr/0006](../adr/0006-hybrid-transformer-mlp.md), D19 |
| How to evaluate | ROC-AUC + **PR-AUC**, non-neural reference baselines, 5 seeds with mean ± sd; early stop on valid PR-AUC | [adr/0007](../adr/0007-training-ablation.md), [DESIGN.md](DESIGN.md) §7 |

Grading emphasis (PDF aclaración): design justification, **comparison of module alternatives**, and **ablation**. Satisfied empirically by three arms (full text / cue-stripped text / tabular-only) plus four reference baselines, not by prose alone — see [adr/0009](../adr/0009-text-behavioural-cue.md).

**Note on the honest finding.** The dataset encodes the target in the catalog text: a one-hot of a single behavioural phrase in `title` predicts `bought` at test ROC-AUC 0.957, against 0.595 for the entire engineered tabular block. Measured outcome of the ablation: Arm B (cue-stripped) 0.5603 ± 0.0086 vs Arm C (no encoder) 0.5665 ± 0.0118 — the Transformer over genuine product language adds nothing. The results section reports this as a leakage-detection outcome rather than presenting the hybrid-vs-tabular gap as evidence that self-attention on language helps.

## Exercise 3 — Personalización (theoretical)

The PDF requires a brief theoretical answer (**one slide, under five minutes**): how would the solution change to include user personalization when defining BTR?

Constraint that shapes the answer: **this dataset has no user identifier.** There is no `user_id`, session key, or history column — `query_id` identifies a search, not a person ([EDA.md](EDA.md)). So the answer is necessarily a design sketch describing what data would be required and how the architecture would absorb it, not an experiment.

Points the slide should cover:

1. **Data that would be needed:** a stable user key, per-user purchase/view history, and a session context — none of which exist here.
2. **Where personalization enters the current architecture:** a user-history encoder producing a user vector, concatenated alongside `text_vec` and the tabular block; or cross-attention from a user-history sequence to the product-text tokens, which is the more Transformer-native option and reuses the encoder already built.
3. **What BTR becomes:** \(P(bought \mid impression, user)\) instead of \(P(bought \mid impression)\) — a per-user rate, so the business metric shifts from "best products overall" to "best products for this user".
4. **New leakage and splitting risks:** splits would need grouping by **user** as well as by query, or a temporal cut, otherwise a user's future purchases leak into their training history. This is the same class of error that [adr/0005](../adr/0005-grouped-splits.md) addresses for queries.
5. **Cold start:** new users and new products both need a fallback to the non-personalized model.

Status: **unassigned** (D24).

## Presentation (~25–30 min)

Problem → implementation decisions → results → challenges → conclusions. ADRs are the decision appendix; slides should be a subset, not a dump of this folder.

## Delivery (PDF §4)

The PDF requires: a repository with `README.md`, **the commit hash**, and the presentation, submitted via Campus. It does **not** prescribe the form of the practical artifact — a single Jupyter notebook is our choice ([adr/0008](../adr/0008-delivery-notebook.md)).

Checklist:

- [x] Git repository initialized and at least one commit made — the commit hash is a hard requirement and cannot be produced without it.
- [x] `README.md` current (phase, how to run, links to `docs/` and `adr/`).
- [x] Pinned `requirements.txt` so the notebook is reproducible.
- [x] `notebooks/btr_transformer.ipynb` runs top to bottom — executed end to end on 2026-08-30, 38/38 code cells with stored outputs, ~2h05 on CPU.
- [ ] **Slides** covering problem → decisions → results → challenges → conclusions, plus the Exercise 3 slide. `slides/` is still empty; this is the only remaining hard deliverable.

Run artifacts: `outputs/results.json` (every configuration's per-seed rows and epoch histories), `outputs/figures/*.png`, `best_arm_{a,b,c}.pt` (gitignored).
