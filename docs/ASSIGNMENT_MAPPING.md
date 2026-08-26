# Assignment mapping

Source: [`DeepLearningTP0.pdf`](DeepLearningTP0.pdf) — *Trabajo Práctico 1, 73.69 Large Language Models, 2026*.

## Goal (PDF §1)

Solve a concrete prediction problem **and** implement a Transformer so the architecture is actually understood. Transfer learning with a frozen BERT encoder would miss the pedagogical target. The Transformer in this project is a **small encoder trained from scratch** on product text.

## Problem (PDF §2)

Predict **Buy Through Rate (BTR)** for supermarket e-commerce: purchased products / impressed products on a search-results page. Operationally this is **row-level binary classification** of `bought` given an impression (one catalog product shown for a search).

Predicted probability \(\hat{p}(bought=1 \mid impression)\) is the product-level BTR contribution. The PDF says **not** to pick a classification threshold.

## Exercise 1 — Formulation and EDA

Must justify:

1. What is predicted (target).
2. Data characteristics (support, distributions, quality).
3. Which features are used.
4. Preprocessing per feature (encoding, scaling, tokenization).

Covered in [DATA_PROFILE.md](DATA_PROFILE.md) and [DESIGN.md](DESIGN.md) §§1–3. The notebook will later include a compact EDA section that **cites these facts**, not a second undocumented analysis.

## Exercise 2 — System

Must include **at least one Transformer**, placed and justified. Design questions from the PDF:

| PDF question | Decision (summary) | Detail |
| --- | --- | --- |
| How to partition data | Train / valid / test, **grouped by `query_id`** | [adr/0005](../adr/0005-grouped-splits.md) |
| How to experiment on architecture | Tiny hybrid first (`d_model < 100`), then optional scale-up | [adr/0006](../adr/0006-hybrid-transformer-mlp.md) |
| How to evaluate | ROC-AUC + **PR-AUC**; early stop on valid PR-AUC | [adr/0007](../adr/0007-training-ablation.md) |

Grading emphasis (PDF aclaración): design justification, **module alternatives**, and **ablation**. The tabular-only flag exists specifically for that comparison.

## Presentation (~25–30 min)

Problem → implementation decisions → results → challenges → conclusions. ADRs are the decision appendix; slides should be a subset, not a dump of this folder.

## Delivery (PDF §4)

Repository with `README.md`, commit hash, and slides via Campus. Execution environment requested for the practical part: **Google Colab** ([adr/0008](../adr/0008-delivery-colab-notebook.md)).
