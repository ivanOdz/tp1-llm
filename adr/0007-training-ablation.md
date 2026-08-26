# ADR 0007 — PR-AUC early stopping and tabular ablation

- **Status:** Accepted
- **Date:** 2026-08-26

## Context

Positive rate is 13%. ROC-AUC can look strong while the model is weak on the purchase class. The PDF names PR-AUC and ROC-AUC and asks for overfitting/underfitting control and an **ablation**.

## Decision

1. Loss: `BCEWithLogitsLoss`.
2. Log **ROC-AUC** and **PR-AUC** on valid each epoch (sklearn).
3. Early stopping **maximizes valid PR-AUC** (patience 5, restore best weights).
4. Ablation framework: boolean `use_text_encoder`. `False` trains the same MLP on tabular features only, **same splits and seeds**.
5. Report a comparison table (params, best epoch, valid/test AUCs).

Do not introduce a BTR probability threshold.

## Consequences

- Hybrid and tabular-only are comparable.
- Presentation has a clear “does the Transformer module help?” answer.
- Underfitting: both curves low vs prevalence baseline (~0.13 PR-AUC). Overfitting: train PR-AUC up, valid down before the patience cut.

## Alternatives

- Early stop on ROC-AUC or loss: worse match to imbalance and to the PDF’s BTR-oriented PR-AUC hint.
- Accuracy / F1 at 0.5: PDF says not to complicate with a threshold.
