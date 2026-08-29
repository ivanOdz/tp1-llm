# ADR 0007 — PR-AUC early stopping and tabular ablation

- **Status:** Accepted
- **Date:** 2026-08-26
- **Updated:** 2026-08-29 — two arms → three ([adr/0009](0009-text-behavioural-cue.md)); baselines and seed repeats added; no-signal case added to the diagnostic

## Context

Positive rate is 13%. ROC-AUC can look strong while the model is weak on the purchase class. The PDF names PR-AUC and ROC-AUC and asks for overfitting/underfitting control, comparison of module alternatives, and an **ablation**.

Measurement after the original version of this ADR showed the tabular block is near-noise (gradient boosting: 0.551 ROC-AUC) while a single behavioural token in the product text reaches 0.960. A two-arm ablation would therefore report a large, real, and **misattributed** effect.

## Decision

1. Loss: `BCEWithLogitsLoss`.
2. Log **ROC-AUC** and **PR-AUC** on **train and valid** each epoch (sklearn). Train curves are mandatory, not optional — the diagnostic below cannot be read without them.
3. Early stopping **maximizes valid PR-AUC** (patience 5, restore best weights).
4. Ablation framework: **three arms**, identical splits, seeds, tabular tensor and training loop — A (full text), B (cue-stripped text), C (tabular-only, `use_text_encoder=False`). See [adr/0009](0009-text-behavioural-cue.md).
5. **Non-neural reference baselines are required**, in the same table: prevalence predictor, logistic regression and gradient boosting on the tabular block, and the title-cue one-hot model.
6. Every arm runs over **5 seeds**; report mean ± sd. The measured single-split noise floor is ±0.016 PR-AUC, so unreplicated deltas below ~0.05 are not claimable.
7. Report a comparison table: params, best epoch, valid/test ROC-AUC and PR-AUC, per-query AP and recall@1.

Do not introduce a BTR probability threshold.

## Consequences

- All three arms are mutually comparable, and both the architecture question and the cue question have separate answers.
- **A vs C** measures cue visibility. **B vs C** is the only pair that supports a claim about self-attention on product language. Conflating them would misattribute a leakage finding to the architecture.
- The diagnostic has three cases, not two:

| Train PR-AUC | Valid PR-AUC | Diagnosis |
| --- | --- | --- |
| Low (~prevalence) | Low (~prevalence) | Underfitting **or no signal** — disambiguate with the gradient-boosting baseline on the same inputs |
| High | Low | Overfitting |
| Rising at patience boundary | Rising | Undertrained |

  The earlier two-case version treated the first row as underfitting outright and prescribed scaling `d_model`. On this dataset that is the wrong action for Arms B and C, whose inputs are near-noise.

## Alternatives

- Early stop on ROC-AUC or loss: worse match to imbalance and to the PDF’s BTR-oriented PR-AUC hint.
- Accuracy / F1 at 0.5: PDF says a threshold is not necessary.
- Two arms only (A vs C): reports a large gap whose cause is misattributed. Rejected in [adr/0009](0009-text-behavioural-cue.md).
- Single run per arm: cheaper, but cannot support any comparative claim at the measured noise floor.
