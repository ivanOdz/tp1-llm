# ADR 0006 — Hybrid: light Transformer on text + MLP on concat

- **Status:** Accepted
- **Date:** 2026-08-26

## Context

The system must contain a Transformer. Compute must stay modest; the PDF suggests starting with `d_model < 100` and then increasing complexity within available compute. Text is short (~40 whitespace tokens, ≤ 74 WordPiece).

The original version of this ADR assumed "tabular signal is likely strong (price, category, brand)". Measurement refuted that: the tabular block reaches 0.551 ROC-AUC under gradient boosting while the text reaches 0.955, driven by a behavioural cue ([EDA.md](../docs/EDA.md) §Text, [adr/0009](0009-text-behavioural-cue.md)). Placing the Transformer on text remains correct, but for a different reason than originally argued.

## Decision

Place the Transformer **only on tokenized product text**:

`Embedding → sinusoidal PE → TransformerEncoder (2 layers, 4 heads, d_model=64) → masked mean pool over all non-pad positions → concat(tabular, 62) → MLP (126 → 128 → 64 → 1 logit)`.

Use `nn.TransformerEncoder` (standard library) rather than a pretrained Hugging Face encoder. That still requires implementing embeddings, masks, pooling, and the hybrid head — enough to explain attention in the presentation without hiding it inside BERT.

Keep the encoder small: `d_model=64 < 100`, 2 layers, 4 heads, FFN 128.

## Consequences

- Parameter count is dominated by `vocab × d_model` (BERT vocab 30 522 × 64 ≈ 1.95M) plus a tiny encoder/MLP. Feasible on a laptop GPU or CPU for this dataset size.
- **Capacity asymmetry to disclose:** the hybrid arm carries ~1.95M+ parameters against ~16k for the tabular-only arm, and only ~7 000 training rows with ~910 positives. Most of the embedding table receives no gradient. This is acceptable — unused rows are never selected at inference on a fixed catalog vocabulary — but it means arm comparisons vary in capacity as well as in inputs, and the write-up must say so.
- Arm A's strong result is largely one behavioural token, so the encoder is doing token lookup rather than compositional language modelling. Arm B exists to measure what is left ([adr/0009](0009-text-behavioural-cue.md)).

## Alternatives

- Transformer over tabular tokens (feature-wise): possible but less aligned with “LLM / text” course framing.
- Cross-attention from query filters to product text: filters are degenerate here; extra complexity.
- Full BERT fine-tune: heavier, weaker architecture learning, `d_model=768` violates the spirit of the size hint.
