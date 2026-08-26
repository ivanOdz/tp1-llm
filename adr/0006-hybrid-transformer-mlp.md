# ADR 0006 — Hybrid: light Transformer on text + MLP on concat

- **Status:** Accepted
- **Date:** 2026-08-26

## Context

The system must contain a Transformer. Compute must stay Colab-friendly; the PDF says start with `d_model < 100`. Text is short (~40 words). Tabular signal is likely strong (price, category, brand).

## Decision

Place the Transformer **only on tokenized product text**:

`Embedding → sinusoidal PE → TransformerEncoder (2 layers, 4 heads, d_model=64) → masked mean pool → concat(tabular) → MLP → 1 logit`.

Use `nn.TransformerEncoder` (standard library) rather than a pretrained Hugging Face encoder. That still requires implementing embeddings, masks, pooling, and the hybrid head — enough to explain attention in the presentation without hiding it inside BERT.

Keep the encoder small: `d_model=64 < 100`, 2 layers, 4 heads, FFN 128.

## Consequences

- Parameter count is dominated by `vocab × d_model` (BERT vocab ~30k × 64 ≈ 1.9M) plus a tiny encoder/MLP. Acceptable on Colab.
- If descriptions are templates, ablation may show little or no gain from text ([adr/0007](0007-training-ablation.md)). That is still a valid experimental result.

## Alternatives

- Transformer over tabular tokens (feature-wise): possible but less aligned with “LLM / text” course framing.
- Cross-attention from query filters to product text: filters are degenerate here; extra complexity.
- Full BERT fine-tune: heavier, weaker architecture learning, `d_model=768` violates the spirit of the size hint.
