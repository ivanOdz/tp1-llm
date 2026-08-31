# ADR 0010 — Architecture scale-up is a 3 × 2 factorial grid, not a ladder

- **Status:** Accepted
- **Date:** 2026-08-30
- **Refines:** [ADR 0006](0006-hybrid-transformer-mlp.md) (base architecture), D19 (scale-up is required)

## Context

D19 settled that the architecture scale-up is in scope and graded: the PDF's second design question asks for a small base architecture *and then* increasing complexity within the available compute. It did not settle **what** gets scaled.

The obvious reading — one bigger Transformer as a second rung — confounds two capacity axes. The system has two independently sizeable components:

1. the text encoder (`d_model`, heads, layers, FFN), which only exists in arms A and B;
2. the MLP head, which exists in **all three arms** and is the only learned component of arm C.

A single "base → large" ladder cannot say which of the two drove any change, and on this dataset that distinction matters more than usual: [ADR 0009](0009-text-behavioural-cue.md) establishes that the tabular block is near-noise and the text signal is one token. If a larger model helps, the interesting question is *which half* of the model absorbed it.

Compute measured on the target machine (CPU-only, 12 cores, this dataset): 4 s/epoch at `d_model=32`, 8 s at 64, 16 s at 96. A full factorial is affordable.

## Decision

Run a **full 3 × 2 factorial** over encoder size and MLP head width, on arms A and B, at fixed splits and seeds.

| Transformer | `d_model` | heads | layers | FFN |
| --- | ---: | ---: | ---: | ---: |
| `T1-small` | 32 | 2 | 2 | 64 |
| `T2-base` | 64 | 4 | 2 | 128 |
| `T3-large` | 96 | 4 | 3 | 192 |

| MLP head | Hidden | Dropout |
| --- | --- | ---: |
| `M1-base` | 128 → 64 | 0.2 |
| `M2-wide` | 256 → 128 | 0.2 |

`T2-base` + `M1-base` is the base configuration of D4, so the three-arm ablation of [ADR 0009](0009-text-behavioural-cue.md) is one cell of this grid rather than a separate experiment. All three encoder sizes respect the PDF's `d_model < 100` hint.

**Dropout is held at 0.2 across both heads** so the MLP axis varies width only. Varying width and regularization together would reproduce, one level down, exactly the confound this ADR exists to avoid.

Arm C has no encoder, so only the MLP axis applies; it is run over the two heads alone and its grid key labels the encoder slot `n/a`.

**Reporting:** the grid reports **validation** metrics only. Each arm's final configuration is the one with the highest mean validation PR-AUC; only that configuration is retrained and evaluated on test, once. This is enforced by a `_TEST_EVALUATED` guard inside `run_arm` that raises on a second test evaluation for the same arm name, rather than by author discipline.

**Interpretation:** main effects are read as marginal means over the other factor, and compared against the mean within-configuration seed sd. A factor whose spread does not exceed that sd is reported as *unresolved*, not as an improvement.

## Consequences

- Six configurations per text arm plus two for arm C, at 5 seeds each: ~90 training runs, roughly 2 hours on the target CPU. Acceptable, and the run is scriptable in one pass.
- The three-arm ablation and the scale-up share one results cache, so the base configuration is trained once and reused by both. No duplicated compute.
- Exercise 2's "comparación de alternativas de los distintos módulos" is satisfied by a design where the two modules are varied *separately*, which is a stronger reading of the requirement than a single ladder.
- Selecting on validation over six configurations is a mild multiple-comparison risk: with differences near the noise floor, the argmax is partly noise. The write-up must report the spread across the grid, not only the winner, so the selection's fragility is visible.
- A `BTR_FAST=1` environment switch shrinks the grid to one seed and two epochs so the notebook can be validated end-to-end in minutes. It changes no design decision and prints a warning that its results are not reportable.

## Alternatives

- **Single base → large ladder (the literal D19 reading):** cheaper, but confounds the two capacity axes and cannot attribute a change to the encoder or the head. Rejected.
- **Vary the encoder only, holding the MLP fixed:** defensible, but leaves arm C with no scale-up at all, so the tabular arm's capacity is never questioned even though it is the reference the language claim rests on. Rejected.
- **Random or Bayesian search over a wider space:** better use of compute for pure performance, worse for a graded explanation of *which module matters*. Rejected — the assignment grades the reasoning, not the last decimal.
- **Also varying dropout, learning rate or `max_length`:** each adds a factor and multiplies the run count; D18 already locked the optimizer settings after a stability analysis, and D10 locked `max_length` on measured token lengths. Out of scope.
