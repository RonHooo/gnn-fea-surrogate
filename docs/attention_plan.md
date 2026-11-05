# Attention-based surrogate: plan and predictions (written before any training run)

Written 2026-11-04, before the first training run of the attention model. Committed to git so the
predictions can be compared with the results later. The predictions are not edited after the results.

## Questions

1. With the same data, split, inputs and outputs, how do the field errors (displacement, von Mises)
   and the scalar errors (f1, life, peak stress) of a slice-attention model compare with the GNN and
   the GP?
2. With the mesh only (no design variables), the GNN without a global node failed on displacement
   (98 % error). Can attention, where every node can reach every other node in one layer, recover
   this?
3. On the shifted family (width, height, hole positions changed), does attention extrapolate better
   than the GNN (von Mises field error 20 %)?

## Fixed setup

* Data, split (`split_dataset(seed=0)`: 280 / 60 / 60, OOD 40, shifted 30), node inputs, targets
  (stress components), normalisation, loss, optimiser (AdamW, weight decay 1e-5, 2-epoch warm-up +
  cosine), gradient clipping, batch size 4 and **50 epochs** are the same as the final GNN
  (`results/runs/gnn_stress`). The comparison uses the same number of epochs, not the same wall-clock
  time.
* Main model (A0), chosen before training: hidden 96, 6 blocks, 4 heads, 32 slices, FFN ratio 2,
  6 Fourier frequencies, 4 pooling heads. 641,302 parameters (GNN: 530,956).
* Learning rate: the only tuned hyper-parameter. Candidates 5e-4, 1e-3, 2e-3 with seed 0; the one with
  the lowest best-epoch validation loss is used for all attention runs. The GNN used 1e-3 without
  tuning.
* Seeds 0, 1, 2 for the attention model and the final GNN (the GNN had only seed 0).
* The test, OOD and shifted sets are evaluated once, after all runs are finished.
* A difference between two models is called real only if the difference of the seed means is larger
  than the larger of the two seed standard deviations.

## Ablations (one change each, compared with A0)

| run | change |
|---|---|
| A1 | 16 or 64 slices instead of 32 |
| A2 | no Fourier features (raw coordinates are still an input) |
| A3 | no design variables (mesh only) |
| A4 | an added local mesh-neighbour step in every block |

For question 2 the GNN mesh-only baselines in the README were trained with von Mises targets, while
A3 uses stress-component targets. To compare with only one difference, the GNN mesh-only variants
(with and without the global node) are re-trained with stress-component targets.

## Predictions

1. **Test set, with design variables.** Displacement error about the same as the GNN or lower
   (every slice sees the whole bracket). Von Mises field error *higher* than the GNN's 11.3 %, because
   a slice token is an average over many nodes and the stress peaks at the fillets are local; the GNN
   passes messages directly along mesh edges. Scalars (f1, peak stress, life) about the same as the
   GNN; the GP stays best on f1.
2. **Mesh only.** Displacement error far below the 98 % of the GNN without a global node, and close
   to the GNN with a global node, because the slice tokens play the role of many global nodes.
3. **Shifted family.** No clear improvement over the GNN; possibly worse. The Fourier features
   encode absolute positions, and in the shifted family the nodes sit at positions that were never
   seen in training. If this is the reason, A2 (no Fourier features) should do better on this set.
