# Graph neural network surrogate for structural FEA

A MeshGraphNet-style graph neural network (GNN) that predicts the static displacement and stress
fields, the first natural frequency, the peak stress and the random-vibration fatigue life of a
parametric steel bracket directly from its mesh, trained on CalculiX finite element results.
The trained model is then used to screen 1,500 new designs in minutes, and the designs it proposes are
checked with the full FE analysis.

The FE model, the verification and the design of experiments are in the companion repository
[`bracket-fea`](https://github.com/RonHooo/bracket-fea).

A slice-attention transformer trained on the same data is compared with the GNN and the GP in
[Attention-based surrogate](#attention-based-surrogate): lower field errors than the GNN
(von Mises 7.0 % vs 11.5 %, 3 seeds each), a GP that is still best on f1, and worse extrapolation to
the shifted family unless the Fourier position features are removed.

## Data

* 400 designs from a Latin hypercube over 6 geometric variables (plate thicknesses, fillet radius,
  optional gussets, optional lightening hole).
* Two extrapolation sets: 40 "out-of-distribution" (OOD) designs with one or two variables outside
  the training ranges, and 30 designs from a **shifted family** in which the width, height,
  mounting-hole height and bolt position change. Those four dimensions are fixed in the training data
  and are not among the 6 design variables.
* Each design: CalculiX static analysis (C3D10, 25k-183k nodes), 10 modes, random-vibration fatigue.
* Split: train 280, validation 60, test 60 (random).
* Graph: the same geometry re-meshed with coarse linear tetrahedra (3-5k nodes), FE displacement and
  stress components interpolated onto its nodes. Edges are the tetrahedron edges.

## Model

* Inputs per node: position, three 0/1 flags (fixed bolt-hole surface, mounting-hole surface, free
  surface) and the 6 design variables. Edge features: relative position and length.
* Encode-process-decode with residual message passing (Pfaff et al., 2021), 10 layers, 64 features.
* A global latent ("virtual node") that reads the mean of all node latents and is fed back to every
  node and edge in every layer, as a path for long-range information.
* Node outputs: displacement (3) and the 6 stress components; von Mises is computed from the predicted
  components. Graph-level outputs: log f1, log10 fatigue life, log peak von Mises of the full FE mesh.
* Loss: MSE of standardised targets, node and graph terms weighted equally. AdamW, cosine schedule,
  50 epochs (about 1 h on an M3 laptop GPU).

## Results

### Accuracy on 60 unseen test designs

| model | u field | von Mises field | f1 | peak von Mises | log10 life | von Mises field, shifted family |
|---|---|---|---|---|---|---|
| GNN, stress-component targets (final) | 9.0 % | 11.3 % | 0.82 % | 3.0 % | 0.085 | 20.0 % |
| GNN, von Mises target | 5.0 % | 21.7 % | 0.86 % | 2.8 % | 0.089 | 39.4 % |
| GNN, von Mises target, no global node | 5.0 % | 20.8 % | 1.15 % | 3.0 % | 0.102 | 36.3 % |
| GNN, von Mises target, mesh only (no design variables) | 15.2 % | 29.0 % | 4.01 % | 8.7 % | 0.245 | 43.0 % |
| GNN, von Mises target, mesh only, no global node | 98.4 % | 33.3 % | 5.43 % | 9.8 % | 0.313 | 44.1 % |
| point-wise MLP, no edges (von Mises target) | 5.9 % | 33.8 % | - | - | - | 43.4 % |
| GP on the 6 design variables | - | - | 0.10 % | 2.5 % | 0.101 | - |

(Relative L2 errors are averaged over designs; f1, peak stress and life are mean absolute errors.
A log10 error of 0.085 means the predicted life is off by a factor of 10^0.085 = 1.22 on average.)

* **Predicting the stress components instead of von Mises halved the von Mises field error**
  (21.7 % -> 11.3 %) and cut the error of the field maximum from 15 % to 6 %. Von Mises has a kink
  where the stress changes sign (for example at the neutral plane of a plate in bending), while the
  components are smooth. The displacement error went up from 5.0 % to 9.0 %, because the same network
  now has 9 node outputs instead of 4.
* **Mesh connectivity matters for stress, much less for displacement.** The point-wise MLP sees the
  same node positions and design variables but no neighbours: its displacement error is close to the
  GNN's (5.9 % vs 5.0 %), its von Mises error is much larger (33.8 % vs 21.7 %).
* **The design variables carry most of the information.** With the mesh only, the displacement error
  triples (15.2 % vs 5.0 %), the f1 and life errors are 4.7x and 2.8x larger, and the von Mises error
  rises from 21.7 % to 29.0 %.
* **The global node matters only when the design variables are missing.** With the design variables
  given to every node, removing it changed little. With the mesh only, removing it made the
  displacement prediction fail (98 % error vs 15 %): ten message-passing steps reach only a small part
  of the bracket, so a node cannot tell how stiff the whole structure is, and the displacement
  magnitude depends on exactly that.
* **For the scalar outputs of this family a GP on the 6 variables is as good or better** (f1 0.10 %
  vs 0.82 %). With 340 training designs and a smooth response, a GP is hard to beat. The GNN adds the
  full fields.

![parity](results/figures/parity_gnn_vs_gp.png)

![field](results/figures/field_median.png)

### Extrapolation

| | GNN (final) | GP |
|---|---|---|
| OOD designs (40): f1 / peak stress / log10 life | 3.6 % / 5.9 % / 0.127 | 0.56 % / 5.1 % / 0.170 |
| Shifted family (30): f1 / peak stress / log10 life | 5.8 % / 7.7 % / 0.097 | 5.8 % / 7.5 % / 0.114 |
| Shifted family: displacement / von Mises field | 18.8 % / 20.0 % | - |

Both surrogates degrade by a similar amount outside the training family. I expected the mesh input
to give the GNN a clear advantage on the shifted family; it did not. In the parity plots the GNN also
saturates at the edge of the training range (high f1, high stress), where the GP extrapolates better.

### Design screening

1,500 new candidate designs were meshed (5 min on 4 cores) and evaluated by the GNN with von Mises
targets (1.5 min, including GPU kernel compilation). Each surrogate proposed its 8 lightest designs
predicted to meet f1 >= 100 Hz and a fatigue life of at least 1,500 h (the 1,000 h target with the
1.5 mesh margin from `bracket-fea`), and those were checked with the full CalculiX analysis.

* GNN and GP proposed the same 8 designs; 6 of 8 met the requirements in FE (the 2 misses had
  1,300-1,400 h instead of the 1,500 h screening limit).
* The best verified design weighs **0.856 kg**, 3 % lighter than the best of the 400 DOE designs
  (0.884 kg) and 8 % lighter than a uniformly thickened bracket (0.932 kg). It meets f1 >= 100 Hz by
  only 0.15 Hz on a fine mesh, so with any frequency margin the DOE design is the safer choice.
* Running all 1,500 candidates through FE would have taken about 32 CPU-hours; the screening used
  about 15 CPU-minutes plus 8 FE checks.

![screening](results/figures/screening.png)

## Attention-based surrogate

A second model, a slice-attention transformer, trained on exactly the same data, split, node inputs,
targets, loss and metrics as the GNN (same code in `gnnsurrogate/`). The three questions and my
predictions were committed before the first training run: [docs/attention_plan.md](docs/attention_plan.md).

1. How do its field and scalar errors compare with the GNN and the GP?
2. With the mesh only (no design variables), the GNN without a global node failed on displacement.
   Can attention, where every node reaches every other node in one layer, recover this?
3. Does it extrapolate better to the shifted family?

### Model

* Node inputs as for the GNN, plus Fourier features of the position (sin and cos of 2^k·π·x,
  k = 0..5, x in units of 100 mm; 36 extra features). No edges.
* 6 pre-norm blocks with 96 features; each block is slice attention plus a feed-forward network.
* **Slice attention** (after Transolver, Wu et al. 2024): a softmax assigns every node softly to 32
  slices, each slice token is the weighted mean of its nodes, the 32 tokens exchange information with
  4-head self-attention, and the result goes back to the nodes with the same weights. The cost is
  O(N·M) instead of O(N²) (N ≈ 3,800 nodes, M = 32 slices), and every node gets information from the
  whole bracket in one layer. The 4 heads share one set of slice weights (Transolver uses one per head).
* Graph-level outputs: attention pooling (4 heads) followed by an MLP.
* Softmax, scaled dot-product attention and multi-head attention are written by hand in
  [`gnnsurrogate/attention.py`](gnnsurrogate/attention.py), without `nn.MultiheadAttention`. The
  tests in [`tests/test_attention.py`](tests/test_attention.py) compare them with
  `torch.nn.functional.scaled_dot_product_attention` (max. difference below 1e-5 in float32) and with
  `nn.MultiheadAttention` using copied weights. They also check that shuffling the node order shuffles
  the output in the same way, and that designs in one batch do not mix.

![slice attention](results/attention/slice_attention_diagram.png)

### Fair comparison

| | GNN | attention (A0) |
|---|---|---|
| parameters | 530,956 | 641,302 |
| epochs, batch, optimiser | 50, 4 designs, AdamW + cosine | same |
| learning rate (5e-4 / 1e-3 / 2e-3, lowest validation loss) | 2e-3 | 2e-3 |
| seeds | 0, 1, 2 | 0, 1, 2 |
| training time, 50 epochs, M3 laptop GPU | 40-46 min | 7-12 min |

* The GNN in the first part of this README used lr 1e-3 without tuning and one seed. Tuning only the
  attention model would have been a second difference, so the GNN got the same grid and the same
  rule. For the GNN, 2e-3 and 1e-3 were a tie (validation loss 0.0267 vs 0.0268), and its test errors
  barely changed (first two rows below).
* Same number of epochs, not the same time. A side run with 300 epochs gives the attention model
  about the GNN's training time (one seed).
* The test, OOD and shifted sets were evaluated once, after all runs had finished. All choices used
  the validation set.
* I call a difference real only if the difference of the seed means is larger than the larger of the
  two seed standard deviations. With 3 seeds this is a minimum bar, not a significance test.

### Results on the 60 test designs (mean ± std over 3 seeds)

| model | seeds | u field | von Mises field | f1 | peak von Mises | log10 life | von Mises field, shifted family |
|---|---|---|---|---|---|---|---|
| GNN, stress targets, lr 2e-3 | 3 | 8.0 ± 0.5 % | 11.5 ± 0.3 % | 0.94 ± 0.11 % | 2.8 ± 0.0 % | 0.085 ± 0.003 | 19.9 ± 0.8 % |
| GNN in the table above (lr 1e-3) | 1 | 9.0 % | 11.3 % | 0.82 % | 3.0 % | 0.085 | 20.0 % |
| **Attention A0** | 3 | **6.8 ± 0.2 %** | **7.0 ± 0.0 %** | 0.70 ± 0.03 % | 2.6 ± 0.1 % | 0.082 ± 0.002 | 26.9 ± 0.2 % |
| A0, 300 epochs (about the GNN's training time) | 1 | 2.6 % | 3.2 % | 0.68 % | 2.5 % | 0.081 | 26.4 % |
| GP on the 6 design variables | - | - | - | **0.10 %** | 2.5 % | 0.101 | - |
| point-wise MLP, no edges (von Mises target) | 1 | 5.9 % | 33.8 % | - | - | - | 43.4 % |

* **The attention model had lower field errors than the GNN**: displacement 6.8 % vs 8.0 %, von Mises
  7.0 % vs 11.5 %. f1 was also lower (0.70 % vs 0.94 %), peak stress slightly lower (2.6 % vs 2.8 %),
  and fatigue life a tie (0.082 vs 0.085, within the seed spread).
* **Part of the von Mises gain comes from the Fourier features, not from attention.** Without them
  (A2, below) the von Mises error is 10.2 %: still lower than the GNN's 11.5 %, but far from 7.0 %. The
  GNN has no Fourier features, so this comparison has two differences. A GNN with Fourier features
  is the missing control.
* **The GP is still the best model for f1** (0.10 % vs 0.70 %, and 0.56 % vs 3.6 % on the OOD
  designs). I think the reason is that f1 is a smooth function of 6 numbers, and a GP fitted to 340
  designs interpolates such a function very well, while the networks learn f1 together with 9 field
  outputs on about 4,000 nodes per design. Longer training did not lower the f1 error (next point).
  On peak stress the GP and the attention model are about the same (2.5 % vs 2.6 %), and on life both
  networks are better than the GP (0.082 and 0.085 vs 0.101).
* With 300 epochs (56 min, a little more than the GNN's 40-46 min; its validation loss had reached
  its final value by 46 min), the field errors fell to 2.6 % and 3.2 %, but f1, peak stress and life
  did not change. So the scalar outputs are limited by something other than training time. This is
  one seed.

![parity](results/attention/parity_gnn_attention_gp.png)

![field](results/attention/field_median_gnn_vs_attention.png)

The test design with the median attention error (seed 0): GNN 9.1 %, attention 6.9 %. Both models
have their largest errors next to the two bolt holes in the base plate; the GNN error is larger there
and spread more widely.

### Mesh only (question 2)

| test set | u field | von Mises field | f1 | log10 life |
|---|---|---|---|---|
| Attention A0 (with the design variables) | 6.8 ± 0.2 % | 7.0 ± 0.0 % | 0.70 ± 0.03 % | 0.082 ± 0.002 |
| Attention A3, mesh only | 7.1 ± 0.3 % | 7.8 ± 0.1 % | 0.78 ± 0.04 % | 0.078 ± 0.003 |
| GNN, mesh only | 20.2 ± 1.5 % | 24.7 ± 1.2 % | 3.59 ± 0.31 % | 0.224 ± 0.042 |
| GNN, mesh only, no global node | 106.0 ± 4.9 % | 25.5 ± 0.8 % | 3.91 ± 0.01 % | 0.261 ± 0.022 |

* The GNN mesh-only runs in the first table used von Mises targets. They were retrained here with
  stress-component targets and 3 seeds, so that the model is the only difference.
* **Without the design variables, the attention model lost little** (displacement 7.1 % vs 6.8 %,
  von Mises 7.8 % vs 7.0 %). On the test set it is still better than the GNN that is given the design
  variables. The GNN without a global node failed again (106 % displacement error).
* I think the reason is that every node sees the whole bracket in the first layer, so the model can
  read the plate thicknesses, fillet and gussets from the geometry. I did not test this directly.
* The design variables still help outside the training ranges: on the OOD designs the mesh-only
  model has 17.6 % displacement error vs 13.8 % with them, and 4.7 % vs 3.6 % on f1.

### Extrapolation (question 3)

| | GNN | Attention A0 | Attention A2 (no Fourier) | GP |
|---|---|---|---|---|
| OOD designs (40): f1 / peak stress / log10 life | 3.45 ± 0.33 % / 5.7 ± 0.5 % / 0.141 ± 0.007 | 3.57 ± 0.41 % / 6.9 ± 0.5 % / 0.142 ± 0.008 | 3.13 ± 0.36 % / 5.4 ± 0.5 % / 0.128 ± 0.005 | 0.56 % / 5.1 % / 0.170 |
| Shifted family (30): f1 / peak stress / log10 life | 5.57 ± 0.13 % / 7.9 ± 0.2 % / 0.100 ± 0.002 | 5.53 ± 0.15 % / 7.1 ± 0.1 % / 0.100 ± 0.010 | 5.00 ± 0.22 % / 6.9 ± 0.1 % / 0.097 ± 0.002 | 5.75 % / 7.5 % / 0.114 |
| OOD designs: displacement / von Mises field | 14.9 ± 1.0 % / 16.4 ± 0.5 % | 13.8 ± 0.2 % / 14.1 ± 0.2 % | 14.2 ± 0.2 % / 14.9 ± 0.7 % | - |
| Shifted family: displacement / von Mises field | 17.2 ± 0.3 % / 19.9 ± 0.8 % | 16.4 ± 0.2 % / **26.9 ± 0.2 %** | 16.6 ± 0.3 % / **16.6 ± 0.5 %** | - |

* **On the shifted family the attention model was worse than the GNN** on the von Mises field
  (26.9 % vs 19.9 %). Displacement was about the same for all models (16-17 %).
* **Without the Fourier features (A2) the shifted von Mises error drops to 16.6 %**, lower than the
  GNN's. The Fourier features encode the absolute position down to about 6 mm, and in the shifted
  family the holes and edges sit at positions never seen in training. A2 pays for this on the test
  set (10.2 % vs 7.0 %). So the position encoding sets a trade-off between accuracy inside the
  training family and extrapolation.
* On the OOD designs the attention model had lower field errors than the GNN, but a higher peak
  stress error (6.9 % vs 5.7 %).

![shift](results/attention/shift_error_distribution.png)

### Ablations (test set)

| model | u field | von Mises field | f1 | peak von Mises | log10 life | von Mises field, shifted family |
|---|---|---|---|---|---|---|
| Attention A0 (32 slices) | 6.8 ± 0.2 % | 7.0 ± 0.0 % | 0.70 ± 0.03 % | 2.6 ± 0.1 % | 0.082 ± 0.002 | 26.9 ± 0.2 % |
| A1: 16 slices | 6.7 ± 0.1 % | 7.0 ± 0.1 % | 0.72 ± 0.06 % | 2.6 ± 0.2 % | 0.081 ± 0.003 | 27.4 ± 0.2 % |
| A1: 64 slices | 6.6 ± 0.3 % | 7.1 ± 0.0 % | 0.76 ± 0.10 % | 2.5 ± 0.1 % | 0.081 ± 0.003 | 27.3 ± 0.4 % |
| A2: no Fourier features | 7.0 ± 0.1 % | 10.2 ± 0.6 % | 0.83 ± 0.08 % | 2.9 ± 0.1 % | 0.083 ± 0.005 | 16.6 ± 0.5 % |
| A3: mesh only | 7.1 ± 0.3 % | 7.8 ± 0.1 % | 0.78 ± 0.04 % | 2.4 ± 0.1 % | 0.078 ± 0.003 | 27.5 ± 0.2 % |
| A4: + local mesh-neighbour step | 6.9 ± 0.2 % | 7.3 ± 0.2 % | 0.67 ± 0.06 % | 2.4 ± 0.3 % | 0.083 ± 0.003 | 26.8 ± 0.2 % |

* The number of slices (16, 32, 64) made no difference that matters (at most 0.1 points of test von
  Mises error).
* Adding a mesh-neighbour averaging step to every block (A4) did not help (7.3 % vs 7.0 %), and it
  made inference 3x slower.

![ablations](results/attention/ablations.png)

### Inference time and training curves

| model | parameters | time per design |
|---|---|---|
| GNN | 530,956 | 52 ms |
| Attention A0 (32 slices) | 641,302 | 5.6 ms |
| A1: 16 / 64 slices | 631,990 / 659,926 | 4.8 / 7.2 ms |
| A4: + local step | 809,494 | 16.4 ms |
| point-wise MLP | 267,524 | 1.4 ms |

One design per forward pass, padded as in training, on the M3 laptop GPU (MPS); median of 5 passes
over the 60 test designs, all models timed one after another in the same process
(`scripts/time_inference.py`). Most of the GNN's work is on the edges (about 39,000 per design,
10 layers); the attention model works on the nodes (about 3,800) and a 32 x 32 attention matrix.

![curves](results/attention/training_curves.png)

### What the slices learned

![slices](results/attention/learned_slices.png)

The most likely slice of each node, seed 0, layers 1, 4 and 6. In the first layer the slices are
scattered patches and 27 of the 32 slices are used. In the last layer only 13 are used, and the large
ones follow the parts of the bracket: the gussets with the fillet, the base plate, the upright.

### Predictions vs results

| prediction (before training) | result |
|---|---|
| Test: displacement about the same as the GNN or lower | Lower: 6.8 % vs 8.0 %. |
| Test: von Mises higher than the GNN's 11.3 %, because slice tokens average out local peaks | **Wrong**: 7.0 %. My guess: each node keeps its own features through the residual path, and the Fourier features describe its position in fine detail (without them: 10.2 %). |
| Test: scalars about the same; the GP stays best on f1 | f1 lower than the GNN (0.70 % vs 0.94 %), peak stress and life about the same; the GP is best on f1. |
| Mesh only: displacement far below 98 %, close to the GNN with a global node | Far below (7.1 %), and also much better than the GNN with a global node (20.2 %). |
| Shifted family: no improvement, possibly worse; if the Fourier features are the reason, A2 does better | Worse (26.9 % vs 19.9 %); A2 does better (16.6 %). |

### Choices I made

| choice | reason |
|---|---|
| One set of slice weights shared by all heads | Simpler; I did not test one set per head. |
| Model size (96 features, 6 blocks, 32 slices) fixed before training; only the learning rate was tuned | Keeps the validation set from being used for many choices. |
| lr 2e-3 is at the edge of the grid; the grid was not extended | The rule was fixed in advance. A larger value might be better for both models. |
| Same epochs as the GNN, plus one 300-epoch run | The GNN takes about 5x longer per epoch. |
| A4 averages over the mesh neighbours, not over k nearest neighbours | The mesh edges are already there and are what the GNN uses. |
| Model weights in git for the main runs only (attention, GNN and 300-epoch runs) | About 2.5 MB each; the ablation and learning-rate runs keep their configs, histories and metrics. |
| The optional re-screening of the 1,500 designs with the attention model was not done | Out of time; the GNN and GP already agreed on the designs. |

### Numbers that change on a rerun

* Training on the Apple GPU is not bit-for-bit repeatable. The seed spread in the tables shows the
  size of the variation (for example 0.3 points on the GNN's von Mises error).
* Times depend on the machine and on what else runs. The same attention model took 7.4-12.1 min for 50
  epochs; the slow runs overlapped with CPU-heavy jobs on the same laptop.
* The 300-epoch run and the slice figure are one seed each.

### Limitations

* One bracket family and 280 training designs. This does not show that attention is better than
  message passing in general.
* The attention model has Fourier features and the GNN does not; a GNN with Fourier features would
  separate the two effects.
* Three seeds per model.

## Limitations and next steps

* One design family, 280 training designs. The GNN did not generalise to the shifted family better
  than the parameter-based GP; a multi-scale (hierarchical) GNN and more varied geometry would be the
  next step.
* The graph is a coarse re-mesh, so stress peaks in small fillets are under-resolved; the peak stress
  comes from the graph-level head.
* Each GNN variant in the first table was trained once (one random seed), so small differences
  between them, such as with and without the global node, may not be significant. The attention
  comparison above uses 3 seeds for both models.

## Notes on the implementation

* Pure PyTorch (no PyG); scatter operations with `index_add_` / `scatter_reduce`.
* Apple-silicon GPU (MPS): the backend recompiles kernels for every new tensor shape, which made
  training with variable-size graphs very slow. Batches are padded to node / edge counts that are
  multiples of 1024 / 8192; padding nodes form a dummy graph that is masked out of the loss.
* PyTorch 2.15 MPS crashes on macOS 14 (`arrayViewWithShape:strides:`); 2.14.1 works.

## Reproduce

Clone this repository and `bracket-fea` into the same folder; the commands below assume
`../bracket-fea`. The FE dataset (about 1.2 GB) is not in the repository. It is regenerated with `bracket-fea`
(`studies/doe.py --n 400 --n-shift 30`, about 11 CPU-hours), which writes `data/raw/*.npz`.

```bash
PYTHONPATH=.:../bracket-fea python scripts/build_graphs.py
python -m gnnsurrogate.train --targets stress --epochs 50 --out results/runs/gnn_stress
python -m gnnsurrogate.evaluate --run results/runs/gnn_stress
python -m gnnsurrogate.baselines
python scripts/screen_designs.py
python scripts/make_figures.py
python scripts/results_table.py
```

Attention model (one seed; the other runs differ only in the flags stored in each run's
`config.json`):

```bash
python -m gnnsurrogate.train --model transformer --targets stress --hidden 96 --layers 6 \
    --heads 4 --slices 32 --lr 2e-3 --epochs 50 --seed 0 --out results/runs/transformer_s0
python -m gnnsurrogate.evaluate --run results/runs/transformer_s0
python scripts/time_inference.py
python scripts/compare_attention.py              # tables and figures in results/attention/
python scripts/attention_explainer_figures.py transformer_s0
python -m pytest tests
```

## License

MIT. See [LICENSE](LICENSE).
