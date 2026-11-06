## Test set (60 designs)
| model | seeds | u field | von Mises field | f1 | peak von Mises | log10 life | von Mises field, shifted family |
|---|---|---|---|---|---|---|---|
| GNN, stress targets (final), lr 2e-3 | 3 | 8.0 ± 0.5 % | 11.5 ± 0.3 % | 0.94 ± 0.11 % | 2.8 ± 0.0 % | 0.085 ± 0.003 | 19.9 ± 0.8 % |
| GNN in the README (lr 1e-3, seed 0) | 1 | 9.0 (1 seed) % | 11.3 (1 seed) % | 0.82 (1 seed) % | 3.0 (1 seed) % | 0.085 (1 seed) | 20.0 (1 seed) % |
| Attention A0 (32 slices) | 3 | 6.8 ± 0.2 % | 7.0 ± 0.0 % | 0.70 ± 0.03 % | 2.6 ± 0.1 % | 0.082 ± 0.002 | 26.9 ± 0.2 % |
| A0, same wall-clock as the GNN | 1 | 2.6 (1 seed) % | 3.2 (1 seed) % | 0.68 (1 seed) % | 2.5 (1 seed) % | 0.081 (1 seed) | 26.4 (1 seed) % |
| GP on the 6 design variables | - | - | - | 0.10 % | 2.5 % | 0.101 | - |
| point-wise MLP, no edges (von Mises target) | 1 | 5.9 (1 seed) % | 33.8 (1 seed) % | - | - | - | 43.4 (1 seed) % |

## Ablations (test set)
| model | seeds | u field | von Mises field | f1 | peak von Mises | log10 life | von Mises field, shifted family |
|---|---|---|---|---|---|---|---|
| Attention A0 (32 slices) | 3 | 6.8 ± 0.2 % | 7.0 ± 0.0 % | 0.70 ± 0.03 % | 2.6 ± 0.1 % | 0.082 ± 0.002 | 26.9 ± 0.2 % |
| A1: 16 slices | 3 | 6.7 ± 0.1 % | 7.0 ± 0.1 % | 0.72 ± 0.06 % | 2.6 ± 0.2 % | 0.081 ± 0.003 | 27.4 ± 0.2 % |
| A1: 64 slices | 3 | 6.6 ± 0.3 % | 7.1 ± 0.0 % | 0.76 ± 0.10 % | 2.5 ± 0.1 % | 0.081 ± 0.003 | 27.3 ± 0.4 % |
| A2: no Fourier features | 3 | 7.0 ± 0.1 % | 10.2 ± 0.6 % | 0.83 ± 0.08 % | 2.9 ± 0.1 % | 0.083 ± 0.005 | 16.6 ± 0.5 % |
| A3: mesh only (no design variables) | 3 | 7.1 ± 0.3 % | 7.8 ± 0.1 % | 0.78 ± 0.04 % | 2.4 ± 0.1 % | 0.078 ± 0.003 | 27.5 ± 0.2 % |
| A4: + local mesh-neighbour step | 3 | 6.9 ± 0.2 % | 7.3 ± 0.2 % | 0.67 ± 0.06 % | 2.4 ± 0.3 % | 0.083 ± 0.003 | 26.8 ± 0.2 % |
| GNN, stress targets (final), lr 2e-3 | 3 | 8.0 ± 0.5 % | 11.5 ± 0.3 % | 0.94 ± 0.11 % | 2.8 ± 0.0 % | 0.085 ± 0.003 | 19.9 ± 0.8 % |
| GNN, mesh only | 3 | 20.2 ± 1.5 % | 24.7 ± 1.2 % | 3.59 ± 0.31 % | 6.0 ± 0.6 % | 0.224 ± 0.042 | 33.2 ± 2.4 % |
| GNN, mesh only, no global node | 3 | 106.0 ± 4.9 % | 25.5 ± 0.8 % | 3.91 ± 0.01 % | 7.0 ± 0.7 % | 0.261 ± 0.022 | 31.8 ± 1.1 % |

## Extrapolation
| | GNN (final) | Attention A0 | Attention A2 (no Fourier) | GP |
|---|---|---|---|---|
| OOD designs (40): f1 / peak stress / log10 life | 3.45 ± 0.33 % / 5.7 ± 0.5 % / 0.141 ± 0.007 | 3.57 ± 0.41 % / 6.9 ± 0.5 % / 0.142 ± 0.008 | 3.13 ± 0.36 % / 5.4 ± 0.5 % / 0.128 ± 0.005 | 0.56 % / 5.1 % / 0.170 |
| Shifted family (30): f1 / peak stress / log10 life | 5.57 ± 0.13 % / 7.9 ± 0.2 % / 0.100 ± 0.002 | 5.53 ± 0.15 % / 7.1 ± 0.1 % / 0.100 ± 0.010 | 5.00 ± 0.22 % / 6.9 ± 0.1 % / 0.097 ± 0.002 | 5.75 % / 7.5 % / 0.114 |
| OOD designs: displacement / von Mises field | 14.9 ± 1.0 % / 16.4 ± 0.5 % | 13.8 ± 0.2 % / 14.1 ± 0.2 % | 14.2 ± 0.2 % / 14.9 ± 0.7 % | - / - |
| Shifted family: displacement / von Mises field | 17.2 ± 0.3 % / 19.9 ± 0.8 % | 16.4 ± 0.2 % / 26.9 ± 0.2 % | 16.6 ± 0.3 % / 16.6 ± 0.5 % | - / - |

## Validation set (used for model selection)
| model | seeds | u field | von Mises field | f1 | peak von Mises | log10 life |
|---|---|---|---|---|---|---|
| GNN, stress targets (final), lr 2e-3 | 3 | 9.4 ± 0.5 % | 12.3 ± 0.4 % | 0.91 ± 0.11 % | 3.4 ± 0.3 % | 0.089 ± 0.004 |
| Attention A0 (32 slices) | 3 | 7.9 ± 0.3 % | 7.5 ± 0.0 % | 0.66 ± 0.11 % | 3.1 ± 0.2 % | 0.096 ± 0.008 |
| A1: 16 slices | 3 | 7.9 ± 0.2 % | 7.5 ± 0.1 % | 0.71 ± 0.05 % | 3.2 ± 0.3 % | 0.097 ± 0.007 |
| A1: 64 slices | 3 | 7.7 ± 0.3 % | 7.6 ± 0.1 % | 0.72 ± 0.03 % | 3.1 ± 0.1 % | 0.095 ± 0.006 |
| A2: no Fourier features | 3 | 8.2 ± 0.1 % | 10.9 ± 0.6 % | 0.82 ± 0.04 % | 3.3 ± 0.1 % | 0.095 ± 0.003 |
| A3: mesh only (no design variables) | 3 | 8.4 ± 0.3 % | 8.5 ± 0.1 % | 0.72 ± 0.02 % | 2.9 ± 0.2 % | 0.105 ± 0.009 |
| A4: + local mesh-neighbour step | 3 | 8.0 ± 0.3 % | 7.8 ± 0.2 % | 0.66 ± 0.01 % | 3.1 ± 0.2 % | 0.094 ± 0.010 |
