# League attribution report

Headline rating: **TrueSkill mu**.

## Ranked ladder

| rank | contender | rating | ci_lo | ci_hi | games |
|---:|---|---:|---:|---:|---:|
| 1 | strong_heuristic | 29.76 | 28.63 | 30.89 | 48 |
| 2 | strong_heuristic3 | 29.29 | 28.27 | 30.31 | 64 |
| 3 | weak_heuristic | 29.29 | 28.22 | 30.36 | 64 |
| 4 | 314dbfcf | 20.74 | 19.66 | 21.81 | 48 |
| 5 | ac264d9f | 17.90 | 16.87 | 18.93 | 64 |
| 6 | 10846a9b | 17.84 | 16.74 | 18.95 | 48 |
| 7 | 2854373c | 17.83 | 16.66 | 18.99 | 48 |

## Per-factor effects

Grouped marginal means (assumption-light headline) and OLS coefficients (marginal effect holding other factors fixed). Both carry uncertainty.

### phases.0.steps

| level | grouped mean | mean ci_lo | mean ci_hi | OLS coef | OLS se |
|---|---:|---:|---:|---:|---:|
| 15000 | 17.83 | 17.82 | 17.85 | (baseline) |  |
| 25000 | 19.32 | 16.54 | 22.10 | 1.484 | 1.426 |

### phases.steps_ratio

| level | grouped mean | mean ci_lo | mean ci_hi | OLS coef | OLS se |
|---|---:|---:|---:|---:|---:|
| None | 18.58 | 17.16 | 19.99 | (baseline) |  |

### schedule.pool_kinds

| level | grouped mean | mean ci_lo | mean ci_hi | OLS coef | OLS se |
|---|---:|---:|---:|---:|---:|
| None | 18.58 | 17.16 | 19.99 | (baseline) |  |

### ppo.shaping_weight

| level | grouped mean | mean ci_lo | mean ci_hi | OLS coef | OLS se |
|---|---:|---:|---:|---:|---:|
| 0.0 | 17.87 | 17.81 | 17.93 | (baseline) |  |
| 0.05 | 19.28 | 16.43 | 22.13 | 1.409 | 1.426 |

### ppo.net_profile

| level | grouped mean | mean ci_lo | mean ci_hi | OLS coef | OLS se |
|---|---:|---:|---:|---:|---:|
| None | 18.58 | 17.16 | 19.99 | (baseline) |  |

### seed

| level | grouped mean | mean ci_lo | mean ci_hi | OLS coef | OLS se |
|---|---:|---:|---:|---:|---:|
| 0 | 18.58 | 17.16 | 19.99 | (baseline) |  |

## How to read this (honest caveats)

- **Observational, not causal.** We swept a grid and *observed* ratings; we did not randomize at the unit level beyond seeds. Factor effects are associations under THIS recipe and track set, not causal guarantees.
- **Ratings have CIs.** Every effect is reported with uncertainty. Differences inside overlapping confidence intervals are 'not distinguishable', not 'zero'.
- **Factors interact.** A marginal effect averages over the other factors' levels; the grouped diff and the OLS can disagree when interactions are strong. When they do, inspect the relevant two-way cell means rather than a single marginal number.
- **Sampled-grid imbalance.** If random/LHS sampling was used the design is unbalanced and the grouped diffs inherit that imbalance -- the OLS becomes the primary read and this report says so.
