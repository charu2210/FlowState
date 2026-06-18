## Interpretability Analysis — Feature Contributions

> Source: `permutation importance (model-backed)`

### Top Predictive Features

| Rank | Feature | Contribution | Bar |
| --- | --- | --- | --- |
| 1 | `coeff_variation` | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 2 | `entropy` | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 3 | `excess_kurtosis` | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 4 | `hjorth_mobility` | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 5 | `mean_iki` | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 6 | `skewness` | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 7 | `std_iki` | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |

### Interpretation

Feature contribution analysis (based on aggregated permutation importance across all trained models) reveals that **coeff_variation** (0.0%) is the strongest predictor of fatigue, followed by **entropy** (0.0%) and **excess_kurtosis** (0.0%). Together, the top three features account for 0.0% of total predictive signal. This pattern is consistent with the hypothesis that fatigue primarily manifests as disrupted rhythm (coeff_variation) and increased typing variability (entropy), rather than raw speed changes alone.

### Top Fatigue Indicators

1. **coeff_variation** — 0.0% contribution
2. **entropy** — 0.0% contribution
3. **excess_kurtosis** — 0.0% contribution
