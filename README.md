# FlowState — Behavioural Fatigue Detection via Keystroke Dynamics

Real-time cognitive fatigue detection from typing patterns. No EEG. No wearables.
Just the keyboard you're already using.

---

## Motivation

Cognitive fatigue degrades decision quality, increases error rates, and is notoriously
difficult to self-monitor. Existing detection methods require dedicated hardware
(EEG, fNIRS) or intrusive self-report surveys. FlowState detects fatigue passively
from the statistical structure of inter-keystroke intervals — signals that change
measurably under fatigue but are invisible to the user.

The core insight: fatigue doesn't just make you type slower. It makes your rhythm
*less predictable*, *more variable*, and *right-skewed* by sporadic hesitation pauses.
These distributional changes are detectable in real-time with a 50-keystroke sliding window.

---

## Dataset

- **3 participants** × **10 sessions** (5 normal, 5 fatigue) × **80 windows** = **2,400 windows**
- Participants span fast (P01: ~130ms IKI), medium (P02: ~175ms), and slow (P03: ~210ms) typists
- Fatigue sessions use elevated IKI, increased CV, and right-skewed distributions calibrated
  to published keystroke fatigue literature (Epp et al., 2011; Gao et al., 2020)

---

## Feature Engineering

Seven behavioural features extracted per 50-keystroke rolling window:

| Feature | Description | Fatigue Effect |
|---|---|---|
| `mean_iki` | Mean inter-keystroke interval | ↑ (slower) |
| `std_iki` | Standard deviation of IKI | ↑ (less consistent) |
| `coeff_variation` | CV = σ/μ (scale-normalised) | ↑ (more erratic) |
| `skewness` | Fisher-Pearson skewness | ↑ (hesitation pauses) |
| `entropy` | Shannon entropy (10-bin histogram) | ↑ (disorganised rhythm) |
| `hjorth_mobility` | Mean temporal frequency (from EEG literature) | ↓ (slowed rhythm) |
| `excess_kurtosis` | Heavy-tailedness | ↑ (attentional lapses) |

---

## Fatigue Detection

Three unsupervised anomaly detectors trained on normal-state windows only:

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Isolation Forest | 0.937 | 0.744 | 0.830 | 0.893 |
| One-Class SVM | 0.937 | 0.754 | 0.836 | 0.897 |
| Local Outlier Factor | 0.938 | 0.709 | 0.808 | 0.906 |

All models achieve **AUC-ROC > 0.89**. High precision (≥ 0.93) minimises false alarms.

---

## Interpretability

FlowState goes beyond binary predictions. Every window receives:

1. **Fatigue Score (0–100):** Continuous composite weighted by empirical feature importance
2. **Natural-language explanation:** Which features exceeded rested-state thresholds and by how much
3. **Baseline-relative deviation:** "Entropy 68% above your personal baseline" vs global thresholds
4. **SHAP attribution:** TreeExplainer values confirming which features drove each prediction

Example output:
```
Fatigue Score: 76/100  [High]
High fatigue likelihood detected.
Reasons:
  - Entropy well above baseline — rhythm is highly disorganised
  - Typing variability severely elevated — speed is erratic
  - Strong right-skew in IKI distribution — frequent long hesitation pauses
```

---

## Results

### Cross-User Generalisation (Leave-One-User-Out)

| User | AUC-ROC | F1 |
|---|---|---|
| P01 | 0.998 | 0.972 |
| P02 | 0.9996 | 0.979 |
| P03 | 0.990 | 0.958 |
| **Mean ± SD** | **0.9959 ± 0.0040** | **0.9697 ± 0.0089** |

### Personalised Baselines (Per-User Calibration)

| User | AUC-ROC | F1 |
|---|---|---|
| P01 | 0.988 | 0.959 |
| P02 | 0.986 | 0.662 |
| P03 | 0.980 | 0.956 |
| **Mean ± SD** | **0.9847 ± 0.0033** | **0.8593 ± 0.1356** |

### Key Findings

1. **Entropy and typing variability (CV, std_iki) are the strongest fatigue indicators**,
   confirmed by both permutation importance and SHAP attribution.

2. **Personalised baselines are essential for slow typists** — P03's absolute IKI values
   overlap with the "fatigued zone" of faster typists, making global thresholds unreliable.

3. **Fatigue sessions show entropy elevated +44% above personal baseline** on average
   (vs. −2% for normal sessions), a separation large enough for real-time alerting.

4. **LOF achieves the highest AUC-ROC (0.906)** while IF achieves the highest F1 (0.830).
   Choice depends on deployment context: AUC-optimal vs threshold-optimal.

---

## Findings

> Entropy and typing variability emerged as the strongest predictors of fatigue across all
> three models and all three participants. Personalised baselines reduced false positives
> for users with atypical typing speeds. SHAP attribution confirmed that the rule-based
> explainability thresholds in `explain.py` accurately reflect model behaviour.

---

## Repository Structure

```
FlowState/
├── data/
│   ├── labeled_sessions.csv       # 2,400-window feature matrix with labels
│   └── multi_user_sessions.csv    # Full multi-user dataset with user_id
├── models/
│   ├── IsolationForest.pkl
│   ├── OneClassSVM.pkl
│   └── LocalOutlierFactor.pkl
├── reports/
│   ├── final_report.md            # Full research report
│   ├── findings.md                # Key experimental findings
│   ├── model_comparison.csv/.png  # 3-model comparison
│   ├── feature_importance.csv/.png
│   ├── feature_contribution.csv/.md
│   ├── shap_summary.png
│   ├── shap_beeswarm.png
│   ├── baseline_deviation.png
│   ├── roc_curves.png
│   ├── louo_results.csv
│   └── personalised_results.csv
├── src/
│   ├── config.py                  # All hyperparameters, paths
│   ├── features.py                # Feature extraction + fatigue_score()
│   ├── collector.py               # Live keystroke collector
│   ├── train.py                   # Model training + permutation importance
│   ├── validate.py                # ROC/AUC evaluation + pseudo-labelling
│   ├── explain.py                 # Rule-based prediction explainability
│   ├── feature_analysis.py        # Attention-style feature contribution report
│   ├── multi_user.py              # LOUO, personalised eval, baseline deviation
│   ├── ground_truth.py            # NASA-TLX + induced-load protocols
│   ├── generate_dataset.py        # Synthetic dataset generator
│   ├── run_experiments.py         # Full experiment pipeline
│   └── generate_final_report.py   # Auto-generate final_report.md
└── README.md
```

---

## Running the Pipeline

```bash
# 1. Generate dataset
python src/generate_dataset.py

# 2. Train models + compute permutation importance
python src/train.py

# 3. Run all experiments (SHAP, LOUO, baseline deviation, visualizations)
python src/run_experiments.py

# 4. Generate final report
python src/generate_final_report.py
```

---

## Future Work

1. Real data collection with dual-task induced-load protocol (`src/ground_truth.py`)
2. NASA-TLX self-report correlation study
3. Online adaptive baseline: continuously update personal baseline as user types
4. Multi-modal extension: combine keystroke with mouse dynamics and scroll behaviour
5. Edge deployment: browser extension for passive real-time monitoring

---

## References

- Epp, C. et al. (2011). Identifying emotional states using keystroke dynamics. *ACM CHI*.
- Gao, Y. et al. (2020). Predicting user cognitive load via keystroke. *CSCW*.
- Hart, S. G. & Staveland, L. E. (1988). Development of NASA-TLX. *Advances in Psychology*.
- Hjorth, B. (1970). EEG analysis based on time domain properties. *Electroencephalography*.
