"""
FlowState: Keystroke Dynamics Cognitive Load Estimator
=======================================================
validate.py — Experimental validation via pseudo-labelling.

The Core Problem
----------------
FlowState operates in an unsupervised regime: there are no ground-truth
labels declaring "this window = fatigued."  This module implements a
principled pseudo-labelling strategy that converts the problem into a
weakly-supervised evaluation, enabling meaningful ROC / AUC analysis.

Pseudo-Labelling Strategy: Time-Block Annotation
-------------------------------------------------
Rationale: Cognitive fatigue in a typing session accumulates monotonically
over time (Ackerman & Kanfer, 2009).  A controlled session is partitioned:

    Phase 1 (t = 0–20 min)   → 'rested'   → label = +1 (inlier)
    Phase 2 (t = 20–40 min)  → 'moderate' → label = +1 (inlier, excluded
                                              from hard evaluation)
    Phase 3 (t = 40–60 min)  → 'fatigued' → label = −1 (anomaly)

This mirrors protocols from keystroke-dynamics fatigue literature
(Sano et al., 2015; Gao et al., 2020) and provides a defensible, if
imperfect, evaluation baseline.

Alternative strategies (described but not implemented here):
  - NASA-TLX self-report correlation
  - ECG/EEG concurrent recording as physiological ground truth
  - Dual-task paradigm (secondary reaction-time task)

Usage
-----
    python validate.py --session_csv data/raw_keystrokes.csv \
                       --model_dir   models/
"""

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    auc,
    roc_curve,
    average_precision_score,
    PrecisionRecallDisplay,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, REPORTS_DIR, LOG_LEVEL
from features import FEATURE_NAMES, extract_feature_vector

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── Pseudo-labelling ───────────────────────────────────────────────────────────

def apply_time_block_labels(
    df: pd.DataFrame,
    rested_frac:  float = 0.33,
    fatigued_frac: float = 0.33,
) -> pd.DataFrame:
    """
    Assign pseudo-labels based on session-relative position.

    The first `rested_frac` of windows are labelled +1 (normal).
    The last `fatigued_frac` of windows are labelled −1 (anomalous).
    The middle segment is dropped from hard evaluation to avoid
    ambiguous transitional windows contaminating metrics.

    Parameters
    ----------
    df            : DataFrame with one row per rolling window, time-ordered.
    rested_frac   : Fraction of session treated as rested baseline.
    fatigued_frac : Fraction of session treated as fatigued.

    Returns
    -------
    DataFrame with added 'label' column {1, −1}.
    """
    n = len(df)
    rested_end   = int(n * rested_frac)
    fatigued_start = int(n * (1 - fatigued_frac))

    labels = np.full(n, np.nan)
    labels[:rested_end]       = 1
    labels[fatigued_start:]   = -1

    df = df.copy()
    df["label"] = labels
    df_labeled = df.dropna(subset=["label"]).copy()
    df_labeled["label"] = df_labeled["label"].astype(int)

    log.info(
        "Pseudo-labels: %d rested (+1) | %d fatigued (−1) | %d transitional (dropped)",
        (labels == 1).sum(), (labels == -1).sum(),
        np.isnan(labels).sum(),
    )
    return df_labeled


# ── ROC / AUC plotting ─────────────────────────────────────────────────────────

def plot_roc_curves(
    X: np.ndarray,
    y: np.ndarray,
    model_dir: Path = MODELS_DIR,
    output_dir: Path = REPORTS_DIR,
) -> pd.DataFrame:
    """
    Load all persisted models, compute decision scores, and render overlaid
    ROC curves and a Precision-Recall curve figure.

    Convention:  label = −1 is the positive class (anomaly / fatigued).

    Returns
    -------
    pd.DataFrame  with per-model AUC-ROC and Average Precision.
    """
    y_bin = (y == -1).astype(int)   # 1 = fatigued (positive class)
    model_paths = list(model_dir.glob("*.pkl"))

    if not model_paths:
        raise FileNotFoundError(f"No saved models found in {model_dir}. Run train.py first.")

    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr,  ax_pr  = plt.subplots(figsize=(7, 6))

    summary = []

    for mp in model_paths:
        pipeline = joblib.load(mp)
        name = mp.stem

        # Obtain anomaly scores (higher = more anomalous)
        if hasattr(pipeline.named_steps["model"], "decision_function"):
            scores = -pipeline.decision_function(X)
        elif hasattr(pipeline.named_steps["model"], "score_samples"):
            scores = -pipeline.score_samples(X)
        else:
            log.warning("%s lacks scoring method; skipping ROC.", name)
            continue

        fpr, tpr, _ = roc_curve(y_bin, scores)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(y_bin, scores)

        ax_roc.plot(fpr, tpr, lw=2, label=f"{name}  (AUC = {roc_auc:.3f})")

        prec, rec, _ = __import__("sklearn.metrics", fromlist=["precision_recall_curve"]).precision_recall_curve(y_bin, scores)
        ax_pr.plot(rec, prec, lw=2, label=f"{name}  (AP = {ap:.3f})")

        summary.append({"Model": name, "AUC_ROC": round(roc_auc, 4), "Avg_Precision": round(ap, 4)})
        log.info("%s → AUC-ROC=%.4f | Avg Precision=%.4f", name, roc_auc, ap)

    # Style ROC figure
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.set_title("FlowState — ROC Curves (Fatigue Detection)", fontsize=14)
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)

    # Style PR figure
    ax_pr.set_xlabel("Recall", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)
    ax_pr.set_title("FlowState — Precision-Recall Curves", fontsize=14)
    ax_pr.legend(loc="upper right")
    ax_pr.grid(alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_roc.tight_layout()
    fig_pr.tight_layout()
    fig_roc.savefig(output_dir / "roc_curves.png", dpi=150)
    fig_pr.savefig(output_dir / "pr_curves.png",  dpi=150)
    log.info("Figures saved to %s", output_dir)

    summary_df = pd.DataFrame(summary).sort_values("AUC_ROC", ascending=False)
    summary_df.to_csv(output_dir / "auc_summary.csv", index=False)
    return summary_df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FlowState Validation Pipeline")
    parser.add_argument("--session_csv", type=Path, required=True,
                        help="Path to raw time-ordered keystroke feature CSV.")
    parser.add_argument("--model_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--rested_frac",   type=float, default=0.33)
    parser.add_argument("--fatigued_frac", type=float, default=0.33)
    args = parser.parse_args()

    df = pd.read_csv(args.session_csv)
    df_labeled = apply_time_block_labels(df, args.rested_frac, args.fatigued_frac)

    X = df_labeled[FEATURE_NAMES].values
    y = df_labeled["label"].values

    summary = plot_roc_curves(X, y, model_dir=args.model_dir)
    print("\n── Evaluation Summary ──────────────────────────────")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
