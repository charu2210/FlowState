"""
FlowState: Keystroke Dynamics Cognitive Load Estimator
=======================================================
multi_user.py — Multi-User Study Framework

THE GAP
-------
A single-user system demonstrates a proof of concept.
A multi-user system demonstrates that the approach *generalises*.
The difference is the gap between a demo and a research contribution.

Two experimental designs address this, with different validity profiles:

DESIGN A: Population Model (Cross-User Generalisation)
------------------------------------------------------
Train on N-1 users, test on the held-out user (Leave-One-User-Out, LOUO).
This is the hardest and most meaningful evaluation: can the model detect
fatigue in a user it has never seen?

Weakness: Individual baseline differences in typing speed and rhythm
are large. A model trained only on fast typists will struggle with slow
typists, not because of fatigue, but because of inter-user variability.

DESIGN B: Personalised Baseline (Per-User Calibration)
------------------------------------------------------
Train on each user's own rested-state baseline (first 5-10 minutes of
their first session). Detect *deviations from their personal baseline*.

This is more realistic for deployment (a brief calibration session is
acceptable in most applications) and typically yields higher AUC.

The correct research framing is to report BOTH and discuss the tradeoff:
  - Population model: lower AUC but zero calibration cost
  - Personalised model: higher AUC but requires per-user calibration

This nuance is what distinguishes a research paper from a demo.

INTER-USER VARIABILITY FEATURES
--------------------------------
The 7-feature vector from features.py already captures CV (scale-normalised),
which partially mitigates speed differences. However, absolute mean IKI is
still user-specific. This module adds a Z-score normalisation step that
re-expresses each user's features relative to their own rested baseline —
a critical preprocessing step for cross-user studies.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ── User session data structure ───────────────────────────────────────────────

@dataclass
class UserSession:
    """
    All feature windows for one participant across one or more sessions.

    user_id     : str  — anonymised participant identifier (e.g., "P01")
    features_df : pd.DataFrame  — columns = FEATURE_NAMES, rows = windows
    labels      : np.ndarray    — {+1, -1} per window
    session_ids : np.ndarray    — session index per window (for multi-session)
    """
    user_id:     str
    features_df: pd.DataFrame
    labels:      np.ndarray
    session_ids: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if len(self.session_ids) == 0:
            self.session_ids = np.zeros(len(self.features_df), dtype=int)

    @property
    def X(self) -> np.ndarray:
        return self.features_df.values

    @property
    def y(self) -> np.ndarray:
        return self.labels

    @property
    def X_normal(self) -> np.ndarray:
        """Rested-state windows only — used as training data."""
        return self.X[self.y == 1]

    def calibration_baseline(self, n_windows: int = 50) -> np.ndarray:
        """
        Return the first n_windows of rested-state data as a
        per-user calibration baseline.  Used in personalised models.
        """
        normal = self.X_normal
        return normal[:min(n_windows, len(normal))]


# ── Normalisation ─────────────────────────────────────────────────────────────

def z_score_to_user_baseline(
    session: UserSession,
    n_calibration_windows: int = 50,
) -> UserSession:
    """
    Re-express each user's feature values relative to their own rested baseline.

    This is the key preprocessing step that enables cross-user generalisation:
    instead of absolute IKI values, the model sees *deviations from personal
    normal*, which are comparable across users with different typing speeds.

    Procedure:
        1. Fit a StandardScaler on the user's calibration baseline
           (first n_calibration_windows of rested data).
        2. Transform all windows with this user-specific scaler.

    Returns a new UserSession with z-scored features.
    """
    baseline = session.calibration_baseline(n_calibration_windows)
    if len(baseline) < 2:
        log.warning("User %s: calibration baseline too small (%d windows). "
                    "Skipping normalisation.", session.user_id, len(baseline))
        return session

    scaler = StandardScaler()
    scaler.fit(baseline)
    X_norm = scaler.transform(session.X)

    return UserSession(
        user_id=session.user_id,
        features_df=pd.DataFrame(X_norm, columns=session.features_df.columns),
        labels=session.labels.copy(),
        session_ids=session.session_ids.copy(),
    )


# ── Evaluation protocols ──────────────────────────────────────────────────────

def leave_one_user_out(
    sessions: list[UserSession],
    model_prototype,
    feature_cols: list[str],
    normalise: bool = True,
) -> pd.DataFrame:
    """
    Leave-One-User-Out (LOUO) cross-validation.

    For each user i:
        - Train on normal-class windows of all users j != i
        - Evaluate on all windows of user i (normal + anomalous)

    This is the gold-standard cross-user evaluation: it answers
    "does this model generalise to unseen users?"

    Parameters
    ----------
    sessions        : list of UserSession (one per participant)
    model_prototype : unfitted sklearn-compatible Pipeline or estimator
    feature_cols    : list of feature column names
    normalise       : if True, apply per-user z-score normalisation first

    Returns
    -------
    pd.DataFrame  with columns: User | AUC_ROC | F1 | N_Windows | Pct_Anomalous
    """
    if normalise:
        sessions = [z_score_to_user_baseline(s) for s in sessions]

    results = []

    for i, test_session in enumerate(sessions):
        train_sessions = [s for j, s in enumerate(sessions) if j != i]

        # Collect normal-class windows from all training users
        X_train = np.vstack([s.X_normal for s in train_sessions])

        # Fit a fresh clone of the model
        model = clone(model_prototype)
        model.fit(X_train)

        # Evaluate on test user
        X_test = test_session.X
        y_test = (test_session.y == -1).astype(int)   # 1 = anomalous

        y_pred = model.predict(X_test)
        y_pred_bin = (y_pred == -1).astype(int)

        # Anomaly scores for AUC
        if hasattr(model, "decision_function"):
            scores = -model.decision_function(X_test)
        elif hasattr(model, "score_samples"):
            scores = -model.score_samples(X_test)
        else:
            scores = y_pred_bin.astype(float)

        auc = roc_auc_score(y_test, scores) if y_test.sum() > 0 else float("nan")
        f1  = f1_score(y_test, y_pred_bin, zero_division=0)

        results.append({
            "User":          test_session.user_id,
            "AUC_ROC":       round(auc, 4),
            "F1":            round(f1, 4),
            "N_Windows":     len(X_test),
            "Pct_Anomalous": round(100 * y_test.mean(), 1),
        })

        log.info("LOUO | User %-5s | AUC=%.4f | F1=%.4f | N=%d",
                 test_session.user_id, auc, f1, len(X_test))

    df = pd.DataFrame(results)

    # Summary row
    summary = {
        "User": "MEAN ± STD",
        "AUC_ROC": f"{df['AUC_ROC'].mean():.4f} ± {df['AUC_ROC'].std():.4f}",
        "F1":      f"{df['F1'].mean():.4f} ± {df['F1'].std():.4f}",
        "N_Windows": int(df["N_Windows"].sum()),
        "Pct_Anomalous": round(df["Pct_Anomalous"].mean(), 1),
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    log.info("\n── LOUO Summary ──────────────────────────────────\n%s",
             df.to_string(index=False))
    return df


def personalised_evaluation(
    sessions: list[UserSession],
    model_prototype,
    n_calibration_windows: int = 50,
) -> pd.DataFrame:
    """
    Per-user personalised evaluation.

    For each user:
        - Fit on their own calibration baseline (first n_calibration_windows
          of rested-state data)
        - Evaluate on the remainder of their session

    Compared with LOUO, this typically yields higher AUC but requires
    a calibration phase. Report both in your paper and discuss the tradeoff.
    """
    results = []

    for session in sessions:
        # Normalise to user's own baseline before fitting
        session_norm = z_score_to_user_baseline(session, n_calibration_windows)

        baseline = session_norm.calibration_baseline(n_calibration_windows)
        rest_X   = session_norm.X[n_calibration_windows:]
        rest_y   = (session_norm.y[n_calibration_windows:] == -1).astype(int)

        if len(rest_X) == 0 or rest_y.sum() == 0:
            log.warning("User %s: insufficient post-calibration data. Skipping.",
                        session.user_id)
            continue

        model = clone(model_prototype)
        model.fit(baseline)

        if hasattr(model, "decision_function"):
            scores = -model.decision_function(rest_X)
        elif hasattr(model, "score_samples"):
            scores = -model.score_samples(rest_X)
        else:
            scores = (model.predict(rest_X) == -1).astype(float)

        auc = roc_auc_score(rest_y, scores)
        f1  = f1_score(rest_y, (model.predict(rest_X) == -1).astype(int),
                       zero_division=0)

        results.append({
            "User":          session.user_id,
            "AUC_ROC":       round(auc, 4),
            "F1":            round(f1, 4),
            "Calibration_N": len(baseline),
            "Test_N":        len(rest_X),
        })
        log.info("Personalised | User %-5s | AUC=%.4f | F1=%.4f",
                 session.user_id, auc, f1)

    df = pd.DataFrame(results)
    if len(df) > 0:
        summary = {
            "User": "MEAN ± STD",
            "AUC_ROC": f"{df['AUC_ROC'].mean():.4f} ± {df['AUC_ROC'].std():.4f}",
            "F1":      f"{df['F1'].mean():.4f} ± {df['F1'].std():.4f}",
            "Calibration_N": "-", "Test_N": int(df["Test_N"].sum()),
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    log.info("\n── Personalised Evaluation Summary ───────────────\n%s",
             df.to_string(index=False))
    return df


# ── Personal Baseline Deviation Report ───────────────────────────────────────

def baseline_deviation_report(
    session: UserSession,
    n_calibration_windows: int = 50,
) -> pd.DataFrame:
    """
    Compute per-window deviation from each user's personal rested baseline.

    Instead of absolute feature values, reports *how far* each window deviates
    from the user's own normal — making fatigue signals interpretable even for
    users with atypical baseline typing speeds.

    For each feature the deviation is expressed as:
        deviation_pct = (current - baseline_mean) / baseline_mean * 100

    A window 69% above the user's normal entropy is far more meaningful than
    reporting entropy=0.71 without context.

    Parameters
    ----------
    session : UserSession
    n_calibration_windows : int
        Number of rested-state windows used to establish the baseline.

    Returns
    -------
    pd.DataFrame
        Columns: window_idx, label, baseline_flags, <feature>_dev_pct for each feature.
    """
    baseline = session.calibration_baseline(n_calibration_windows)
    if len(baseline) < 2:
        log.warning("User %s: calibration baseline too small for deviation report.",
                    session.user_id)
        return pd.DataFrame()

    baseline_mean = baseline.mean(axis=0)
    baseline_std  = baseline.std(axis=0)

    feat_cols = list(session.features_df.columns)
    rows = []

    for i, (x_row, label) in enumerate(zip(session.X, session.y)):
        deviation_pcts = {}
        flags = []
        for j, feat in enumerate(feat_cols):
            bm = baseline_mean[j]
            dev = round((x_row[j] - bm) / abs(bm) * 100, 1) if not np.isclose(bm, 0.0) else float("nan")
            deviation_pcts[f"{feat}_dev_pct"] = dev

            # Flag features deviating > 2 std from baseline
            if not (isinstance(dev, float) and np.isnan(dev)) and baseline_std[j] > 0:
                z = (x_row[j] - bm) / baseline_std[j]
                if z > 2.0:
                    flags.append(f"{feat}↑")
                elif z < -2.0:
                    flags.append(f"{feat}↓")

        rows.append({
            "window_idx":    i,
            "label":         "fatigued" if label == -1 else "normal",
            "baseline_flags": "; ".join(flags) if flags else "within_normal",
            **deviation_pcts,
        })

    df = pd.DataFrame(rows)
    log.info("Baseline deviation report — User %s: %d windows, %d features tracked",
             session.user_id, len(df), len(feat_cols))
    return df
