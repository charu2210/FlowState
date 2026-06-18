"""
FlowState: Keystroke Dynamics Cognitive Load Estimator
=======================================================
explain.py — Rule-Based Prediction Explainability Engine

Translates raw feature values into human-readable explanations of why a
particular window was flagged as fatigue-indicative.  The design is
deliberately rule-based rather than LLM-based: rules are deterministic,
auditable, and directly interpretable by a domain expert.

This module implements a lightweight form of mechanistic interpretability:
  - Which features exceeded their rested-state thresholds?
  - By how much?
  - What does that mean behaviourally?

The explanation output can be logged alongside predictions in collector.py,
included in automated reports via report.py, and cited as an interpretability
contribution in research submissions.
"""

from __future__ import annotations
import numpy as np


# ── Threshold table (derived from feature literature & rested-state norms) ────

_THRESHOLDS = {
    "entropy": {
        "high":    3.2,    # above → rhythm highly disorganised
        "medium":  2.2,    # above → mild rhythm disruption
    },
    "coeff_variation": {
        "high":    0.50,   # above → typing speed highly erratic
        "medium":  0.30,   # above → moderately elevated variability
    },
    "skewness": {
        "high":    1.5,    # above → frequent long hesitation pauses
        "medium":  0.7,    # above → mild pause asymmetry
    },
    "excess_kurtosis": {
        "high":    3.0,    # above → attentional lapses (extreme pauses)
        "medium":  1.0,    # above → heavier-tailed than normal
    },
    "hjorth_mobility": {
        "low":     0.8,    # below → rhythm slowed significantly (fatigue)
        "medium":  1.2,    # below → mildly reduced rhythm frequency
    },
    "mean_iki": {
        "high":    250.0,  # above → typing speed notably slow (ms)
        "medium":  180.0,
    },
}

_EXPLANATIONS = {
    "entropy": {
        "high":   "Entropy well above baseline — rhythm is highly disorganised",
        "medium": "Entropy elevated — mild rhythm disruption detected",
    },
    "coeff_variation": {
        "high":   "Typing variability severely elevated — speed is erratic",
        "medium": "Typing variability above baseline — moderate inconsistency",
    },
    "skewness": {
        "high":   "Strong right-skew in IKI distribution — frequent long hesitation pauses",
        "medium": "Mild skew detected — occasional hesitation pauses",
    },
    "excess_kurtosis": {
        "high":   "Heavy-tailed IKI distribution — attentional lapses present",
        "medium": "Kurtosis above Gaussian baseline — isolated extreme pauses",
    },
    "hjorth_mobility": {
        "low":    "Hjorth Mobility well below baseline — rhythm frequency has dropped",
        "medium": "Hjorth Mobility slightly reduced — mildly slowed rhythm",
    },
    "mean_iki": {
        "high":   "Mean inter-keystroke interval elevated — typing speed reduced",
        "medium": "Typing speed slightly below personal average",
    },
}


# ── Core explainer ─────────────────────────────────────────────────────────────

def explain_prediction(feature_vector: dict, fatigue_score: float | None = None) -> str:
    """
    Generate a human-readable explanation for a prediction window.

    Parameters
    ----------
    feature_vector : dict
        Output of features.extract_feature_vector().
    fatigue_score : float, optional
        Pre-computed fatigue score (0–100).  If provided, displayed in header.

    Returns
    -------
    str
        Multi-line explanation string.

    Examples
    --------
    >>> explain_prediction({"entropy": 3.8, "coeff_variation": 0.62, "skewness": 1.7})
    High fatigue likelihood detected.
    Reasons:
    - Entropy well above baseline — rhythm is highly disorganised
    - Typing variability severely elevated — speed is erratic
    - Strong right-skew in IKI distribution — frequent long hesitation pauses
    """
    reasons = _collect_reasons(feature_vector)

    if len(reasons) == 0:
        level = "Normal"
        header = "No significant fatigue signals detected — typing rhythm appears normal."
    elif len(reasons) <= 1:
        level = "Low"
        header = "Low fatigue signal detected."
    elif len(reasons) <= 3:
        level = "Moderate"
        header = "Moderate fatigue likelihood detected."
    else:
        level = "High"
        header = "High fatigue likelihood detected."

    lines = []
    if fatigue_score is not None:
        lines.append(f"Fatigue Score: {fatigue_score:.1f}/100  [{level}]")
    else:
        lines.append(f"Fatigue Level: {level}")
    lines.append(header)

    if reasons:
        lines.append("Reasons:")
        for r in reasons:
            lines.append(f"  - {r}")

    return "\n".join(lines)


def explain_short(feature_vector: dict) -> str:
    """
    Compact single-line explanation for CSV logging.

    Returns
    -------
    str
        Semicolon-separated list of triggered signals, or 'Normal'.
    """
    reasons = _collect_reasons(feature_vector)
    if not reasons:
        return "Normal"
    # Truncate each reason to a short tag
    tags = []
    for r in reasons:
        # Take first few words as a tag
        words = r.split("—")[0].strip().split()[:4]
        tags.append(" ".join(words))
    return "; ".join(tags)


def _collect_reasons(feature_vector: dict) -> list[str]:
    """Internal: collect all triggered explanation strings."""
    reasons = []

    def _check(feat: str, value: float | None) -> None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        t = _THRESHOLDS.get(feat, {})
        e = _EXPLANATIONS.get(feat, {})

        # Directional check: most features → fatigue when HIGH
        if feat == "hjorth_mobility":
            if "low" in t and value < t["low"]:
                reasons.append(e["low"])
            elif "medium" in t and value < t["medium"]:
                reasons.append(e["medium"])
        else:
            if "high" in t and value > t["high"]:
                reasons.append(e["high"])
            elif "medium" in t and value > t["medium"]:
                reasons.append(e["medium"])

    for feat in ["entropy", "coeff_variation", "skewness",
                 "excess_kurtosis", "hjorth_mobility", "mean_iki"]:
        _check(feat, feature_vector.get(feat))

    return reasons


# ── Batch explainer ────────────────────────────────────────────────────────────

def explain_batch(feature_df, fatigue_scores=None) -> list[str]:
    """
    Generate short explanations for every row of a feature DataFrame.

    Parameters
    ----------
    feature_df   : pd.DataFrame  — rows = windows, cols = feature names
    fatigue_scores : array-like or None

    Returns
    -------
    list of str  — one short explanation per row
    """
    explanations = []
    for i, row in feature_df.iterrows():
        fv = row.to_dict()
        score = float(fatigue_scores[i]) if fatigue_scores is not None else None
        explanations.append(explain_short(fv))
    return explanations


if __name__ == "__main__":
    # Quick smoke-test
    test_fv = {
        "entropy": 3.8,
        "coeff_variation": 0.62,
        "skewness": 1.7,
        "excess_kurtosis": 0.4,
        "hjorth_mobility": 1.1,
        "mean_iki": 145.0,
        "std_iki": 60.0,
    }
    print(explain_prediction(test_fv, fatigue_score=74.3))
    print()
    print("Short:", explain_short(test_fv))
