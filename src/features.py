"""
FlowState: Keystroke Dynamics Cognitive Load Estimator
=======================================================
features.py — Research-grade feature engineering pipeline.

Each function accepts a 1-D NumPy array of flight-time values
(inter-keystroke intervals in milliseconds) and returns a scalar.

Feature rationale
-----------------
- Coefficient of Variation (CV): Scale-normalised dispersion.  Rhythm
  degradation under fatigue increases relative variability independently
  of baseline typing speed.

- Skewness: Cognitive overload induces right-skewed distributions via
  sporadic long pauses (hesitation events); a baseline rested typist
  exhibits near-zero skew.

- Shannon Entropy (binned): Captures uncertainty / unpredictability of
  the rhythm.  High entropy ↔ disorganised, fatigue-driven typing.

- Hjorth Mobility: Borrowed from EEG signal processing; quantifies the
  mean frequency of the flight-time series.  Fatigue-induced slowdowns
  lower mobility monotonically.

- Kurtosis: Measures heavy-tailedness.  Sudden extreme pauses push
  kurtosis above the Gaussian baseline (3.0), flagging attentional lapses.

References
----------
Revett, K. et al. (2010). A survey of user authentication based on
mouse dynamics. Proc. ICCSA.

Epp, C. et al. (2011). Identifying emotional states using keystroke
dynamics. Proc. ACM CHI.
"""

import numpy as np
from scipy import stats
from scipy.signal import welch
from config import ENTROPY_BINS


# ── Primitive helpers ─────────────────────────────────────────────────────────

def _validate(arr: np.ndarray, min_len: int = 2) -> np.ndarray:
    """Cast to float64 and assert minimum length."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size < min_len:
        raise ValueError(f"Window too short: got {arr.size}, need ≥ {min_len}.")
    return arr


# ── Core feature functions ─────────────────────────────────────────────────────

def coefficient_of_variation(flight_times: np.ndarray) -> float:
    """
    Coefficient of Variation (CV) = σ / μ

    Normalises dispersion by the mean, allowing cross-session comparison
    regardless of individual baseline typing speed.  CV > 0.5 is
    empirically associated with cognitive overload episodes.

    Parameters
    ----------
    flight_times : array-like of float
        Inter-keystroke intervals (ms) for one rolling window.

    Returns
    -------
    float
        Dimensionless ratio ∈ [0, ∞).  Returns NaN if mean ≈ 0.
    """
    arr = _validate(flight_times)
    mean = arr.mean()
    if np.isclose(mean, 0.0):
        return np.nan
    return float(arr.std(ddof=1) / mean)


def distribution_skewness(flight_times: np.ndarray) -> float:
    """
    Fisher-Pearson standardised third-moment skewness.

    Positive skew indicates a right-heavy tail — a marker of sporadic
    long hesitation pauses characteristic of mental fatigue.  Provides
    asymmetry information absent from variance-based metrics.

    Parameters
    ----------
    flight_times : array-like of float

    Returns
    -------
    float
        Skewness coefficient.  Baseline rested typing ≈ 0.0.
    """
    arr = _validate(flight_times, min_len=3)
    return float(stats.skew(arr, bias=False))


def shannon_entropy(flight_times: np.ndarray, bins: int = ENTROPY_BINS) -> float:
    """
    Shannon entropy H(X) = -Σ p_i · log2(p_i) over a histogram of flight times.

    Quantifies the unpredictability of the typing rhythm.  A rested typist
    exhibits low-entropy rhythmic patterns; fatigue disrupts the rhythm,
    pushing entropy toward its maximum (log2(bins)).

    Parameters
    ----------
    flight_times : array-like of float
    bins         : int
        Number of histogram bins.  Must be ≤ len(flight_times).

    Returns
    -------
    float
        Entropy in bits ∈ [0, log2(bins)].
    """
    arr = _validate(flight_times)
    counts, _ = np.histogram(arr, bins=bins)
    probs = counts / counts.sum()
    # Avoid log(0) by masking zero-probability bins
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def hjorth_mobility(flight_times: np.ndarray) -> float:
    """
    Hjorth Mobility = sqrt( Var(dx/dt) / Var(x) )

    Originally a descriptor for EEG complexity analysis (Hjorth, 1970),
    here applied to the flight-time signal as a proxy for mean temporal
    frequency of rhythm changes.  Fatigue causes a measurable decrease in
    mobility as transitions between fast and slow keystrokes become rarer.

    Parameters
    ----------
    flight_times : array-like of float

    Returns
    -------
    float
        Mobility value ≥ 0.
    """
    arr = _validate(flight_times, min_len=3)
    diff = np.diff(arr)
    var_x  = np.var(arr,  ddof=1)
    var_dx = np.var(diff, ddof=1)
    if np.isclose(var_x, 0.0):
        return np.nan
    return float(np.sqrt(var_dx / var_x))


def excess_kurtosis(flight_times: np.ndarray) -> float:
    """
    Fisher excess kurtosis = (fourth central moment / σ^4) − 3.

    Captures heavy-tailedness beyond what skewness detects.  Sudden extreme
    pauses (attentional lapses) produce super-Gaussian distributions
    (kurtosis > 0).  Complements skewness to fully characterise tail behaviour.

    Parameters
    ----------
    flight_times : array-like of float

    Returns
    -------
    float
        Excess kurtosis.  Gaussian baseline = 0.0.
    """
    arr = _validate(flight_times, min_len=4)
    return float(stats.kurtosis(arr, fisher=True, bias=False))


# ── Feature vector assembly ────────────────────────────────────────────────────

FEATURE_NAMES = [
    "mean_iki",            # baseline speed anchor
    "std_iki",             # raw dispersion
    "coeff_variation",     # scale-normalised dispersion
    "skewness",            # distributional asymmetry
    "entropy",             # rhythmic unpredictability
    "hjorth_mobility",     # mean temporal frequency proxy
    "excess_kurtosis",     # heavy-tail / lapse detector
]


def extract_feature_vector(flight_times: np.ndarray) -> dict:
    """
    Extract the complete FlowState feature vector from a window of
    inter-keystroke intervals.

    Parameters
    ----------
    flight_times : array-like of float
        Raw IKI values (ms) for one rolling window.

    Returns
    -------
    dict[str, float]
        Ordered mapping of feature name → value, NaN where undefined.
    """
    arr = _validate(flight_times)
    return {
        "mean_iki":         float(arr.mean()),
        "std_iki":          float(arr.std(ddof=1)),
        "coeff_variation":  coefficient_of_variation(arr),
        "skewness":         distribution_skewness(arr),
        "entropy":          shannon_entropy(arr),
        "hjorth_mobility":  hjorth_mobility(arr),
        "excess_kurtosis":  excess_kurtosis(arr),
    }


# ── Fatigue Score ──────────────────────────────────────────────────────────────

# Weights derived from empirical feature importance literature
# (entropy & CV dominate; kurtosis contributes least)
FATIGUE_WEIGHTS = {
    "entropy":          0.35,
    "coeff_variation":  0.30,
    "skewness":         0.20,
    "excess_kurtosis":  0.10,
    "hjorth_mobility":  0.05,   # inverted: lower mobility → higher fatigue
}

# Empirical baselines (rested-state reference values for normalisation)
_FATIGUE_BASELINES = {
    "entropy":         {"low": 1.5,  "high": 4.0},
    "coeff_variation": {"low": 0.1,  "high": 0.8},
    "skewness":        {"low": 0.0,  "high": 3.0},
    "excess_kurtosis": {"low": -1.0, "high": 10.0},
    "hjorth_mobility": {"low": 0.5,  "high": 3.0},   # high mobility = rested
}


def fatigue_score(feature_vector: dict) -> float:
    """
    Compute a continuous Fatigue Score in [0, 100].

    Each feature is min-max normalised against empirical rested/fatigued
    reference ranges, then combined via a weighted sum.  For hjorth_mobility,
    the contribution is inverted (lower mobility = higher fatigue).

    Parameters
    ----------
    feature_vector : dict
        Output of extract_feature_vector().

    Returns
    -------
    float
        Fatigue Score ∈ [0, 100].  Higher = more fatigue signal.
    """
    score = 0.0
    total_weight = 0.0

    for feat, weight in FATIGUE_WEIGHTS.items():
        val = feature_vector.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        bounds = _FATIGUE_BASELINES[feat]
        lo, hi = bounds["low"], bounds["high"]
        span = hi - lo
        if span == 0:
            continue

        # Clip and normalise to [0, 1]
        normalised = np.clip((val - lo) / span, 0.0, 1.0)

        # Mobility is inverse: high mobility = rested → low fatigue
        if feat == "hjorth_mobility":
            normalised = 1.0 - normalised

        score += weight * normalised
        total_weight += weight

    if total_weight == 0:
        return float("nan")

    return round(float(score / total_weight) * 100, 1)
