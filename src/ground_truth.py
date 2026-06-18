"""
FlowState: Keystroke Dynamics Cognitive Load Estimator
=======================================================
ground_truth.py — Rigorous Ground-Truth Acquisition Strategy

THE CORE PROBLEM
----------------
Pseudo-labels (time-block annotation) are a reasonable baseline but carry
a fundamental validity threat: the assumption that fatigue accumulates
*monotonically* and *linearly* is violated by recovery periods, task
switches, and individual differences in fatigue trajectories.

A reviewer at IISc/IIT will ask: "How do you know your labels are correct?"
This module implements two complementary strategies that together constitute
a *convergent validation* approach — each strategy's weaknesses are offset
by the other's strengths.

STRATEGY 1: NASA-TLX Self-Report Correlation
--------------------------------------------
NASA Task Load Index (Hart & Staveland, 1988) is the gold standard for
subjective cognitive load measurement.  It is:
  - Validated across 40+ years of HCI and cognitive science research
  - Freely available (public domain)
  - Accepted in all major HCI and ML venues

Protocol:
  1. Participant types for 60 minutes (free text, e.g., transcription task)
  2. Every 10 minutes, a pop-up prompts them to complete a 6-item NASA-TLX
     survey (Mental Demand, Physical Demand, Temporal Demand, Performance,
     Effort, Frustration — each 0-100)
  3. The Weighted Workload Score (WWL) is computed and mapped to a binary
     label: WWL > threshold -> label = -1 (high load)

This converts the problem from pure unsupervised to *weakly supervised*,
which is methodologically defensible at top-tier labs.

STRATEGY 2: Induced Cognitive Load Paradigm (Dual-Task)
-------------------------------------------------------
Instead of waiting for fatigue to accumulate naturally, *induce* it
experimentally at known time points:

  Phase A (Baseline):   Type a simple familiar passage. No load.
                        Ground truth: label = +1 (normal)

  Phase B (High Load):  Simultaneously type AND perform mental arithmetic
                        (e.g., serial subtraction: count down from 500 by 7).
                        Ground truth: label = -1 (high cognitive load)

  Phase C (Recovery):   Return to simple typing.
                        Ground truth: label = +1 (normal)

This yields *known-label intervals* without any subjective survey, making
it a stronger experimental design. The pattern A->B->C->B->A can be repeated
to obtain multiple label-switching events per session.

CONVERGENT VALIDATION
----------------------
Run BOTH strategies on the same session. Compute Cohen's kappa between the
two label sets. kappa > 0.60 ("substantial agreement") validates that both
strategies are measuring the same underlying construct.

References
----------
Hart, S. G., & Staveland, L. E. (1988). Development of NASA-TLX.
  Advances in Psychology, 52, 139-183.
Brouwer, A-M., et al. (2012). Estimating workload using EEG spectral
  power and ERPs in the n-back task. J Neural Eng.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

log = logging.getLogger(__name__)

# ── NASA-TLX ─────────────────────────────────────────────────────────────────

NASA_TLX_DIMENSIONS = [
    "mental_demand", "physical_demand", "temporal_demand",
    "performance", "effort", "frustration",
]

# Weights from Hart & Staveland (1988); elicit per-participant in a full study
DEFAULT_TLX_WEIGHTS = {
    "mental_demand": 0.2109, "physical_demand": 0.0977,
    "temporal_demand": 0.1641, "performance": 0.1563,
    "effort": 0.1797, "frustration": 0.1914,
}

HIGH_LOAD_THRESHOLD = 60.0


@dataclass
class TlxRating:
    """
    One NASA-TLX administration (6 subscales, 0-100 each).
    timestamp_s: session-relative time when the rating was collected.
    """
    timestamp_s: float
    ratings: dict[str, float]
    weights: dict[str, float] = field(default_factory=lambda: DEFAULT_TLX_WEIGHTS)

    def weighted_workload(self) -> float:
        """
        WWL = sum(weight_i * rating_i).
        'Performance' is inverted: high reported performance = low load.
        Returns float in [0, 100].
        """
        r = self.ratings.copy()
        r["performance"] = 100.0 - r.get("performance", 50.0)
        return sum(self.weights[d] * r.get(d, 50.0) for d in NASA_TLX_DIMENSIONS)

    def label(self, threshold: float = HIGH_LOAD_THRESHOLD) -> int:
        return -1 if self.weighted_workload() >= threshold else 1


# ── Induced-Load Protocol ─────────────────────────────────────────────────────

LoadPhase = Literal["baseline", "high_load", "recovery"]
PHASE_LABELS: dict[str, int] = {"baseline": 1, "high_load": -1, "recovery": 1}


@dataclass
class LoadInterval:
    """One phase of a dual-task induced-load session."""
    phase: LoadPhase
    start_s: float
    end_s: float

    @property
    def label(self) -> int:
        return PHASE_LABELS[self.phase]


# ── Labelling functions ───────────────────────────────────────────────────────

def label_windows_by_tlx(
    feature_df: pd.DataFrame,
    tlx_ratings: list[TlxRating],
    time_col: str = "timestamp_s",
) -> pd.DataFrame:
    """
    Assign NASA-TLX derived labels to feature windows by temporal proximity.

    Each window inherits the label of the most recent TLX rating that
    precedes it. Windows before the first rating are dropped.
    Adds columns: 'tlx_wwl' and 'label'.
    """
    if not tlx_ratings:
        raise ValueError("At least one TLX rating is required.")

    tlx_times  = np.array([r.timestamp_s for r in tlx_ratings])
    tlx_wwls   = np.array([r.weighted_workload() for r in tlx_ratings])
    tlx_labels = np.array([r.label() for r in tlx_ratings])

    df = feature_df.copy()
    df["tlx_wwl"] = np.nan
    df["label"]   = np.nan

    for i, row in df.iterrows():
        preceding = np.where(tlx_times <= row[time_col])[0]
        if len(preceding) == 0:
            continue
        idx = preceding[-1]
        df.at[i, "tlx_wwl"] = tlx_wwls[idx]
        df.at[i, "label"]   = tlx_labels[idx]

    dropped = df["label"].isna().sum()
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    log.info("TLX labels: %d windows | %d dropped | %.1f%% high-load",
             len(df), dropped, 100 * (df["label"] == -1).mean())
    return df


def label_windows_by_induced_load(
    feature_df: pd.DataFrame,
    intervals: list[LoadInterval],
    time_col: str = "timestamp_s",
) -> pd.DataFrame:
    """
    Assign ground-truth labels from the dual-task induced-load protocol.
    Windows not covered by any interval are dropped.
    """
    df = feature_df.copy()
    df["label"] = np.nan

    for interval in intervals:
        mask = (df[time_col] >= interval.start_s) & (df[time_col] < interval.end_s)
        df.loc[mask, "label"] = interval.label

    uncovered = df["label"].isna().sum()
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    log.info("Induced-load labels: %d windows | %d uncovered | %.1f%% high-load",
             len(df), uncovered, 100 * (df["label"] == -1).mean())
    return df


def label_agreement_kappa(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    name_a: str = "Strategy A",
    name_b: str = "Strategy B",
) -> float:
    """
    Compute Cohen's kappa between two labelling strategies on the same windows.

    kappa > 0.60 = substantial agreement (validates your labelling approach).
    Report this in the paper: it is a direct response to the validity concern.
    """
    kappa = cohen_kappa_score(labels_a, labels_b)
    verdict = (
        "substantial agreement — valid ✓" if kappa > 0.60 else
        "moderate agreement"              if kappa > 0.40 else
        "weak agreement — revisit labelling ✗"
    )
    log.info("Label agreement (%s vs %s): kappa = %.4f  [%s]",
             name_a, name_b, kappa, verdict)
    return kappa


def make_example_induced_session(session_duration_s: float = 3600.0) -> list[LoadInterval]:
    """
    Canonical A->B->A->B->A dual-task session (5 phases x 12 min each).
    Use this structure in your IRB protocol submission.
    """
    p = session_duration_s / 5
    return [
        LoadInterval("baseline",  0 * p, 1 * p),
        LoadInterval("high_load", 1 * p, 2 * p),
        LoadInterval("baseline",  2 * p, 3 * p),
        LoadInterval("high_load", 3 * p, 4 * p),
        LoadInterval("recovery",  4 * p, 5 * p),
    ]
