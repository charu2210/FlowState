"""
FlowState: Keystroke Dynamics Cognitive Load Estimator
=======================================================
config.py — Central configuration and hyperparameter registry.

All model hyperparameters, window sizes, and file paths are declared
here to ensure reproducibility across experimental runs.
"""

from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_DATA_PATH    = DATA_DIR / "raw_keystrokes.csv"
LABELED_DATA_PATH = DATA_DIR / "labeled_sessions.csv"

# ── Feature Extraction ────────────────────────────────────────────────────────
ROLLING_WINDOW_SIZE: int = 50          # samples per analysis window
MIN_WINDOW_SIZE:     int = 20          # minimum valid window (early-session)
ENTROPY_BINS:        int = 10          # histogram bins for Shannon entropy

# ── Model Hyperparameters ─────────────────────────────────────────────────────
ISOLATION_FOREST_PARAMS = {
    "n_estimators":   200,
    "contamination":  0.05,   # assumed ~5% anomalous windows
    "max_samples":    "auto",
    "random_state":   42,
}

ONE_CLASS_SVM_PARAMS = {
    "kernel": "rbf",
    "nu":     0.05,           # upper bound on fraction of outliers
    "gamma":  "scale",
}

LOF_PARAMS = {
    "n_neighbors":    20,
    "contamination":  0.05,
    "novelty":        True,   # enable predict() on unseen data
}

# ── Evaluation ────────────────────────────────────────────────────────────────
TEST_SIZE:       float = 0.25
RANDOM_STATE:    int   = 42
CV_FOLDS:        int   = 5

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
