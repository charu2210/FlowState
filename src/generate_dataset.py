"""
generate_dataset.py — Synthetic but physiologically-grounded dataset generator.

Generates keystroke IKI data for 3 users x 10 sessions each (5 normal, 5 fatigue).
Fatigue sessions have statistically distinct distributions (higher mean IKI,
higher CV, higher entropy, lower Hjorth mobility) based on empirical keystroke
fatigue literature (Epp et al., 2011; Gao et al., 2020).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from features import extract_feature_vector, FEATURE_NAMES
from config import DATA_DIR

rng = np.random.default_rng(42)

USERS = {
    "P01": {"base_iki": 130, "base_cv": 0.22},  # fast typist
    "P02": {"base_iki": 175, "base_cv": 0.28},  # medium typist
    "P03": {"base_iki": 210, "base_cv": 0.32},  # slow typist
}

WINDOW_SIZE = 50
WINDOWS_PER_SESSION = 80  # 80 windows per session

def gen_ikis(base_iki, base_cv, fatigue: bool, rng):
    """Generate IKI window with physiologically plausible fatigue distortion."""
    if fatigue:
        mean  = base_iki * rng.uniform(1.18, 1.35)   # slower
        cv    = base_cv  * rng.uniform(1.30, 1.70)   # more variable
        skew_noise = rng.exponential(0.3)             # occasional long pauses
    else:
        mean  = base_iki * rng.uniform(0.92, 1.08)
        cv    = base_cv  * rng.uniform(0.85, 1.15)
        skew_noise = 0.0

    std = mean * cv
    ikis = rng.normal(mean, std, WINDOW_SIZE)
    # Add right-skew hesitation pauses in fatigue
    if fatigue:
        n_pauses = rng.integers(1, 6)
        pause_idx = rng.integers(0, WINDOW_SIZE, n_pauses)
        ikis[pause_idx] += rng.exponential(mean * 0.6, n_pauses)
    ikis = np.clip(ikis, 30, 1200)
    return ikis

rows = []
for uid, params in USERS.items():
    base_iki = params["base_iki"]
    base_cv  = params["base_cv"]
    for sess_idx in range(10):
        is_fatigue = sess_idx >= 5
        label = -1 if is_fatigue else 1
        for w in range(WINDOWS_PER_SESSION):
            ikis = gen_ikis(base_iki, base_cv, is_fatigue, rng)
            fv = extract_feature_vector(ikis)
            row = {"user_id": uid, "session": sess_idx,
                   "window": w, "label": label, **fv}
            rows.append(row)

df = pd.DataFrame(rows)

# Save full multi-user dataset
DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(DATA_DIR / "multi_user_sessions.csv", index=False)

# Save labeled_sessions.csv (aggregate, all users) for train.py
df[FEATURE_NAMES + ["label"]].to_csv(DATA_DIR / "labeled_sessions.csv", index=False)

print(f"Generated {len(df)} windows across {len(USERS)} users x 10 sessions")
print(f"Normal: {(df.label==1).sum()}  Fatigue: {(df.label==-1).sum()}")
print(f"Files: data/multi_user_sessions.csv, data/labeled_sessions.csv")
