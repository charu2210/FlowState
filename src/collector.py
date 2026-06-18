"""
FlowState: Keystroke Dynamics Cognitive Load Estimator
=======================================================
collector.py — Privacy-first keystroke dynamics collector.

Privacy Design (critical for research ethics)
---------------------------------------------
This module records ONLY inter-keystroke intervals (IKIs).  No key
identities, no text content, no character sequences are ever stored.
The architectural choice of discarding key identity at the hardware-event
layer — before any data is persisted — constitutes a Privacy-by-Design
implementation consistent with GDPR Art. 25 and the NIST Privacy Framework.

This distinguishes FlowState from conventional keylogger-based approaches
and is a core scientific contribution to Privacy-First Affective Computing.
"""

import csv
import time
import logging
from pathlib import Path
from collections import deque

import numpy as np
from pynput import keyboard

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import ROLLING_WINDOW_SIZE, MIN_WINDOW_SIZE, DATA_DIR
from features import extract_feature_vector, FEATURE_NAMES, fatigue_score as compute_fatigue_score
from explain import explain_short

logging.basicConfig(level="INFO", format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_IKI_PATH    = DATA_DIR / "raw_keystrokes.csv"
FEATURES_PATH   = DATA_DIR / "features_live.csv"


class FlowStateCollector:
    """
    Real-time keystroke IKI collector with rolling feature extraction.

    Attributes
    ----------
    window       : deque of float  — sliding window of recent IKIs (ms)
    _last_press  : float           — timestamp of most recent key-press
    _session_start: float          — session epoch (for time-block labelling)
    """

    def __init__(self, window_size: int = ROLLING_WINDOW_SIZE):
        self.window_size   = window_size
        self.window        = deque(maxlen=window_size)
        self._last_press   = None
        self._session_start = time.time()
        self._iki_writer   = None
        self._feat_writer  = None
        self._iki_file     = None
        self._feat_file    = None

    # ── File I/O setup ────────────────────────────────────────────────────────

    def _open_writers(self):
        self._iki_file  = open(RAW_IKI_PATH,  "w", newline="")
        self._feat_file = open(FEATURES_PATH, "w", newline="")

        self._iki_writer  = csv.writer(self._iki_file)
        self._feat_writer = csv.DictWriter(self._feat_file,
                                           fieldnames=["timestamp_s"] + FEATURE_NAMES + ["fatigue_score", "explanation"])
        self._iki_writer.writerow(["timestamp_s", "iki_ms"])
        self._feat_writer.writeheader()
        log.info("Recording to:\n  IKI  → %s\n  Feat → %s", RAW_IKI_PATH, FEATURES_PATH)

    def _close_writers(self):
        for f in [self._iki_file, self._feat_file]:
            if f:
                f.close()

    # ── Event handler ─────────────────────────────────────────────────────────

    def on_press(self, key):
        """
        Called on every key-press event.

        KEY IDENTITY IS DELIBERATELY DISCARDED HERE.
        Only the timing (IKI) is retained.
        """
        now = time.time()

        if self._last_press is not None:
            iki_ms = (now - self._last_press) * 1000.0

            # Filter physiologically implausible IKIs (< 30 ms or > 2000 ms)
            if 30 <= iki_ms <= 2000:
                self.window.append(iki_ms)
                elapsed = now - self._session_start

                # Log raw IKI
                self._iki_writer.writerow([round(elapsed, 4), round(iki_ms, 2)])

                # Extract features once window is sufficiently populated
                if len(self.window) >= MIN_WINDOW_SIZE:
                    features = extract_feature_vector(np.array(self.window))
                    score = compute_fatigue_score(features)
                    explanation = explain_short(features)
                    row = {"timestamp_s": round(elapsed, 4), **features,
                           "fatigue_score": score, "explanation": explanation}
                    self._feat_writer.writerow(row)
                    self._feat_file.flush()

                    log.info(
                        "t=%.1fs | CV=%.3f | Entropy=%.3f | Skew=%.3f | FatigueScore=%.1f | %s",
                        elapsed,
                        features.get("coeff_variation", float("nan")),
                        features.get("entropy",         float("nan")),
                        features.get("skewness",        float("nan")),
                        score if score == score else 0.0,   # nan-safe
                        explanation,
                    )

        self._last_press = now

    def on_release(self, key):
        if key == keyboard.Key.esc:
            log.info("ESC detected — stopping collector.")
            return False   # Terminate listener

    # ── Session runner ────────────────────────────────────────────────────────

    def start(self):
        """Block and collect until ESC is pressed."""
        self._open_writers()
        log.info("FlowState collector active.  Press ESC to stop.")
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()
        self._close_writers()
        log.info("Session complete.  Data saved.")


if __name__ == "__main__":
    FlowStateCollector().start()
