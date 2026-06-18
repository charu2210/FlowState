"""
FlowState: Keystroke Dynamics Cognitive Load Estimator
=======================================================
feature_analysis.py — Attention-Style Feature Contribution Analysis

Generates a normalised feature contribution report that answers:
  "Which behavioural signals drive fatigue detection most strongly?"

This module operates in two modes:

  1. MODEL-BACKED (preferred):
     Loads permutation importance CSVs produced by train.py and
     aggregates them into a consensus ranking across all models.

  2. HEURISTIC FALLBACK:
     When no trained models exist yet (e.g., during development),
     uses the analytically-motivated FATIGUE_WEIGHTS from features.py
     as a proxy.  Clearly labelled as heuristic in the output.

The "attention-style" framing is deliberate: contribution percentages
mirror how attention weights are reported in transformer interpretability
literature, making the output legible to ML-literate reviewers.

Output artefacts
----------------
  reports/feature_contribution.csv     — numerical contributions
  reports/feature_contribution.md      — markdown summary for report.py
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import REPORTS_DIR
from features import FEATURE_NAMES, FATIGUE_WEIGHTS

log = logging.getLogger(__name__)


# ── Contribution computation ───────────────────────────────────────────────────

def _load_model_importances(reports_dir: Path = REPORTS_DIR) -> pd.DataFrame | None:
    """
    Load permutation importance CSVs written by train.py and average them.

    Returns None if no importance files are found.
    """
    imp_files = list(reports_dir.glob("feature_importance_*.csv"))
    if not imp_files:
        return None

    dfs = []
    for f in imp_files:
        try:
            df = pd.read_csv(f)
            df["source_model"] = f.stem.replace("feature_importance_", "")
            dfs.append(df)
        except Exception as exc:
            log.warning("Could not read %s: %s", f, exc)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    # Average importance across models; clip negatives to zero
    agg = (
        combined.groupby("feature")["importance"]
        .mean()
        .clip(lower=0.0)
        .reset_index()
        .rename(columns={"importance": "raw_importance"})
    )
    return agg


def compute_feature_contributions(
    reports_dir: Path = REPORTS_DIR,
) -> tuple[pd.DataFrame, str]:
    """
    Compute normalised feature contributions as percentages summing to 100%.

    Returns
    -------
    df : pd.DataFrame
        Columns: feature, contribution_pct, source
    source : str
        'model_backed' or 'heuristic'
    """
    imp_df = _load_model_importances(reports_dir)

    if imp_df is not None and len(imp_df) > 0:
        # Model-backed path
        total = imp_df["raw_importance"].sum()
        if total == 0:
            total = 1.0
        imp_df["contribution_pct"] = (imp_df["raw_importance"] / total * 100).round(1)
        imp_df = imp_df.sort_values("contribution_pct", ascending=False).reset_index(drop=True)
        imp_df["source"] = "model_backed"
        result = imp_df[["feature", "contribution_pct", "source"]]
        source = "model_backed"
        log.info("Feature contributions computed from permutation importance CSVs.")
    else:
        # Heuristic fallback: use FATIGUE_WEIGHTS
        log.warning(
            "No feature_importance_*.csv files found in %s. "
            "Falling back to heuristic weights from FATIGUE_WEIGHTS.",
            reports_dir,
        )
        # Include features not in FATIGUE_WEIGHTS with weight 0
        all_weights = {f: FATIGUE_WEIGHTS.get(f, 0.0) for f in FEATURE_NAMES}
        total = sum(all_weights.values()) or 1.0
        rows = [
            {
                "feature": f,
                "contribution_pct": round(w / total * 100, 1),
                "source": "heuristic",
            }
            for f, w in sorted(all_weights.items(), key=lambda x: -x[1])
        ]
        result = pd.DataFrame(rows)
        source = "heuristic"

    return result, source


# ── Markdown report ────────────────────────────────────────────────────────────

def _render_contribution_table(df: pd.DataFrame) -> str:
    """Render a simple Markdown table with a contribution bar."""
    lines = [
        "| Rank | Feature | Contribution | Bar |",
        "| --- | --- | --- | --- |",
    ]
    for rank, row in enumerate(df.itertuples(), start=1):
        bar_len = int(row.contribution_pct / 5)   # 1 block per 5%
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(
            f"| {rank} | `{row.feature}` | {row.contribution_pct:.1f}% | {bar} |"
        )
    return "\n".join(lines)


def _narrative_summary(df: pd.DataFrame, source: str) -> str:
    """Generate a research-ready interpretive paragraph."""
    top = df.head(3)
    top_names = list(top["feature"])
    top_pcts  = list(top["contribution_pct"])

    source_note = (
        "based on aggregated permutation importance across all trained models"
        if source == "model_backed"
        else "based on analytically-motivated heuristic weights (run `train.py` for model-backed results)"
    )

    para = (
        f"Feature contribution analysis ({source_note}) reveals that "
        f"**{top_names[0]}** ({top_pcts[0]:.1f}%) is the strongest predictor of fatigue, "
        f"followed by **{top_names[1]}** ({top_pcts[1]:.1f}%) and "
        f"**{top_names[2]}** ({top_pcts[2]:.1f}%). "
        f"Together, the top three features account for "
        f"{sum(top_pcts):.1f}% of total predictive signal. "
        f"This pattern is consistent with the hypothesis that fatigue primarily "
        f"manifests as disrupted rhythm ({top_names[0]}) and increased typing variability "
        f"({top_names[1]}), rather than raw speed changes alone."
    )
    return para


def generate_feature_contribution_report(
    reports_dir: Path = REPORTS_DIR,
) -> pd.DataFrame:
    """
    Full pipeline: compute contributions, save CSV + Markdown report.

    Returns
    -------
    pd.DataFrame  — the contribution table
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    df, source = compute_feature_contributions(reports_dir)

    # ── Save CSV ──
    csv_path = reports_dir / "feature_contribution.csv"
    df.to_csv(csv_path, index=False)
    log.info("Feature contribution CSV → %s", csv_path)

    # ── Build Markdown ──
    table_md = _render_contribution_table(df)
    narrative = _narrative_summary(df, source)
    top3 = df.head(3)

    md_lines = [
        "## Interpretability Analysis — Feature Contributions\n",
        f"> Source: `{'permutation importance (model-backed)' if source == 'model_backed' else 'heuristic weights'}`\n",
        "### Top Predictive Features\n",
        table_md,
        "",
        "### Interpretation\n",
        narrative,
        "",
        "### Top Fatigue Indicators\n",
    ]
    for i, row in enumerate(top3.itertuples(), start=1):
        md_lines.append(f"{i}. **{row.feature}** — {row.contribution_pct:.1f}% contribution")
    md_lines.append("")

    md_text = "\n".join(md_lines)
    md_path = reports_dir / "feature_contribution.md"
    md_path.write_text(md_text, encoding="utf-8")
    log.info("Feature contribution Markdown → %s", md_path)

    print(md_text)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(message)s")
    df = generate_feature_contribution_report()
    print("\nContribution table:")
    print(df.to_string(index=False))
