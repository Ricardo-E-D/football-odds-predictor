"""Backtest summary for the UI/API, precomputed to JSON.

The full predictions CSV (and the raw data behind it) never ship to the
server — `python -m backend.summary` exports a small JSON that gets
committed and deployed instead.
"""

import json

import numpy as np
import pandas as pd

from backend.config import LEAGUES, PROCESSED_DIR
from models.metrics import brier_score, log_loss, outcome_index
from models.odds import implied_probabilities

SUMMARY_FILE = PROCESSED_DIR / "backtest_summary.json"
PREDICTIONS_CSV = PROCESSED_DIR / "backtest_predictions.csv"


def compute() -> dict:
    preds = pd.read_csv(PREDICTIONS_CSV)
    preds = preds[preds["known"]]
    y = outcome_index(preds["result"])
    p_model = preds[["p_h", "p_d", "p_a"]].to_numpy()
    bh, bd, ba, _ = implied_probabilities(preds["odds_h"], preds["odds_d"], preds["odds_a"])
    p_book = np.column_stack([bh, bd, ba])

    per_league = []
    for code, g in preds.groupby("league"):
        gy = outcome_index(g["result"])
        gm = g[["p_h", "p_d", "p_a"]].to_numpy()
        gbh, gbd, gba, _ = implied_probabilities(g["odds_h"], g["odds_d"], g["odds_a"])
        per_league.append({
            "league": code, "name": LEAGUES[code][0], "matches": len(g),
            "model_logloss": round(log_loss(gy, gm), 4),
            "book_logloss": round(log_loss(gy, np.column_stack([gbh, gbd, gba])), 4),
        })

    return {
        "available": True,
        "matches": len(preds),
        "model_logloss": round(log_loss(y, p_model), 4),
        "book_logloss": round(log_loss(y, p_book), 4),
        "model_brier": round(brier_score(y, p_model), 4),
        "book_brier": round(brier_score(y, p_book), 4),
        "verdict": "The bookmaker beats the model in every league tested. "
                   "Value betting on model-market disagreements lost ~9.5% ROI "
                   "over 44k simulated bets (2013-2026). No edge.",
        "per_league": per_league,
    }


def export() -> dict:
    result = compute()
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {SUMMARY_FILE}")
    return result


def load() -> dict:
    """Prefer the committed JSON; fall back to computing from the local CSV."""
    if SUMMARY_FILE.exists():
        return json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    if PREDICTIONS_CSV.exists():
        return compute()
    return {"available": False}


if __name__ == "__main__":
    export()
