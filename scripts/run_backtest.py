"""Run the full walk-forward backtest and produce the evaluation report.

Outputs:
    data/processed/backtest_predictions.csv   one row per predicted match
    notebooks/backtest_chart.png              log-loss comparison + betting P&L
    stdout                                    metric tables and betting results

Pass --reuse to skip the backtest and re-analyze the saved predictions.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.backtest import walk_forward
from models.data import load_matches
from models.metrics import brier_score, log_loss, outcome_index
from models.odds import implied_probabilities

PREDICTIONS_CSV = ROOT / "data" / "processed" / "backtest_predictions.csv"
CHART_PNG = ROOT / "notebooks" / "backtest_chart.png"

EV_THRESHOLDS = [0.00, 0.05, 0.10]


def score_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-league (plus overall) log-loss and Brier for model vs bookmaker."""
    rows = []
    for label, g in [*preds.groupby("league"), ("ALL", preds)]:
        y = outcome_index(g["result"])
        p_model = g[["p_h", "p_d", "p_a"]].to_numpy()
        bh, bd, ba, _ = implied_probabilities(g["odds_h"], g["odds_d"], g["odds_a"])
        p_book = np.column_stack([bh, bd, ba])
        rows.append({
            "league": label, "matches": len(g),
            "model_logloss": log_loss(y, p_model), "book_logloss": log_loss(y, p_book),
            "model_brier": brier_score(y, p_model), "book_brier": brier_score(y, p_book),
        })
    table = pd.DataFrame(rows).set_index("league")
    table["logloss_gap"] = table["model_logloss"] - table["book_logloss"]
    return table


def betting_simulation(preds: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Flat-stake value betting: bet 1 unit on the outcome with the highest
    expected value, when that EV clears the threshold.

    EV per unit = model_prob * decimal_odds - 1: positive when the model
    thinks the price is generous."""
    p = preds[["p_h", "p_d", "p_a"]].to_numpy()
    odds = preds[["odds_h", "odds_d", "odds_a"]].to_numpy()
    ev = p * odds - 1.0
    pick = ev.argmax(axis=1)
    take = ev.max(axis=1) > threshold

    y = outcome_index(preds["result"])
    won = pick == y
    profit = np.where(won, odds[np.arange(len(y)), pick] - 1.0, -1.0)

    sim = preds[["date", "league"]].copy()
    sim["profit"] = np.where(take, profit, 0.0)
    sim["staked"] = take.astype(float)

    per_league = sim.groupby("league")[["profit", "staked"]].sum()
    per_league.loc["ALL"] = per_league.sum()
    per_league["roi_%"] = 100 * per_league["profit"] / per_league["staked"].replace(0, np.nan)
    per_league["threshold"] = threshold
    return per_league.round(2), sim[take]


def disagreement_calibration(preds: pd.DataFrame) -> pd.DataFrame:
    """Where model and market disagree on the home-win probability, whose
    number was closer to reality? Buckets of (model - market) difference."""
    bh, _, _, _ = implied_probabilities(preds["odds_h"], preds["odds_d"], preds["odds_a"])
    diff = preds["p_h"].to_numpy() - np.asarray(bh)
    happened = (preds["result"] == "H").to_numpy().astype(float)
    buckets = pd.cut(diff, [-1, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 1])
    table = pd.DataFrame({
        "bucket": buckets, "model_p": preds["p_h"].to_numpy(),
        "market_p": np.asarray(bh), "home_won": happened,
    }).groupby("bucket", observed=True).agg(
        n=("home_won", "size"), model_says=("model_p", "mean"),
        market_says=("market_p", "mean"), actually=("home_won", "mean"),
    )
    return table.round(3)


def make_chart(table: pd.DataFrame, pnl: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    leagues = table.drop(index="ALL")
    x = np.arange(len(leagues))
    ax1.bar(x - 0.2, leagues["model_logloss"], 0.4, label="Poisson model")
    ax1.bar(x + 0.2, leagues["book_logloss"], 0.4, label="Bet365 (implied)")
    ax1.set_xticks(x, leagues.index, rotation=45)
    ax1.set_ylabel("log-loss (lower = better)")
    ax1.set_ylim(bottom=0.9)
    ax1.set_title("Model vs bookmaker, per league")
    ax1.legend()

    daily = pnl.sort_values("date").groupby("date")["profit"].sum().cumsum()
    ax2.plot(daily.index, daily.values)
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_ylabel("cumulative profit (units, flat 1u stakes)")
    ax2.set_title("Value-betting P&L, EV threshold 5%")

    fig.tight_layout()
    fig.savefig(CHART_PNG, dpi=120)
    print(f"\nchart written to {CHART_PNG}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true", help="re-analyze saved predictions")
    args = parser.parse_args()

    if args.reuse and PREDICTIONS_CSV.exists():
        preds = pd.read_csv(PREDICTIONS_CSV, parse_dates=["date"])
    else:
        preds = walk_forward(load_matches())
        PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(PREDICTIONS_CSV, index=False)

    known = preds[preds["known"]]
    print(f"\npredictions: {len(preds):,} ({len(preds) - len(known):,} with cold-start teams)")

    table = score_table(known)
    print("\n=== scoring (known teams only) ===")
    print(table.round(4).to_string())

    print("\n=== flat-stake value betting vs Bet365 (known teams only) ===")
    pnl_for_chart = None
    for thr in EV_THRESHOLDS:
        league_table, pnl = betting_simulation(known, thr)
        overall = league_table.loc["ALL"]
        print(f"threshold {thr:.0%}: {int(overall['staked']):,} bets, "
              f"profit {overall['profit']:+.1f}u, ROI {overall['roi_%']:+.2f}%")
        if thr == 0.05:
            pnl_for_chart = pnl
            print(league_table.to_string())

    print("\n=== who is right when they disagree? (home-win probability) ===")
    print(disagreement_calibration(known).to_string())

    make_chart(table, pnl_for_chart)


if __name__ == "__main__":
    main()
