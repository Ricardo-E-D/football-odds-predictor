"""Walk-forward backtest: refit monthly, predict only strictly future matches.

For each league and each calendar month in the test window, the model is
fit on every match played *before* that month, then predicts that month's
matches. This mirrors live use and rules out look-ahead leakage.
"""

import numpy as np
import pandas as pd

from models.poisson import PoissonModel

MIN_TRAIN_MATCHES = 800  # ~2 seasons; below this, ratings are too noisy to score


def walk_forward(matches: pd.DataFrame, xi: float = 0.0019,
                 start: str = "2013-08-01", verbose: bool = True) -> pd.DataFrame:
    """Return one row per predicted match with model probs, book odds, and result.

    `known` marks matches where both teams appeared in the training data;
    for the rest the model falls back to league-average ratings and its
    prediction carries no team-specific information.
    """
    start_ts = pd.Timestamp(start)
    out = []

    for league, lg in matches.groupby("league"):
        lg = lg.sort_values("date")
        test_months = lg.loc[lg["date"] >= start_ts, "date"].dt.to_period("M").unique()

        for month in test_months:
            month_start = month.to_timestamp()
            train = lg[lg["date"] < month_start]
            if len(train) < MIN_TRAIN_MATCHES:
                continue
            model = PoissonModel(xi=xi).fit(train)

            chunk = lg[(lg["date"] >= month_start) & (lg["date"] < month_start + pd.offsets.MonthBegin(1))]
            for m in chunk.itertuples():
                p_h, p_d, p_a = model.match_probabilities(m.home, m.away)
                out.append({
                    "date": m.date, "league": league, "home": m.home, "away": m.away,
                    "result": m.result,
                    "p_h": p_h, "p_d": p_d, "p_a": p_a,
                    "odds_h": m.b365_h, "odds_d": m.b365_d, "odds_a": m.b365_a,
                    "known": model.knows(m.home) and model.knows(m.away),
                })
        if verbose:
            print(f"{league}: {len(test_months)} months backtested")

    return pd.DataFrame(out)
