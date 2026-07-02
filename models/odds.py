"""Convert decimal bookmaker odds to implied probabilities."""

import numpy as np


def implied_probabilities(odds_h, odds_d, odds_a):
    """Return (p_home, p_draw, p_away, overround) from decimal odds.

    Raw inverse odds sum to more than 1 — the excess is the bookmaker's
    margin (overround). The returned probabilities are normalized to sum
    to 1, which is the market's actual opinion with the margin removed.
    Accepts scalars or numpy/pandas arrays.
    """
    raw_h = 1.0 / np.asarray(odds_h, dtype=float)
    raw_d = 1.0 / np.asarray(odds_d, dtype=float)
    raw_a = 1.0 / np.asarray(odds_a, dtype=float)
    overround = raw_h + raw_d + raw_a
    return raw_h / overround, raw_d / overround, raw_a / overround, overround
