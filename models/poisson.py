"""Poisson goal model (Maher 1982) for one league.

Every team gets an attack and a defence rating; one shared home-advantage
parameter. Expected goals for the home side in a given match:

    lambda_home = exp(base + attack[home] - defence[away] + home_adv)
    lambda_away = exp(base + attack[away] - defence[home])

All parameters are fit jointly by maximum likelihood. Recent matches can be
weighted more heavily via exponential time decay (Dixon & Coles 1997): a
match played `d` days before the fit date gets weight exp(-xi * d).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10  # scoreline grid upper bound; P(>10 goals) is negligible


@dataclass
class PoissonModel:
    xi: float = 0.0  # time-decay rate per day; 0 = weight all matches equally
    base: float = field(default=0.0, init=False)
    home_adv: float = field(default=0.0, init=False)
    attack: dict = field(default_factory=dict, init=False)
    defence: dict = field(default_factory=dict, init=False)

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        """Fit on a single league's matches (columns: date, home, away, hg, ag)."""
        teams = sorted(set(matches["home"]) | set(matches["away"]))
        index = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h = matches["home"].map(index).to_numpy()
        a = matches["away"].map(index).to_numpy()
        hg = matches["hg"].to_numpy(dtype=float)
        ag = matches["ag"].to_numpy(dtype=float)
        days_ago = (matches["date"].max() - matches["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * days_ago)

        ridge = 1e-4  # tiny L2 penalty pins down the attack/defence level

        def neg_log_likelihood(params):
            att, dfc = params[:n], params[n:2 * n]
            base, home_adv = params[-2], params[-1]
            log_lh = base + att[h] - dfc[a] + home_adv
            log_la = base + att[a] - dfc[h]
            mu_h, mu_a = np.exp(log_lh), np.exp(log_la)
            # Poisson log-pmf without the constant log(k!) term
            ll = w * (hg * log_lh - mu_h + ag * log_la - mu_a)
            nll = -ll.sum() + ridge * (att @ att + dfc @ dfc)

            gh = w * (hg - mu_h)  # d(ll)/d(log_lh) per match
            ga = w * (ag - mu_a)
            grad = np.empty_like(params)
            grad[:n] = np.bincount(h, gh, n) + np.bincount(a, ga, n) - 2 * ridge * att
            grad[n:2 * n] = -np.bincount(a, gh, n) - np.bincount(h, ga, n) - 2 * ridge * dfc
            grad[-2] = gh.sum() + ga.sum()
            grad[-1] = gh.sum()
            return nll, -grad

        x0 = np.zeros(2 * n + 2)
        x0[-2] = np.log(max(hg.mean(), 0.1))
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", jac=True)
        if not result.success:
            raise RuntimeError(f"Poisson fit did not converge: {result.message}")

        # center ratings at 0 (league average); fold the shift into base
        att, dfc = result.x[:n], result.x[n:2 * n]
        att_mean, dfc_mean = att.mean(), dfc.mean()
        self.attack = {t: att[i] - att_mean for t, i in index.items()}
        self.defence = {t: dfc[i] - dfc_mean for t, i in index.items()}
        self.base = result.x[-2] + att_mean - dfc_mean
        self.home_adv = result.x[-1]
        return self

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """(lambda_home, lambda_away). Unknown teams get league-average ratings (0)."""
        att_h = self.attack.get(home, 0.0)
        def_h = self.defence.get(home, 0.0)
        att_a = self.attack.get(away, 0.0)
        def_a = self.defence.get(away, 0.0)
        lam_h = np.exp(self.base + att_h - def_a + self.home_adv)
        lam_a = np.exp(self.base + att_a - def_h)
        return float(lam_h), float(lam_a)

    def match_probabilities(self, home: str, away: str) -> tuple[float, float, float]:
        """(p_home, p_draw, p_away) by summing the scoreline probability grid."""
        lam_h, lam_a = self.expected_goals(home, away)
        goals = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
        grid /= grid.sum()  # renormalize away the tiny P(>MAX_GOALS) tail
        p_home = float(np.tril(grid, -1).sum())  # home goals > away goals
        p_draw = float(np.trace(grid))
        p_away = float(np.triu(grid, 1).sum())
        return p_home, p_draw, p_away

    def knows(self, team: str) -> bool:
        return team in self.attack
