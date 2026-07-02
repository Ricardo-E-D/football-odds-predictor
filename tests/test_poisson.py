import numpy as np
import pandas as pd
import pytest

from models.poisson import PoissonModel


@pytest.fixture(scope="module")
def fitted():
    """Synthetic league: Strong beats Weak often; Mid is average."""
    rng = np.random.default_rng(42)
    teams = {"Strong": 2.0, "Mid": 1.3, "Weak": 0.7}  # true scoring rates
    rows = []
    date = pd.Timestamp("2024-01-01")
    for _ in range(60):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                rows.append({
                    "date": date,
                    "home": home,
                    "away": away,
                    "hg": rng.poisson(teams[home] * 1.2),  # 1.2 = home boost
                    "ag": rng.poisson(teams[away]),
                })
                date += pd.Timedelta(hours=8)
    return PoissonModel().fit(pd.DataFrame(rows))


def test_probabilities_sum_to_one(fitted):
    p = fitted.match_probabilities("Strong", "Weak")
    assert np.isclose(sum(p), 1.0)


def test_strong_team_favoured(fitted):
    p_home, _, p_away = fitted.match_probabilities("Strong", "Weak")
    assert p_home > 0.5 > p_away


def test_home_advantage_recovered(fitted):
    assert fitted.home_adv > 0
    lam_h, _ = fitted.expected_goals("Mid", "Mid")
    _, lam_a = fitted.expected_goals("Mid", "Mid")
    assert lam_h > lam_a


def test_unknown_team_gets_average_rating(fitted):
    assert not fitted.knows("Newcomer")
    p = fitted.match_probabilities("Newcomer", "Mid")
    assert np.isclose(sum(p), 1.0)
    lam_h, _ = fitted.expected_goals("Newcomer", "Mid")
    lam_mid, _ = fitted.expected_goals("Mid", "Mid")
    # a newcomer is treated as a league-average side, weaker than "Strong"
    lam_strong, _ = fitted.expected_goals("Strong", "Mid")
    assert lam_h == pytest.approx(lam_mid, abs=1.0) and lam_h < lam_strong
