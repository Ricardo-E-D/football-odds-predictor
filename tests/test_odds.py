import numpy as np

from models.odds import implied_probabilities


def test_probabilities_sum_to_one_and_overround_exceeds_one():
    p_h, p_d, p_a, overround = implied_probabilities(2.10, 3.40, 3.60)
    assert np.isclose(p_h + p_d + p_a, 1.0)
    assert overround > 1.0
    assert p_h > p_a  # shorter price = higher probability


def test_vectorized_input():
    p_h, _, _, over = implied_probabilities(
        np.array([2.0, 1.5]), np.array([3.5, 4.0]), np.array([3.8, 6.5])
    )
    assert p_h.shape == (2,)
    assert (over > 1.0).all()
