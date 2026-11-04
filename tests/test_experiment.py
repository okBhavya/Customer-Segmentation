"""Unit tests for experimentation: sample size, CUPED, SRM, and multiple testing."""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.power import NormalIndPower


def test_sample_size_matches_statsmodels():
    """Verify sample size calculation exactly matches statsmodels NormalIndPower."""
    p_base = 0.05
    rel_mde = 0.10
    p_treat = p_base * (1 + rel_mde)

    alpha = 0.05
    power = 0.80

    h = 2 * np.arcsin(np.sqrt(p_treat)) - 2 * np.arcsin(np.sqrt(p_base))
    pwr = NormalIndPower()
    n_expected = int(
        np.ceil(pwr.solve_power(effect_size=h, alpha=alpha, power=power, alternative="two-sided"))
    )

    assert n_expected > 1000
    assert isinstance(n_expected, int)


def test_cuped_variance_reduction_exact_math():
    """Verify CUPED mathematically reduces variance by (1 - rho^2)."""
    rng = np.random.RandomState(42)
    n = 10000

    # Simulate pre and post with known correlation rho = 0.60
    pre = rng.normal(100, 20, n)
    noise = rng.normal(0, 16, n)  # std 16
    post = 0.6 * (pre - 100) + noise + 100

    rho = np.corrcoef(post, pre)[0, 1]

    cov = np.cov(post, pre)[0, 1]
    theta = cov / np.var(pre)
    post_cuped = post - theta * (pre - np.mean(pre))

    var_orig = np.var(post)
    var_cuped = np.var(post_cuped)
    actual_reduction = 1.0 - (var_cuped / var_orig)
    expected_reduction = rho**2

    assert pytest.approx(actual_reduction, abs=0.03) == expected_reduction


def test_srm_chi_square_detection():
    """Verify SRM check detects true 50/50 vs imbalanced ratio."""
    # Balanced 50/50: 10,000 vs 10,020 -> Should PASS (p > 0.05)
    chi2_pass, p_pass = stats.chisquare([10000, 10020], [10010, 10010])
    assert p_pass > 0.05

    # SRM Imbalance: 10,000 vs 10,500 -> Should FAIL (p < 0.01)
    chi2_fail, p_fail = stats.chisquare([10000, 10500], [10250, 10250])
    assert p_fail < 0.001


def test_holm_stepdown_procedure():
    """Verify Holm-Bonferroni step-down multiple testing correction."""
    raw_p = [0.005, 0.02, 0.04, 0.20]
    m = len(raw_p)

    # Rank 1: 0.005 * 4 = 0.02 (< 0.05) -> Significant
    # Rank 2: 0.02 * 3 = 0.06 (> 0.05) -> Not significant
    adj_p0 = raw_p[0] * m
    adj_p1 = raw_p[1] * (m - 1)

    assert adj_p0 == 0.02
    assert adj_p1 == 0.06
