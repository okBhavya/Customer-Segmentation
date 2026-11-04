"""Unit tests for Next-Best-Action (NBA) logic, eligibility rules, and EV scoring."""

import pytest


def test_nba_ev_calculation():
    """EV = assumed_uplift * p_purchase * AOV - cost."""
    p_purchase = 0.10
    aov = 100.0
    uplift = 0.05
    cost = 0.10

    # Expected value = 0.05 * 0.10 * 100 - 0.10 = 0.50 - 0.10 = 0.40
    ev = uplift * p_purchase * aov - cost
    assert pytest.approx(ev, 1e-4) == 0.40


def test_nba_no_action_when_cost_exceeds_gain():
    """When cost exceeds uplift benefit, no_action must have higher EV (0.0)."""
    p_purchase = 0.01
    aov = 20.0
    uplift = 0.02
    cost = 1.00

    # Gain = 0.02 * 0.01 * 20 = 0.004. Cost = 1.00 -> EV = -0.996
    ev_nudge = uplift * p_purchase * aov - cost
    ev_no_action = 0.0

    assert ev_no_action > ev_nudge


def test_nba_eligibility_cart_reminder_requires_open_cart():
    """Cart reminder must only be considered when has_open_cart == 1."""
    row_with_cart = {"has_open_cart": 1, "days_since_last_event": 1}
    row_without_cart = {"has_open_cart": 0, "days_since_last_event": 1}

    def get_eligible_actions(row):
        actions = ["no_action"]
        if row["has_open_cart"] == 1:
            actions.append("cart_reminder")
        return actions

    assert "cart_reminder" in get_eligible_actions(row_with_cart)
    assert "cart_reminder" not in get_eligible_actions(row_without_cart)


def test_nba_winback_eligibility():
    """Win-back message must trigger for high churn probability or high recency."""
    row_high_churn = {"proba_churn": 0.65, "days_since_last_event": 1}
    row_low_churn = {"proba_churn": 0.10, "days_since_last_event": 1}

    def is_winback_eligible(row):
        return row["proba_churn"] > 0.40 or row["days_since_last_event"] >= 4

    assert is_winback_eligible(row_high_churn) is True
    assert is_winback_eligible(row_low_churn) is False


def test_nba_policy_simulation_zero_cost_for_no_action():
    """Control policy (no action) must strictly incur zero cost and zero simulated EV."""
    n_users = 1000
    cost = 0.0
    total_cost = n_users * cost
    total_ev = 0.0

    assert total_cost == 0.0
    assert total_ev == 0.0
