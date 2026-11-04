"""Next-best-action (NBA) engine with expected value scoring and policy simulation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_nba(cfg: dict, root: Path):
    """Run next-best-action allocation and policy simulation on test snapshot."""
    features_dir = root / cfg["paths"]["features_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]
    figures_dir = root / cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    nba_cfg = cfg.get("nba", {})

    test_path = features_dir / "test_segments.parquet"
    if not test_path.exists():
        test_path = features_dir / "test_features.parquet"

    test_df = pd.read_parquet(test_path)
    print(f"  Test users for NBA: {len(test_df):,}")

    # Merge calibrated purchase and churn probabilities
    p_purch_path = features_dir / "test_purchase_probas.parquet"
    p_churn_path = features_dir / "test_churn_probas.parquet"

    if p_purch_path.exists():
        p_purch = pd.read_parquet(p_purch_path)
        test_df = test_df.merge(p_purch, on="user_id", how="left")
    else:
        test_df["proba_purchase"] = test_df.get("purchase_count", 0).clip(upper=1) * 0.1

    if p_churn_path.exists():
        p_churn = pd.read_parquet(p_churn_path)
        test_df = test_df.merge(p_churn, on="user_id", how="left")
    else:
        test_df["proba_churn"] = (test_df.get("days_since_last_event", 0) > 5).astype(float) * 0.3

    test_df["proba_purchase"] = test_df["proba_purchase"].fillna(
        test_df["proba_purchase"].mean() or 0.05
    )
    test_df["proba_churn"] = test_df["proba_churn"].fillna(test_df["proba_churn"].mean() or 0.20)

    # Base AOV or estimated order value
    global_aov = test_df.loc[test_df["avg_order_value"] > 0, "avg_order_value"].median()
    if np.isnan(global_aov) or global_aov <= 0:
        global_aov = 50.0
    test_df["user_aov"] = np.where(
        test_df["avg_order_value"] > 0, test_df["avg_order_value"], global_aov
    )

    actions = nba_cfg.get(
        "actions",
        [
            {"name": "no_action", "assumed_uplift": 0.0, "cost": 0.0},
            {"name": "cart_reminder", "assumed_uplift": 0.05, "cost": 0.10},
            {"name": "category_cross_sell", "assumed_uplift": 0.03, "cost": 0.50},
            {"name": "price_led_nudge", "assumed_uplift": 0.04, "cost": 1.00},
            {"name": "loyalty_early_access", "assumed_uplift": 0.02, "cost": 0.25},
            {"name": "win_back_message", "assumed_uplift": 0.06, "cost": 0.75},
        ],
    )
    action_dict = {a["name"]: a for a in actions}

    # Evaluate Model-based Expected Value (EV) Policy
    print("  Scoring actions with EV policy...")
    p_p_arr = test_df["proba_purchase"].values
    p_c_arr = test_df["proba_churn"].values
    aov_arr = test_df["user_aov"].values
    has_cart_arr = (
        test_df["has_open_cart"].values
        if "has_open_cart" in test_df.columns
        else np.zeros(len(test_df))
    )
    recency_arr = (
        test_df["days_since_last_event"].values
        if "days_since_last_event" in test_df.columns
        else np.zeros(len(test_df))
    )
    seg_col = (
        "segment_kmeans"
        if "segment_kmeans" in test_df.columns
        else ("rfm_segment" if "rfm_segment" in test_df.columns else None)
    )
    seg_arr = (
        test_df[seg_col].astype(str).values if seg_col else np.array(["unknown"] * len(test_df))
    )
    cat_cnt_arr = (
        test_df["unique_categories"].values
        if "unique_categories" in test_df.columns
        else np.zeros(len(test_df))
    )
    price_rel_arr = (
        test_df["price_relative_to_category_median"].values
        if "price_relative_to_category_median" in test_df.columns
        else np.ones(len(test_df))
    )

    allocated_actions = []
    reason_codes = []
    expected_values = []

    for p_p, p_c, aov, has_cart, recency, segment, cat_cnt, price_rel in zip(
        p_p_arr,
        p_c_arr,
        aov_arr,
        has_cart_arr,
        recency_arr,
        seg_arr,
        cat_cnt_arr,
        price_rel_arr,
        strict=False,
    ):
        eligible_actions = ["no_action"]
        reasons = {"no_action": "DEFAULT_BASELINE"}

        # Rule 1: Cart reminder
        if has_cart == 1:
            eligible_actions.append("cart_reminder")
            reasons["cart_reminder"] = "OPEN_CART_DETECTED"

        # Rule 2: Win-back message
        if p_c > 0.40 or recency >= 4:
            eligible_actions.append("win_back_message")
            reasons["win_back_message"] = "HIGH_CHURN_RISK"

        # Rule 3: Category cross-sell
        if cat_cnt >= 2 or p_p > 0.15:
            eligible_actions.append("category_cross_sell")
            reasons["category_cross_sell"] = "MULTI_CATEGORY_AFFINITY"

        # Rule 4: Price-led nudge
        if price_rel < 0.90 or segment in ["window_shoppers", "need_attention", "0", "4"]:
            eligible_actions.append("price_led_nudge")
            reasons["price_led_nudge"] = "PRICE_SENSITIVE_BEHAVIOR"

        # Rule 5: Loyalty early access
        if p_p > 0.25 or segment in ["champions", "loyal", "2", "3"]:
            eligible_actions.append("loyalty_early_access")
            reasons["loyalty_early_access"] = "HIGH_VALUE_OR_LOYAL"

        # Compute Expected Value (EV) for eligible actions:
        # EV = assumed_uplift * p_purchase * AOV - cost
        best_act = "no_action"
        best_ev = 0.0

        for act in eligible_actions:
            cfg_act = action_dict.get(act, {"assumed_uplift": 0.0, "cost": 0.0})
            uplift = cfg_act["assumed_uplift"]
            cost = cfg_act["cost"]
            # Churn-saving actions factor in retained value
            if act == "win_back_message":
                ev = uplift * p_c * aov * 0.5 - cost
            else:
                ev = uplift * p_p * aov - cost

            if ev > best_ev:
                best_ev = ev
                best_act = act

        allocated_actions.append(best_act)
        reason_codes.append(reasons.get(best_act, "BEST_EXPECTED_VALUE"))
        expected_values.append(round(best_ev, 3))

    test_df["nba_action"] = allocated_actions
    test_df["nba_reason"] = reason_codes
    test_df["nba_ev"] = expected_values

    # Policy Simulations
    print("  Running policy simulations...")
    simulations = {}

    # Policy 1: No Action
    simulations["no_action"] = {
        "description": "Control group: no interventions sent",
        "total_cost": 0.0,
        "total_simulated_ev": 0.0,
        "avg_ev_per_user": 0.0,
        "actions_taken": 0,
    }

    # Policy 2: Blanket Discount / Price Nudge to all
    cost_pn = action_dict.get("price_led_nudge", {}).get("cost", 1.0)
    up_pn = action_dict.get("price_led_nudge", {}).get("assumed_uplift", 0.04)
    tot_cost_blanket = len(test_df) * cost_pn
    tot_ev_blanket = (test_df["proba_purchase"] * up_pn * test_df["user_aov"] - cost_pn).sum()
    simulations["blanket_price_nudge"] = {
        "description": "Send price-led nudge to 100% of audience",
        "total_cost": round(float(tot_cost_blanket), 2),
        "total_simulated_ev": round(float(tot_ev_blanket), 2),
        "avg_ev_per_user": round(float(tot_ev_blanket / len(test_df)), 4),
        "actions_taken": len(test_df),
    }

    # Policy 3: Random Assignment among eligible
    rng = np.random.RandomState(cfg.get("seed", 42))
    non_null_acts = [a["name"] for a in actions if a["name"] != "no_action"]
    rand_acts = rng.choice(non_null_acts, size=len(test_df))
    rand_costs = np.array([action_dict[a]["cost"] for a in rand_acts])
    rand_uplifts = np.array([action_dict[a]["assumed_uplift"] for a in rand_acts])
    rand_ev = (
        test_df["proba_purchase"].values * rand_uplifts * test_df["user_aov"].values - rand_costs
    ).sum()
    simulations["random_allocation"] = {
        "description": "Random action chosen among non-control actions",
        "total_cost": round(float(rand_costs.sum()), 2),
        "total_simulated_ev": round(float(rand_ev), 2),
        "avg_ev_per_user": round(float(rand_ev / len(test_df)), 4),
        "actions_taken": len(test_df),
    }

    # Policy 4: Segment-Rule Heuristic
    # Champions/Loyal -> loyalty perk; Open Cart -> cart reminder; At-Risk -> win-back; Else -> cross-sell
    rfm_col = "rfm_segment" if "rfm_segment" in test_df.columns else None
    rfm_arr = (
        test_df[rfm_col].astype(str).values if rfm_col else np.array(["unknown"] * len(test_df))
    )

    seg_costs, seg_evs, seg_n = 0.0, 0.0, 0
    for p_p, p_c, aov, seg, has_cart in zip(
        p_p_arr, p_c_arr, aov_arr, rfm_arr, has_cart_arr, strict=False
    ):
        if has_cart:
            act = "cart_reminder"
        elif seg in ["champions", "loyal"]:
            act = "loyalty_early_access"
        elif p_c > 0.5:
            act = "win_back_message"
        else:
            act = "category_cross_sell"

        c = action_dict[act]["cost"]
        up = action_dict[act]["assumed_uplift"]
        ev = (up * p_p * aov - c) if act != "win_back_message" else (up * p_c * aov * 0.5 - c)
        seg_costs += c
        seg_evs += ev
        seg_n += 1

    simulations["segment_rules"] = {
        "description": "Rule-based routing using segment and cart state",
        "total_cost": round(float(seg_costs), 2),
        "total_simulated_ev": round(float(seg_evs), 2),
        "avg_ev_per_user": round(float(seg_evs / len(test_df)), 4),
        "actions_taken": seg_n,
    }

    # Policy 5: Model-driven Expected Value Policy
    model_costs = sum(action_dict[a]["cost"] for a in test_df["nba_action"])
    model_ev = float(test_df["nba_ev"].sum())
    simulations["model_expected_value"] = {
        "description": "Calibrated propensity model with eligibility and cost-benefit EV optimization",
        "total_cost": round(float(model_costs), 2),
        "total_simulated_ev": round(float(model_ev), 2),
        "avg_ev_per_user": round(float(model_ev / len(test_df)), 4),
        "actions_taken": int((test_df["nba_action"] != "no_action").sum()),
    }

    # Sensitivity Analysis on Uplift & Cost
    print("  Running sensitivity analysis...")
    action_counts_map = test_df["nba_action"].value_counts().to_dict()
    mean_p_p = float(test_df["proba_purchase"].mean())
    sensitivity = []
    for uplift_mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for cost_mult in [0.5, 1.0, 1.5]:
            sim_ev = 0.0
            for act_name, count in action_counts_map.items():
                if act_name == "no_action":
                    continue
                c = action_dict[act_name]["cost"] * cost_mult
                up = action_dict[act_name]["assumed_uplift"] * uplift_mult
                sim_ev += count * (up * mean_p_p * global_aov - c)
            sensitivity.append(
                {
                    "uplift_multiplier": uplift_mult,
                    "cost_multiplier": cost_mult,
                    "total_simulated_ev": round(float(sim_ev), 2),
                }
            )

    # Action distribution
    action_counts = test_df["nba_action"].value_counts().to_dict()
    reason_counts = test_df["nba_reason"].value_counts().to_dict()

    nba_report = {
        "n_users_evaluated": len(test_df),
        "action_distribution": action_counts,
        "reason_distribution": reason_counts,
        "simulations": simulations,
        "sensitivity_analysis": sensitivity,
        "disclaimer": (
            "IMPORTANT: All figures are simulated offline estimates under stated cost and "
            "uplift assumptions. No customer interventions were logged in the raw dataset. "
            "These metrics represent expected value model comparisons, not measured A/B test lift."
        ),
    }

    with open(reports_dir / "nba_report.json", "w") as f:
        json.dump(nba_report, f, indent=2)

    # Save NBA allocations for sampled users (for Streamlit app)
    out_cols = [
        "user_id",
        "nba_action",
        "nba_reason",
        "nba_ev",
        "proba_purchase",
        "proba_churn",
        "user_aov",
        "days_since_last_event",
        "total_events",
        "total_spend",
    ]
    avail_cols = [c for c in out_cols if c in test_df.columns]
    test_df[avail_cols].to_parquet(features_dir / "test_nba_allocated.parquet", index=False)

    # Plot policy comparison
    _plot_policy_simulation(simulations, figures_dir)
    print(f"  NBA report saved to {reports_dir / 'nba_report.json'}")


def _plot_policy_simulation(simulations: dict, figures_dir: Path):
    policies = list(simulations.keys())
    evs = [simulations[p]["total_simulated_ev"] for p in policies]
    labels = [p.replace("_", " ").title() for p in policies]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#888888", "#d95f02", "#7570b3", "#1b9e77", "#2b5c8f"]
    bars = ax.bar(labels, evs, color=colors[: len(labels)], edgecolor="white")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"${h:,.0f}",
            (b.get_x() + b.get_width() / 2, h),
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=9,
        )
    ax.set_ylabel("Total Simulated Expected Value ($)")
    ax.set_title("Policy Simulation: Expected Value Comparison (Offline Estimate)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "nba_policy_simulation.png", dpi=150)
    plt.close()
