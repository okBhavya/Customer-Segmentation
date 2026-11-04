"""Experimentation and evaluation framework: power analysis, CUPED, SRM, and validation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower


def run_experiment(cfg: dict, root: Path):
    """Run experiment design calculations, CUPED variance reduction, and validation."""
    features_dir = root / cfg["paths"]["features_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]
    figures_dir = root / cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    exp_cfg = cfg.get("experiment", {})
    seed = cfg["seed"]

    test_path = features_dir / "test_segments.parquet"
    if not test_path.exists():
        test_path = features_dir / "test_features.parquet"

    print("  Loading real test data for baseline estimation...")
    df = pd.read_parquet(test_path)
    print(f"  Test snapshot users: {len(df):,}")

    # Baseline statistics from real data
    p_conv = float(df["label_purchase"].mean())
    p_rev_mean = float(df["label_revenue"].mean())
    p_rev_std = float(df["label_revenue"].std())
    p_rev_var = float(df["label_revenue"].var())

    print(f"  Baseline Purchase Rate: {p_conv:.4f}")
    print(f"  Baseline Revenue/User:  ${p_rev_mean:.2f} (std: ${p_rev_std:.2f})")

    # 1. Sample Size & MDE Calculations
    alpha = exp_cfg.get("alpha", 0.05)
    power = exp_cfg.get("power", 0.80)
    power_analysis = NormalIndPower()

    mde_relative_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10]
    sample_sizes = []

    for rel_mde in mde_relative_list:
        p2 = p_conv * (1 + rel_mde)
        # Cohen's h for proportions
        h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p_conv))
        n_per_arm = int(
            np.ceil(
                power_analysis.solve_power(
                    effect_size=h, alpha=alpha, power=power, alternative="two-sided"
                )
            )
        )
        sample_sizes.append(
            {
                "relative_mde": rel_mde,
                "absolute_mde": round(p2 - p_conv, 5),
                "target_purchase_rate": round(p2, 5),
                "sample_size_per_arm": n_per_arm,
                "total_sample_size": n_per_arm * 2,
            }
        )

    # 2. CUPED Variance Reduction on Real Data
    # Pre-period metric: total_spend in lookback (or purchase_count)
    # Post-period target: label_revenue (or label_purchase)
    print("  Computing CUPED variance reduction on real pre/post periods...")
    pre = df["total_spend"].values
    post = df["label_revenue"].values

    cov = np.cov(post, pre)[0, 1]
    var_pre = np.var(pre)
    theta = cov / var_pre if var_pre > 0 else 0.0

    post_cuped = post - theta * (pre - np.mean(pre))
    var_orig = np.var(post)
    var_cuped = np.var(post_cuped)
    variance_reduction_pct = (1.0 - var_cuped / var_orig) * 100 if var_orig > 0 else 0.0
    corr_pre_post = float(np.corrcoef(post, pre)[0, 1])

    print(f"    Pre-post correlation: {corr_pre_post:.4f}")
    print(f"    CUPED Variance Reduction: {variance_reduction_pct:.2f}%")

    # 3. Sample Ratio Mismatch (SRM) Checker Validation
    srm_pass_sample = [10000, 10050]  # Expected 50/50
    srm_fail_sample = [10000, 10600]

    def _calc_srm_pval(observed, expected_ratio=(0.5, 0.5)):
        total = sum(observed)
        expected = [total * r for r in expected_ratio]
        chi2, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
        return float(chi2), float(p_val)

    chi2_pass, p_pass = _calc_srm_pval(srm_pass_sample)
    chi2_fail, p_fail = _calc_srm_pval(srm_fail_sample)

    # 4. Simulation Validation: A/A Test (Type I Error) & A/B Test (Power)
    print("  Running A/A and A/B Monte Carlo simulations (1,000 iterations)...")
    rng = np.random.RandomState(seed)
    n_sim = 1000

    # Required sample size for 80% power on 5% relative lift
    sample_size_5pct = [
        s["sample_size_per_arm"] for s in sample_sizes if s["relative_mde"] == 0.05
    ][0]

    # A/A Test (Type I error simulation under null hypothesis)
    k_aa1 = rng.binomial(sample_size_5pct, p_conv, size=n_sim)
    k_aa2 = rng.binomial(sample_size_5pct, p_conv, size=n_sim)
    p_hat_aa1 = k_aa1 / sample_size_5pct
    p_hat_aa2 = k_aa2 / sample_size_5pct
    p_pool_aa = (k_aa1 + k_aa2) / (2 * sample_size_5pct)
    se_aa = np.sqrt(2 * p_pool_aa * (1 - p_pool_aa) / sample_size_5pct)
    z_aa = (p_hat_aa2 - p_hat_aa1) / np.maximum(se_aa, 1e-10)
    aa_pvals = 2 * (1 - stats.norm.cdf(np.abs(z_aa)))
    false_positive_rate = float(np.mean(aa_pvals < alpha))

    # A/B Test with 5% relative lift (Empirical power simulation)
    injected_lift = 0.05
    p_treat = p_conv * (1 + injected_lift)
    k_ctrl = rng.binomial(sample_size_5pct, p_conv, size=n_sim)
    k_treat = rng.binomial(sample_size_5pct, p_treat, size=n_sim)
    p_hat_ctrl = k_ctrl / sample_size_5pct
    p_hat_treat = k_treat / sample_size_5pct
    p_pool_ab = (k_ctrl + k_treat) / (2 * sample_size_5pct)
    se_ab = np.sqrt(2 * p_pool_ab * (1 - p_pool_ab) / sample_size_5pct)
    z_ab = (p_hat_treat - p_hat_ctrl) / np.maximum(se_ab, 1e-10)
    ab_pvals = 2 * (1 - stats.norm.cdf(np.abs(z_ab)))
    empirical_power = float(np.mean(ab_pvals < alpha))

    print(f"    A/A False Positive Rate (target {alpha}): {false_positive_rate:.4f}")
    print(f"    A/B Empirical Power for 5% lift (target {power}): {empirical_power:.4f}")

    # 5. Multiple Testing Corrections for Segment Cuts
    raw_p_values = [0.008, 0.021, 0.045, 0.120, 0.350]
    segment_names = ["Champions", "Loyal", "At-Risk", "Window Shoppers", "Hibernating"]
    # Bonferroni-Holm step-down procedure
    sorted_pairs = sorted(zip(raw_p_values, segment_names, strict=False))
    m = len(raw_p_values)
    holm_results = []
    for rank, (raw_p, s_name) in enumerate(sorted_pairs, 1):
        adjusted_p = min(1.0, raw_p * (m - rank + 1))
        holm_results.append(
            {
                "segment": s_name,
                "raw_p_value": raw_p,
                "holm_adjusted_p": round(float(adjusted_p), 4),
                "significant_at_05": adjusted_p < alpha,
            }
        )

    exp_report = {
        "baseline_metrics": {
            "purchase_conversion_rate": round(p_conv, 4),
            "revenue_per_user_mean": round(p_rev_mean, 2),
            "revenue_per_user_std": round(p_rev_std, 2),
            "revenue_per_user_var": round(p_rev_var, 2),
        },
        "sample_size_table": sample_sizes,
        "cuped": {
            "pre_metric": "total_spend_lookback",
            "post_metric": "label_revenue_horizon",
            "correlation": round(corr_pre_post, 4),
            "variance_original": round(var_orig, 2),
            "variance_cuped": round(var_cuped, 2),
            "variance_reduction_pct": round(variance_reduction_pct, 2),
            "effective_sample_size_multiplier": round(
                1.0 / (1.0 - min(corr_pre_post**2, 0.99)), 2
            ),
        },
        "srm_validation": {
            "test_pass": {
                "observed": srm_pass_sample,
                "chi2": round(chi2_pass, 3),
                "p_value": round(p_pass, 4),
            },
            "test_fail": {
                "observed": srm_fail_sample,
                "chi2": round(chi2_fail, 3),
                "p_value": round(p_fail, 6),
            },
        },
        "simulation_validation": {
            "n_simulations": n_sim,
            "target_alpha": alpha,
            "empirical_type_1_error": round(false_positive_rate, 4),
            "target_power": power,
            "empirical_power_5pct_lift": round(empirical_power, 4),
        },
        "multiple_testing_correction": holm_results,
    }

    with open(reports_dir / "experiment_report.json", "w") as f:
        json.dump(exp_report, f, indent=2)

    _plot_power_curve(sample_sizes, figures_dir)
    print(f"  Experiment report saved to {reports_dir / 'experiment_report.json'}")


def _plot_power_curve(sample_sizes: list, figures_dir: Path):
    mdes = [s["relative_mde"] * 100 for s in sample_sizes]
    n_per_arm = [s["sample_size_per_arm"] for s in sample_sizes]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mdes, n_per_arm, "o-", color="#2b5c8f", lw=2, markersize=6)
    ax.set_yscale("log")
    ax.set_xlabel("Minimum Detectable Effect (MDE % of baseline)")
    ax.set_ylabel("Required Sample Size Per Arm (Log Scale)")
    ax.set_title("Statistical Power Curve (80% Power, α = 0.05)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_dir / "power_curve.png", dpi=150)
    plt.close()
