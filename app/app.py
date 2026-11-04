"""Customer Intelligence Platform: Segmentation, Next-Best-Action & Banner Ranking.

Streamlit demo application running from compact precomputed data in app_data/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DATA_DIR = Path(__file__).resolve().parents[1] / "app_data"


@st.cache_data
def load_json_file(filename: str) -> dict | None:
    p = APP_DATA_DIR / filename
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_parquet_file(filename: str) -> pd.DataFrame | None:
    p = APP_DATA_DIR / filename
    if not p.exists():
        return None
    return pd.read_parquet(p)


def check_data_availability():
    if not APP_DATA_DIR.exists() or not (APP_DATA_DIR / "sampled_users.parquet").exists():
        st.error(
            "Demo data not found in `app_data/`. Please run `make app-data` or `python -m custintel app-data` first."
        )
        st.stop()


def main():
    check_data_availability()

    st.sidebar.title("Customer Intelligence")
    st.sidebar.caption("Portfolio Engine: eCommerce Personalization & Decisioning")

    pages = [
        "1. Executive Overview",
        "2. Funnel & Opportunities",
        "3. Customer Segments",
        "4. Propensity Scorecard",
        "5. Next-Best-Action (NBA)",
        "6. Banner Ranking Engine",
        "7. Experiment Planner",
    ]
    page = st.sidebar.radio("Navigate", pages)

    st.sidebar.markdown("---")
    st.sidebar.caption("Stack: DuckDB, LightGBM, Scikit-Learn, Streamlit")

    if page == "1. Executive Overview":
        render_overview()
    elif page == "2. Funnel & Opportunities":
        render_funnel()
    elif page == "3. Customer Segments":
        render_segments()
    elif page == "4. Propensity Scorecard":
        render_propensity()
    elif page == "5. Next-Best-Action (NBA)":
        render_nba()
    elif page == "6. Banner Ranking Engine":
        render_ranking()
    elif page == "7. Experiment Planner":
        render_experiment()


def render_overview():
    st.header("Executive Overview & Architecture")
    st.markdown(
        """
        **Customer Segmentation, Next-Best-Action & Banner Ranking Engine** is an end-to-end
        ML and statistical decisioning engine built on the 42M+ event REES46 eCommerce dataset.
        It unifies customer behavioural segmentation, predictive propensities, and onsite personalization
        with rigorous experimental evaluation.
        """
    )

    profile = load_json_file("data_profile.json")
    if profile:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Clean Events", f"{profile.get('total_clean_events', 0):,}")
        col2.metric("Unique Users", f"{profile.get('unique_users', 0):,}")
        col3.metric("Unique Catalog Items", f"{profile.get('unique_products', 0):,}")
        col4.metric("Deduplicated Rows", f"{profile.get('duplicates_removed', 0):,}")

    st.subheader("System Architecture")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            A[Raw Events: 42M+] -->|DuckDB Cleaning| B[Clean Parquet]
            B -->|Lookback SQL| C[50+ Behavioural Features]
            C -->|Clustering| D[K-Means / RFM Segments]
            C -->|LightGBM| E[Calibrated Propensity Models]
            D & E -->|EV Scoring| F[Next-Best-Action Engine]
            B & D -->|LambdaRank| G[Banner Ranking Engine]
            F & G -->|CUPED & Power| H[Experimentation & Testing]
        ```
        """
    )

    st.subheader("Key Findings & Methodology")
    st.markdown(
        """
        - **Data Integrity**: Processed via DuckDB with strict leakage-free temporal snapshots (10-day lookback, 7-day horizon).
        - **Predictive Performance**: Calibrated LightGBM models significantly outperform classical RFM and Logistic baselines.
        - **Honest Evaluation**: All banner impressions and intervention gains are explicitly framed as offline simulations, with counterfactual inverse propensity scoring (SNIPS) applied to control for observational popularity bias.
        """
    )


def render_funnel():
    st.header("Funnel & Onsite Opportunities")
    funnel = load_json_file("funnel_report.json")
    if not funnel:
        st.info("Funnel report not loaded.")
        return

    ov = funnel.get("overall_funnel", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Viewers", f"{ov.get('viewers', 0):,}")
    col2.metric("Carters (View→Cart)", f"{ov.get('view_to_cart_rate', 0):.2%}")
    col3.metric("Purchasers (Cart→Purchase)", f"{ov.get('cart_to_purchase_rate', 0):.2%}")

    st.markdown("---")
    st.subheader("Cart Abandonment & Purchase Velocity")
    ab = funnel.get("cart_abandonment", {})
    tp = funnel.get("time_to_purchase", {})
    c1, c2 = st.columns(2)
    c1.metric("Cart Abandonment Rate", f"{ab.get('abandonment_rate', 0):.2%}")
    c2.metric("Median Time to Purchase", f"{tp.get('median_minutes', 0)} mins")

    st.subheader("Top Onsite Opportunity Categories")
    opps = funnel.get("opportunities", [])
    if opps:
        df_opps = pd.DataFrame(opps)
        st.dataframe(df_opps, width="stretch")
        st.caption(funnel.get("opportunity_assumptions", ""))


def render_segments():
    st.header("Customer Segmentation")
    metrics = load_json_file("segment_metrics.json")
    profiles = load_json_file("segment_profiles.json")

    k_sel = metrics.get("k_selection", {}) if metrics else {}
    if k_sel:
        st.subheader("K-Means Cluster Optimization (k Selection)")
        k_df = pd.DataFrame.from_dict(k_sel, orient="index")
        st.dataframe(k_df, width="stretch")

    st.subheader("Segment Profiles (Test Snapshot)")
    test_profs = profiles.get("test_segment_kmeans", {}) if profiles else {}
    if test_profs:
        prof_df = pd.DataFrame.from_dict(test_profs, orient="index")
        st.dataframe(prof_df, width="stretch")

    st.subheader("Segment Migration")
    mig = metrics.get("migration", {}) if metrics else {}
    if mig:
        st.json(mig)


def render_propensity():
    st.header("Propensity Scorecard: Purchase & Churn")
    prop = load_json_file("propensity_metrics.json")
    if not prop:
        st.info("Propensity metrics not available.")
        return

    tab1, tab2 = st.tabs(["Purchase Propensity", "Churn Propensity"])

    with tab1:
        st.subheader("Purchase Model Evaluation (Test Snapshot)")
        purch = prop.get("purchase", {})
        res_rows = []
        for m_name in [
            "rfm_heuristic",
            "logistic_regression",
            "lightgbm_raw",
            "lightgbm_calibrated",
        ]:
            if m_name in purch:
                r = {"Model": m_name}
                r.update(purch[m_name])
                res_rows.append(r)
        st.dataframe(pd.DataFrame(res_rows), width="stretch")

        st.subheader("Decile Lift Analysis")
        dec = purch.get("decile_table", [])
        if dec:
            st.dataframe(pd.DataFrame(dec), width="stretch")

    with tab2:
        st.subheader("Churn Model Evaluation (Test Snapshot)")
        churn = prop.get("churn", {})
        res_rows = []
        for m_name in [
            "rfm_heuristic",
            "logistic_regression",
            "lightgbm_raw",
            "lightgbm_calibrated",
        ]:
            if m_name in churn:
                r = {"Model": m_name}
                r.update(churn[m_name])
                res_rows.append(r)
        st.dataframe(pd.DataFrame(res_rows), width="stretch")


def render_nba():
    st.header("Next-Best-Action (NBA) Engine")
    users = load_parquet_file("sampled_users.parquet")
    nba = load_json_file("nba_report.json")

    if users is None:
        st.info("Sampled user data not found.")
        return

    st.subheader("Interactive User Recommendation")
    uid_list = users["user_id"].unique()[:100]
    chosen_uid = st.selectbox("Select User ID for Evaluation:", uid_list)

    user_row = users[users["user_id"] == chosen_uid].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assigned NBA Action", str(user_row.get("nba_action", "N/A")))
    c2.metric("Reason Code", str(user_row.get("nba_reason", "N/A")))
    c3.metric("Purchase Propensity", f"{user_row.get('proba_purchase', 0):.2%}")
    c4.metric("Churn Risk", f"{user_row.get('proba_churn', 0):.2%}")

    st.subheader("Policy Simulation Comparison (Offline Expected Value)")
    if nba:
        sims = nba.get("simulations", {})
        sim_df = pd.DataFrame.from_dict(sims, orient="index")
        st.dataframe(sim_df, width="stretch")
        st.info(nba.get("disclaimer", ""))


def render_ranking():
    st.header("Banner Ranking Engine: Category Personalization")
    rank_rep = load_json_file("ranking_metrics.json")
    rankings = load_parquet_file("sample_rankings.parquet")

    if rank_rep:
        st.subheader("Offline Ranking Quality (NDCG@10 & Recall@10)")
        models = ["global_popularity", "segment_popularity", "user_most_viewed", "lightgbm_ltr"]
        r_rows = []
        for m in models:
            if m in rank_rep:
                r = {"Model": m}
                r.update(rank_rep[m])
                r_rows.append(r)
        st.dataframe(pd.DataFrame(r_rows), width="stretch")

        st.subheader("Counterfactual Evaluation (SNIPS CTR Proxy)")
        st.json(rank_rep.get("ips_ctr_proxy", {}))

    if rankings is not None:
        st.subheader("Personalized Category Ranking Sample")
        u_list = rankings["user_id"].unique()[:50]
        sel_u = st.selectbox("Select User for Category Ranking:", u_list)
        u_rank = rankings[rankings["user_id"] == sel_u].sort_values("rank")
        st.dataframe(
            u_rank[["rank", "category_code", "score", "actual_relevance"]],
            width="stretch",
        )


def render_experiment():
    st.header("Experiment Planner & Power Calculator")
    exp = load_json_file("experiment_report.json")

    base_rate = 0.05
    if exp:
        base_rate = exp.get("baseline_metrics", {}).get("purchase_conversion_rate", 0.05)

    st.subheader("Interactive Sample Size Calculator")
    col1, col2 = st.columns(2)
    with col1:
        user_base_rate = st.slider(
            "Baseline Purchase Conversion Rate", 0.01, 0.20, float(base_rate), step=0.005
        )
        desired_power = st.slider("Statistical Power (1 - β)", 0.70, 0.95, 0.80, step=0.05)
    with col2:
        mde_pct = st.slider(
            "Target Minimum Detectable Effect (Relative %)", 1.0, 15.0, 5.0, step=0.5
        )
        sig_alpha = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)

    p1 = user_base_rate
    p2 = user_base_rate * (1 + mde_pct / 100.0)
    h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

    from statsmodels.stats.power import NormalIndPower

    pwr = NormalIndPower()
    n_per_arm = int(
        np.ceil(
            pwr.solve_power(
                effect_size=h, alpha=sig_alpha, power=desired_power, alternative="two-sided"
            )
        )
    )

    st.success(
        f"**Required Sample Size:** **{n_per_arm:,}** users per arm (**{n_per_arm * 2:,}** total users for 2 arms)"
    )

    if exp:
        st.subheader("CUPED Variance Reduction (Real Data)")
        cuped_info = exp.get("cuped", {})
        st.json(cuped_info)


if __name__ == "__main__":
    main()
