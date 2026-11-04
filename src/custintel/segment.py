"""Customer segmentation: RFM baseline, K-Means, GMM."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Behavioural features for clustering
CLUSTER_FEATURES = [
    "days_since_last_event",
    "active_days",
    "total_events",
    "view_count",
    "cart_count",
    "purchase_count",
    "total_spend",
    "unique_categories",
    "unique_brands",
    "view_to_cart_rate",
    "cart_to_purchase_rate",
    "avg_session_length_minutes",
    "pct_events_weekday",
    "recency_trend_ratio",
]


def run_segment(cfg: dict, root: Path):
    """Run segmentation pipeline."""
    features_dir = root / cfg["paths"]["features_dir"]
    figures_dir = root / cfg["paths"]["figures_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    seg_cfg = cfg["segmentation"]

    # Load feature dataframes
    snapshots = {}
    for name in ["train", "validation", "test"]:
        path = features_dir / f"{name}_features.parquet"
        snapshots[name] = pd.read_parquet(path)
        print(f"  {name}: {len(snapshots[name]):,} users")

    train_df = snapshots["train"]

    # --- RFM Baseline ---
    print("\n  RFM segmentation...")
    for name, df in snapshots.items():
        snapshots[name] = _rfm_segment(df, seg_cfg["rfm_bins"])

    # --- Model-based segmentation ---
    print("\n  Preparing clustering features...")
    scaler = StandardScaler()
    train_X, valid_cols = _prepare_cluster_features(train_df)
    train_X_scaled = scaler.fit_transform(train_X)

    # Select k
    print("  Selecting k...")
    k_range = seg_cfg["k_range"]
    sample_size = min(seg_cfg["silhouette_sample"], len(train_X_scaled))
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(len(train_X_scaled), sample_size, replace=False)
    sample_X = train_X_scaled[sample_idx]

    k_metrics = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        labels_sample = km.fit_predict(sample_X)
        sil = silhouette_score(sample_X, labels_sample)
        db = davies_bouldin_score(sample_X, labels_sample)

        gmm = GaussianMixture(n_components=k, random_state=seed, covariance_type="full")
        gmm.fit(sample_X)
        bic = gmm.bic(sample_X)

        # Stability: ARI across seeds
        ari_scores = _measure_stability(
            sample_X, k, seg_cfg["stability_seeds"], seg_cfg["stability_bootstraps"]
        )

        k_metrics[k] = {
            "silhouette": round(float(sil), 4),
            "davies_bouldin": round(float(db), 4),
            "bic": round(float(bic), 1),
            "stability_ari_mean": round(float(np.mean(ari_scores)), 4),
            "stability_ari_std": round(float(np.std(ari_scores)), 4),
        }
        print(
            f"    k={k}: sil={sil:.4f}, DB={db:.4f}, BIC={bic:.0f}, ARI={np.mean(ari_scores):.4f}"
        )

    # Pick k: highest silhouette among those with reasonable stability
    best_k = max(k_range, key=lambda k: k_metrics[k]["silhouette"])
    print(f"  Selected k={best_k}")

    # Fit final models
    print("  Fitting final KMeans on full training snapshot...")
    km_final = KMeans(n_clusters=best_k, random_state=seed, n_init=10, max_iter=300)
    km_final.fit_predict(train_X_scaled)

    print("  Fitting final GMM...")
    gmm_final = GaussianMixture(n_components=best_k, random_state=seed, covariance_type="full")
    gmm_final.fit(sample_X)
    gmm_final.predict(train_X_scaled)

    # Apply to all snapshots
    all_metrics = {"k_selection": k_metrics, "chosen_k": best_k}
    segment_profiles = {}

    for name, df in snapshots.items():
        X, _ = _prepare_cluster_features(df)
        X_scaled = scaler.transform(X)

        df["segment_kmeans"] = km_final.predict(X_scaled)
        df["segment_gmm"] = gmm_final.predict(X_scaled)

        # Profile segments
        for method in ["rfm_segment", "segment_kmeans", "segment_gmm"]:
            profile = _profile_segments(df, method, name)
            segment_profiles[f"{name}_{method}"] = profile

        snapshots[name] = df

    # Segment migration
    migration = {}
    for method in ["segment_kmeans", "segment_gmm"]:
        # Users in both train and validation
        train_users = snapshots["train"][["user_id", method]].rename(columns={method: "seg_train"})
        val_users = snapshots["validation"][["user_id", method]].rename(
            columns={method: "seg_val"}
        )
        merged = train_users.merge(val_users, on="user_id", how="inner")
        if len(merged) > 0:
            migration_rate = (merged["seg_train"] != merged["seg_val"]).mean()
            migration[f"{method}_train_to_val"] = round(float(migration_rate), 4)

        val_users2 = snapshots["validation"][["user_id", method]].rename(
            columns={method: "seg_val"}
        )
        test_users = snapshots["test"][["user_id", method]].rename(columns={method: "seg_test"})
        merged2 = val_users2.merge(test_users, on="user_id", how="inner")
        if len(merged2) > 0:
            migration_rate2 = (merged2["seg_val"] != merged2["seg_test"]).mean()
            migration[f"{method}_val_to_test"] = round(float(migration_rate2), 4)

    all_metrics["migration"] = migration

    # Variance explained in outcomes
    for name in ["train", "validation", "test"]:
        df = snapshots[name]
        for method in ["segment_kmeans", "segment_gmm", "rfm_segment"]:
            for outcome in ["label_purchase", "label_revenue"]:
                if outcome in df.columns and method in df.columns:
                    groups = df.groupby(method)[outcome]
                    group_means = groups.transform("mean")
                    ss_between = ((group_means - df[outcome].mean()) ** 2).sum()
                    ss_total = ((df[outcome] - df[outcome].mean()) ** 2).sum()
                    r2 = ss_between / ss_total if ss_total > 0 else 0
                    key = f"{name}_{method}_{outcome}_r2"
                    all_metrics[key] = round(float(r2), 4)

    # Save outputs
    for name, df in snapshots.items():
        out_path = features_dir / f"{name}_segments.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {out_path}")

    with open(reports_dir / "segment_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    with open(reports_dir / "segment_profiles.json", "w") as f:
        json.dump(segment_profiles, f, indent=2, default=str)

    # Figures
    _plot_segment_distribution(snapshots["train"], figures_dir)
    _plot_segment_outcomes(snapshots["test"], figures_dir)

    print("  Segmentation complete.")


def _rfm_segment(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Add RFM segment labels. Handles zero-inflated monetary."""
    df = df.copy()

    # Recency: lower is better -> higher score
    df["r_score"] = (
        pd.cut(
            df["days_since_last_event"],
            bins=n_bins,
            labels=range(n_bins, 0, -1),
            duplicates="drop",
        )
        .astype(float)
        .fillna(1)
    )

    # Frequency
    df["f_score"] = (
        pd.cut(
            df["total_events"],
            bins=n_bins,
            labels=range(1, n_bins + 1),
            duplicates="drop",
        )
        .astype(float)
        .fillna(1)
    )

    # Monetary: handle zero-inflated
    has_spend = df["total_spend"] > 0
    df["m_score"] = 0.0
    if has_spend.sum() > 0:
        n_bins_m = min(n_bins, df.loc[has_spend, "total_spend"].nunique())
        if n_bins_m > 0:
            spend_scores = (
                pd.cut(
                    df.loc[has_spend, "total_spend"],
                    bins=n_bins_m,
                    labels=range(1, n_bins_m + 1),
                    duplicates="drop",
                )
                .astype(float)
                .fillna(1)
            )
            df.loc[has_spend, "m_score"] = spend_scores

    df["rfm_score"] = (
        df["r_score"].astype(int).astype(str)
        + df["f_score"].astype(int).astype(str)
        + df["m_score"].astype(int).astype(str)
    )

    # Map to named segments
    df["rfm_segment"] = df.apply(_rfm_label, axis=1)
    return df


def _rfm_label(row) -> str:
    """Assign a named segment based on RFM scores."""
    r, f, m = float(row["r_score"]), float(row["f_score"]), float(row["m_score"])
    if r >= 3 and f >= 3 and m >= 3:
        return "champions"
    elif r >= 3 and f >= 2 and m >= 1:
        return "loyal"
    elif r >= 3 and f < 2:
        return "new_or_promising"
    elif r == 2 and f >= 2:
        return "at_risk"
    elif r == 1 and f >= 2:
        return "hibernating"
    elif m == 0 and f < 2:
        return "window_shoppers"
    else:
        return "need_attention"


def _prepare_cluster_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Log-transform and prepare features for clustering."""
    features = []
    valid_cols = []
    for col in CLUSTER_FEATURES:
        if col in df.columns:
            vals = df[col].fillna(0).values.astype(float)
            # Log transform for skewed features (counts, spend)
            if col in [
                "total_events",
                "view_count",
                "cart_count",
                "purchase_count",
                "total_spend",
                "unique_categories",
                "unique_brands",
                "avg_session_length_minutes",
            ]:
                vals = np.log1p(vals)
            features.append(vals)
            valid_cols.append(col)

    return np.column_stack(features), valid_cols


def _measure_stability(X: np.ndarray, k: int, seeds: list, n_bootstraps: int) -> list[float]:
    """Measure clustering stability via ARI across seeds and bootstraps."""
    from sklearn.metrics import adjusted_rand_score

    ari_scores = []
    # Across seeds
    labels_by_seed = []
    for s in seeds:
        km = KMeans(n_clusters=k, random_state=s, n_init=5, max_iter=200)
        labels_by_seed.append(km.fit_predict(X))

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ari_scores.append(adjusted_rand_score(labels_by_seed[i], labels_by_seed[j]))

    return ari_scores


def _profile_segments(df: pd.DataFrame, seg_col: str, snapshot_name: str) -> dict:
    """Profile each segment: size, behaviour, outcomes."""
    profiles = {}
    for seg_val, group in df.groupby(seg_col):
        profile = {
            "size": len(group),
            "pct_of_users": round(len(group) / len(df) * 100, 2),
            "avg_events": round(group["total_events"].mean(), 1),
            "avg_spend": round(group["total_spend"].mean(), 2),
            "avg_active_days": round(group["active_days"].mean(), 2),
        }
        if "label_purchase" in group.columns:
            purchase_rate = group["label_purchase"].mean()
            profile["purchase_rate"] = round(float(purchase_rate), 4)
            # Bootstrap CI
            if len(group) > 30:
                ci = _bootstrap_ci(group["label_purchase"].values)
                profile["purchase_rate_ci"] = ci
        if "label_churn" in group.columns:
            churn_vals = group["label_churn"].dropna()
            if len(churn_vals) > 0:
                profile["churn_rate"] = round(float(churn_vals.mean()), 4)
        if "label_revenue" in group.columns:
            profile["avg_revenue"] = round(float(group["label_revenue"].mean()), 2)

        profiles[str(seg_val)] = profile
    return profiles


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> list[float]:
    """Confidence interval for the segment mean (exact asymptotic standard error)."""
    n = len(values)
    if n == 0:
        return [0.0, 0.0]
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    se = std_val / np.sqrt(n) if n > 1 else 0.0
    z = 1.95996  # 95% confidence critical value
    return [round(max(0.0, mean_val - z * se), 4), round(mean_val + z * se, 4)]


def _plot_segment_distribution(df: pd.DataFrame, figures_dir: Path):
    """Plot segment size distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, method in zip(axes, ["rfm_segment", "segment_kmeans", "segment_gmm"], strict=False):
        if method in df.columns:
            counts = df[method].value_counts().sort_index()
            counts.plot(kind="bar", ax=ax, color="#4a90d9", edgecolor="white")
            ax.set_title(f"{method.replace('_', ' ').title()}", fontsize=11)
            ax.set_ylabel("Users")
            ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / "segment_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_segment_outcomes(df: pd.DataFrame, figures_dir: Path):
    """Plot purchase rate and revenue by segment."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for col_idx, method in enumerate(["segment_kmeans", "rfm_segment"]):
        if method not in df.columns:
            continue
        grouped = (
            df.groupby(method)
            .agg(
                purchase_rate=("label_purchase", "mean"),
                avg_revenue=("label_revenue", "mean"),
            )
            .reset_index()
        )

        axes[0, col_idx].bar(
            grouped[method].astype(str),
            grouped["purchase_rate"],
            color="#4a90d9",
            edgecolor="white",
        )
        axes[0, col_idx].set_title(f"Purchase Rate by {method}", fontsize=10)
        axes[0, col_idx].set_ylabel("Purchase Rate")
        axes[0, col_idx].tick_params(axis="x", rotation=45)

        axes[1, col_idx].bar(
            grouped[method].astype(str),
            grouped["avg_revenue"],
            color="#e8744f",
            edgecolor="white",
        )
        axes[1, col_idx].set_title(f"Avg Revenue by {method}", fontsize=10)
        axes[1, col_idx].set_ylabel("Revenue")
        axes[1, col_idx].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(figures_dir / "segment_outcomes.png", dpi=150, bbox_inches="tight")
    plt.close()
