"""Propensity models: purchase and churn prediction with LightGBM and baselines."""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


LABEL_COLS = ["label_purchase", "label_churn", "label_revenue"]
META_COLS = [
    "user_id",
    "user_session",
    "rfm_segment",
    "rfm_score",
    "r_score",
    "f_score",
    "m_score",
    "segment_kmeans",
    "segment_gmm",
    "top_category",
    "top_brand",
]
RFM_FEATURES = [
    "days_since_last_event",
    "total_events",
    "total_spend",
    "purchase_count",
    "active_days",
]

FEATURE_GROUPS = {
    "recency_frequency": [
        "days_since_last_event",
        "days_since_last_purchase",
        "active_days",
        "total_sessions",
        "avg_session_length_minutes",
        "avg_session_gap_hours",
        "total_events",
        "events_per_active_day",
        "events_per_session",
    ],
    "funnel_conversion": [
        "view_count",
        "cart_count",
        "remove_count",
        "purchase_count",
        "view_to_cart_rate",
        "cart_to_purchase_rate",
        "cart_abandonment_rate",
        "has_open_cart",
        "cart_churn_events",
        "cart_churn_rate",
        "purchase_event_share",
        "view_to_purchase_rate",
    ],
    "monetary": [
        "total_spend",
        "avg_order_value",
        "purchase_count_nonzero",
        "spend_per_purchase",
        "log_total_spend",
        "price_mean",
        "price_std",
        "price_median",
        "price_relative_to_category_median",
    ],
    "breadth_temporal": [
        "unique_categories",
        "unique_brands",
        "top_category_share",
        "top_brand_share",
        "events_per_category",
        "events_per_brand",
        "pct_events_weekday",
        "peak_hour",
        "pct_events_peak_hour",
        "pct_events_morning",
        "pct_events_afternoon",
        "pct_events_evening",
        "pct_events_night",
        "last_3d_events",
        "first_7d_events",
        "recency_trend_ratio",
        "log_total_events",
        "log_total_sessions",
    ],
}


def run_propensity(cfg: dict, root: Path):
    """Train and evaluate propensity models on snapshot data."""
    features_dir = root / cfg["paths"]["features_dir"]
    figures_dir = root / cfg["paths"]["figures_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]

    train_path = features_dir / "train_segments.parquet"
    if not train_path.exists():
        train_path = features_dir / "train_features.parquet"
    val_path = features_dir / "validation_segments.parquet"
    if not val_path.exists():
        val_path = features_dir / "validation_features.parquet"
    test_path = features_dir / "test_segments.parquet"
    if not test_path.exists():
        test_path = features_dir / "test_features.parquet"

    print("  Loading feature datasets...")
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)
    print(f"  Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

    drop_cols = set(LABEL_COLS + META_COLS)
    feature_cols = [
        c for c in train.columns if c not in drop_cols and np.issubdtype(train[c].dtype, np.number)
    ]
    print(f"  Feature count: {len(feature_cols)}")

    all_metrics = {}

    for target in ["label_purchase", "label_churn"]:
        target_name = target.replace("label_", "")
        print(f"\n  === Propensity: {target_name.upper()} ===")

        # Filter out NaN labels (churn only defined for active_days >= 2)
        tr_mask = train[target].notna()
        va_mask = val[target].notna()
        te_mask = test[target].notna()

        X_tr = train.loc[tr_mask, feature_cols].fillna(0).values.astype(np.float32)
        y_tr = train.loc[tr_mask, target].values.astype(int)

        X_va = val.loc[va_mask, feature_cols].fillna(0).values.astype(np.float32)
        y_va = val.loc[va_mask, target].values.astype(int)

        X_te = test.loc[te_mask, feature_cols].fillna(0).values.astype(np.float32)
        y_te = test.loc[te_mask, target].values.astype(int)

        print(f"    Train size: {len(y_tr):,} (base rate: {y_tr.mean():.4f})")
        print(f"    Val size:   {len(y_va):,} (base rate: {y_va.mean():.4f})")
        print(f"    Test size:  {len(y_te):,} (base rate: {y_te.mean():.4f})")

        # 1. Baseline: RFM heuristic
        rfm_idx = [feature_cols.index(c) for c in RFM_FEATURES if c in feature_cols]
        if rfm_idx:
            rfm_te_norm = X_te[:, rfm_idx].copy()
            for col_i in range(rfm_te_norm.shape[1]):
                mn, mx = rfm_te_norm[:, col_i].min(), rfm_te_norm[:, col_i].max()
                if mx > mn:
                    rfm_te_norm[:, col_i] = (rfm_te_norm[:, col_i] - mn) / (mx - mn)
            # Days since last event: inverse for purchase, direct for churn
            if target == "label_purchase":
                rfm_te_norm[:, 0] = 1.0 - rfm_te_norm[:, 0]
            rfm_score = rfm_te_norm.mean(axis=1)
        else:
            rfm_score = np.zeros(len(y_te))

        # 2. Baseline: Logistic Regression
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_te)

        lr = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=300,
            random_state=seed,
            class_weight="balanced",
        )
        lr.fit(X_tr_scaled, y_tr)
        lr_proba = lr.predict_proba(X_te_scaled)[:, 1]

        # 3. Model: LightGBM with Optuna tuning on validation
        pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

        # Sample for fast tuning if train set is very large
        tune_sample_size = min(len(X_tr), 100000)
        rng = np.random.RandomState(seed)
        tune_idx = rng.choice(len(X_tr), tune_sample_size, replace=False)
        X_tr_tune = X_tr[tune_idx]
        y_tr_tune = y_tr[tune_idx]

        def objective(
            trial,
            X_tune=X_tr_tune,
            y_tune=y_tr_tune,
            X_val=X_va,
            y_val=y_va,
            weight=pos_weight,
        ):
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "n_estimators": 500,
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
                "scale_pos_weight": weight,
                "random_state": seed,
                "deterministic": True,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tune,
                y_tune,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            va_p = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, va_p)

        n_trials = min(cfg.get("propensity", {}).get("tuning_trials", 20), 20)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update(
            {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "n_estimators": 800,
                "scale_pos_weight": pos_weight,
                "random_state": seed,
                "deterministic": True,
            }
        )

        lgbm = lgb.LGBMClassifier(**best_params)
        lgbm.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )
        lgbm_raw_proba = lgbm.predict_proba(X_te)[:, 1]

        # Calibration on validation set
        val_raw_p = lgbm.predict_proba(X_va)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(val_raw_p, y_va)
        lgbm_cal_proba = np.clip(iso.predict(lgbm_raw_proba), 0.0, 1.0)

        # Evaluate models on test snapshot
        target_metrics = {}
        for name, p in [
            ("rfm_heuristic", rfm_score),
            ("logistic_regression", lr_proba),
            ("lightgbm_raw", lgbm_raw_proba),
            ("lightgbm_calibrated", lgbm_cal_proba),
        ]:
            m = _eval_scores(y_te, p)
            target_metrics[name] = m
            print(
                f"    {name:22s} | ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | Brier: {m['brier']:.4f}"
            )

        # Bootstrap CIs for AUC
        n_boot = cfg.get("propensity", {}).get("bootstrap_ci_n", 1000)
        alpha = cfg.get("propensity", {}).get("bootstrap_ci_alpha", 0.05)
        for name, p in [
            ("lightgbm_calibrated", lgbm_cal_proba),
            ("logistic_regression", lr_proba),
            ("rfm_heuristic", rfm_score),
        ]:
            target_metrics[name]["roc_auc_ci"] = _bootstrap_auc_ci(y_te, p, n_boot, alpha, seed)

        # Bootstrap CIs for AUC differences vs baselines
        target_metrics["diff_lgbm_vs_rfm_auc_ci"] = _bootstrap_diff_ci(
            y_te, lgbm_cal_proba, rfm_score, n_boot, alpha, seed
        )
        target_metrics["diff_lgbm_vs_lr_auc_ci"] = _bootstrap_diff_ci(
            y_te, lgbm_cal_proba, lr_proba, n_boot, alpha, seed
        )

        # Decile lift table
        target_metrics["decile_table"] = _decile_analysis(y_te, lgbm_cal_proba)

        # Top 10% precision
        top10_n = max(1, int(len(y_te) * 0.10))
        top10_idx = np.argsort(lgbm_cal_proba)[-top10_n:]
        target_metrics["precision_top_10pct"] = round(float(y_te[top10_idx].mean()), 4)

        # Feature importance
        importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": lgbm.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        target_metrics["top_features"] = importance.head(20).to_dict(orient="records")

        # Feature group ablation on validation
        ablation = {}
        full_val_auc = roc_auc_score(y_va, lgbm.predict_proba(X_va)[:, 1])
        ablation["full_model"] = round(float(full_val_auc), 4)

        for grp_name, grp_cols in FEATURE_GROUPS.items():
            rem_cols = [c for c in feature_cols if c not in grp_cols]
            rem_idx = [feature_cols.index(c) for c in rem_cols]
            if len(rem_idx) > 5:
                abl_model = lgb.LGBMClassifier(
                    n_estimators=150,
                    learning_rate=0.08,
                    max_depth=5,
                    scale_pos_weight=pos_weight,
                    random_state=seed,
                    deterministic=True,
                    verbosity=-1,
                )
                abl_model.fit(X_tr_tune[:, rem_idx], y_tr_tune)
                p_abl = abl_model.predict_proba(X_va[:, rem_idx])[:, 1]
                abl_auc = roc_auc_score(y_va, p_abl)
                ablation[f"without_{grp_name}"] = round(float(abl_auc), 4)
                ablation[f"drop_{grp_name}_delta"] = round(float(full_val_auc - abl_auc), 4)

        target_metrics["feature_group_ablation"] = ablation
        all_metrics[target_name] = target_metrics

        # Save figures
        _plot_roc_comparison(
            y_te,
            {
                "RFM Heuristic": rfm_score,
                "Logistic Regression": lr_proba,
                "LightGBM (Calibrated)": lgbm_cal_proba,
            },
            target_name,
            figures_dir,
        )

        _plot_calibration_curve(y_te, lgbm_cal_proba, lgbm_raw_proba, target_name, figures_dir)
        _plot_top_features(importance, target_name, figures_dir)

        # Save test probabilities for NBA
        test_sub = test.loc[te_mask].copy()
        test_sub[f"proba_{target_name}"] = lgbm_cal_proba
        test_sub[["user_id", f"proba_{target_name}"]].to_parquet(
            features_dir / f"test_{target_name}_probas.parquet", index=False
        )

    with open(reports_dir / "propensity_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=_json_serialize)

    print(f"\n  Propensity metrics saved to {reports_dir / 'propensity_metrics.json'}")


def _eval_scores(y_true: np.ndarray, proba: np.ndarray) -> dict:
    clip_p = np.clip(proba, 1e-7, 1.0 - 1e-7)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "log_loss": round(float(log_loss(y_true, clip_p)), 4),
        "brier": round(float(brier_score_loss(y_true, clip_p)), 4),
    }


def _bootstrap_auc_ci(y_true, proba, n_boot, alpha, seed):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    sub_n = min(n, 25000)
    n_iter = min(n_boot, 200) if n > 5000 else n_boot
    for _ in range(n_iter):
        idx = rng.choice(n, sub_n, replace=True) if n > sub_n else rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], proba[idx]))
    point_auc = float(roc_auc_score(y_true, proba))
    lower = round(float(np.percentile(aucs, 100 * (alpha / 2))), 4)
    upper = round(float(np.percentile(aucs, 100 * (1 - alpha / 2))), 4)
    lower = min(lower, round(point_auc, 4))
    upper = max(upper, round(point_auc, 4))
    return [lower, upper]


def _bootstrap_diff_ci(y_true, p_model, p_baseline, n_boot, alpha, seed):
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(y_true)
    sub_n = min(n, 25000)
    n_iter = min(n_boot, 200) if n > 5000 else n_boot
    for _ in range(n_iter):
        idx = rng.choice(n, sub_n, replace=True) if n > sub_n else rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_m = roc_auc_score(y_true[idx], p_model[idx])
        auc_b = roc_auc_score(y_true[idx], p_baseline[idx])
        diffs.append(auc_m - auc_b)
    return [
        round(float(np.percentile(diffs, 100 * (alpha / 2))), 4),
        round(float(np.percentile(diffs, 100 * (1 - alpha / 2))), 4),
    ]


def _decile_analysis(y_true: np.ndarray, proba: np.ndarray) -> list[dict]:
    df = pd.DataFrame({"y": y_true, "p": proba})
    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    base_rate = float(y_true.mean())

    agg = (
        df.groupby("decile")
        .agg(
            count=("y", "count"),
            positives=("y", "sum"),
            mean_pred=("p", "mean"),
        )
        .reset_index()
        .sort_values("decile", ascending=False)
    )

    agg["precision"] = (agg["positives"] / agg["count"]).round(4)
    agg["lift"] = (agg["precision"] / base_rate).round(2)
    agg["cum_positives"] = agg["positives"].cumsum()
    agg["cum_recall"] = (agg["cum_positives"] / y_true.sum()).round(4)

    return agg.to_dict(orient="records")


def _plot_roc_comparison(y_true, proba_map, target, figures_dir):
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#888888", "#2b5c8f", "#d95f02"]
    for (name, p), c in zip(proba_map.items(), colors, strict=False):
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=c, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves - {target.title()}")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_dir / f"roc_{target}.png", dpi=150)
    plt.close()


def _plot_calibration_curve(y_true, p_cal, p_raw, target, figures_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    f_cal, m_cal = calibration_curve(y_true, p_cal, n_bins=10, strategy="uniform")
    f_raw, m_raw = calibration_curve(y_true, p_raw, n_bins=10, strategy="uniform")
    ax.plot(m_raw, f_raw, "s--", label="Raw LightGBM", color="#888888")
    ax.plot(m_cal, f_cal, "o-", label="Calibrated LightGBM", color="#d95f02")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration - {target.title()}")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_dir / f"calibration_{target}.png", dpi=150)
    plt.close()


def _plot_top_features(importance, target, figures_dir):
    top = importance.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#2b5c8f", edgecolor="white")
    ax.set_xlabel("Feature Importance (Splits)")
    ax.set_title(f"Top 15 Features - {target.title()} Propensity")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_dir / f"feature_importance_{target}.png", dpi=150)
    plt.close()


def _json_serialize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
