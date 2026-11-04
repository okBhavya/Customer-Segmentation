"""Banner ranking engine: category-level slot ranking via LambdaRank and baselines."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from custintel.db import get_connection


def run_ranking(cfg: dict, root: Path):
    """Train and evaluate category banner ranker."""
    parquet_path = root / cfg["paths"]["parquet"]
    features_dir = root / cfg["paths"]["features_dir"]
    figures_dir = root / cfg["paths"]["figures_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    rank_cfg = cfg.get("ranking", {})
    snap_cfg = cfg["snapshots"]

    con = get_connection(cfg, root)
    con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{parquet_path.as_posix()}')")

    # Banner units: top categories with >= min_unit_events (target ~100-300 units)
    min_events = rank_cfg.get("min_unit_events", 50)
    print("  Selecting category banner units...")
    units = con.execute(f"""
        SELECT
            category_id,
            MODE(category_code) AS category_code,
            COUNT(*) AS event_count,
            COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchase_count
        FROM events
        WHERE category_code != 'unknown'
        GROUP BY category_id
        HAVING COUNT(*) >= {min_events}
        ORDER BY event_count DESC
        LIMIT 300
    """).fetchdf()

    if len(units) < 10:
        # Fallback if too strict
        units = con.execute("""
            SELECT
                category_id,
                MODE(category_code) AS category_code,
                COUNT(*) AS event_count,
                COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchase_count
            FROM events
            GROUP BY category_id
            ORDER BY event_count DESC
            LIMIT 200
        """).fetchdf()

    print(f"  Selected {len(units)} banner units (categories).")
    unit_ids = units["category_id"].tolist()
    unit_code_map = dict(zip(units["category_id"], units["category_code"], strict=False))

    # Read segment maps
    tr_seg_path = features_dir / "train_segments.parquet"
    if not tr_seg_path.exists():
        tr_seg_path = features_dir / "train_features.parquet"
    te_seg_path = features_dir / "test_segments.parquet"
    if not te_seg_path.exists():
        te_seg_path = features_dir / "test_features.parquet"

    train_seg = pd.read_parquet(tr_seg_path)
    test_seg = pd.read_parquet(te_seg_path)

    seg_col = "segment_kmeans" if "segment_kmeans" in train_seg.columns else "rfm_segment"
    if seg_col not in train_seg.columns:
        train_seg[seg_col] = 0
        test_seg[seg_col] = 0

    train_seg_map = dict(zip(train_seg["user_id"], train_seg[seg_col], strict=False))
    test_seg_map = dict(zip(test_seg["user_id"], test_seg[seg_col], strict=False))

    print("  Building training dataset for LambdaRank...")
    train_data = _build_ranking_split(
        con,
        snap_cfg["train"],
        unit_ids,
        rank_cfg,
        seed,
        train_seg_map,
        sample_size=rank_cfg.get("user_sample_size", 30000),
    )

    print("  Building validation dataset...")
    val_data = _build_ranking_split(
        con, snap_cfg["validation"], unit_ids, rank_cfg, seed + 1, train_seg_map, sample_size=10000
    )

    print("  Building test evaluation dataset...")
    test_data = _build_ranking_split(
        con,
        snap_cfg["test"],
        unit_ids,
        rank_cfg,
        seed + 2,
        test_seg_map,
        sample_size=rank_cfg.get("eval_user_sample", 8000),
    )

    if not train_data or len(train_data["X"]) == 0:
        print("  Error: Empty ranking training data.")
        con.close()
        return

    # Train LightGBM LambdaRank
    feat_names = train_data["feature_names"]
    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "n_estimators": rank_cfg.get("lgbm_ranker", {}).get("n_estimators", 300),
        "learning_rate": rank_cfg.get("lgbm_ranker", {}).get("learning_rate", 0.05),
        "max_depth": rank_cfg.get("lgbm_ranker", {}).get("max_depth", 6),
        "num_leaves": rank_cfg.get("lgbm_ranker", {}).get("num_leaves", 31),
        "min_child_samples": rank_cfg.get("lgbm_ranker", {}).get("min_child_samples", 20),
        "subsample": rank_cfg.get("lgbm_ranker", {}).get("subsample", 0.8),
        "colsample_bytree": rank_cfg.get("lgbm_ranker", {}).get("colsample_bytree", 0.8),
        "random_state": seed,
        "deterministic": True,
        "verbosity": -1,
    }

    train_ds = lgb.Dataset(
        train_data["X"], label=train_data["y"], group=train_data["groups"], feature_name=feat_names
    )
    val_ds = lgb.Dataset(
        val_data["X"],
        label=val_data["y"],
        group=val_data["groups"],
        feature_name=feat_names,
        reference=train_ds,
    )

    ranker = lgb.train(
        lgb_params,
        train_ds,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    # Evaluate on held-out test users
    print("  Evaluating test ranking performance...")
    test_scores = ranker.predict(test_data["X"])

    metrics = _evaluate_all_models(test_data, test_scores, unit_ids, seed)

    # Offline CTR Proxy with Inverse Propensity Scoring (IPS)
    print("  Calculating Inverse Propensity Scoring (IPS) CTR proxy...")
    ips_results = _ips_evaluation(test_data, test_scores, seed)
    metrics["ips_ctr_proxy"] = ips_results

    # Summary and metadata
    metrics["n_units"] = len(unit_ids)
    metrics["model_framing"] = (
        "No banner impression logs exist in this eCommerce dataset. Category banners "
        "are modelled as category promotional units. The ranking task predicts next-7-day "
        "category engagement (purchase=3, cart=2, view=1). Evaluation is conducted "
        "on a held-out test user sample using NDCG@10, Recall@10, and self-normalized IPS."
    )

    with open(reports_dir / "ranking_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_json_serialize)

    # Save ranking sample for demo app
    _save_app_ranking_sample(test_data, test_scores, unit_code_map, root)

    _plot_ranking_results(metrics, figures_dir)
    con.close()
    print(f"  Ranking metrics saved to {reports_dir / 'ranking_metrics.json'}")


def _build_ranking_split(con, snap_dates, unit_ids, rank_cfg, seed, seg_map, sample_size):
    """Build pairwise/listwise dataset for ranking."""
    f_start = snap_dates["feature_start"]
    f_end = snap_dates["feature_end"]
    l_start = snap_dates["label_start"]
    l_end = snap_dates["label_end"]
    neg_ratio = rank_cfg.get("negative_sample_ratio", 5)

    u_filter = ",".join(str(u) for u in unit_ids)

    # Lookback user-category stats
    lb = con.execute(f"""
        SELECT
            user_id,
            category_id,
            SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS views,
            SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS carts,
            SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases,
            COUNT(*) AS total_interactions,
            EXTRACT(EPOCH FROM (TIMESTAMP '{f_end}' - MAX(event_time))) / 86400.0 AS recency_days
        FROM events
        WHERE event_time >= '{f_start}' AND event_time < '{f_end}'
          AND category_id IN ({u_filter})
        GROUP BY user_id, category_id
    """).fetchdf()

    if len(lb) == 0:
        return None

    # Global category popularity in lookback
    pop = lb.groupby("category_id")["total_interactions"].sum().to_dict()
    tot_pop = max(sum(pop.values()), 1)
    unit_popularity = {k: v / tot_pop for k, v in pop.items()}

    # Segment category popularity
    lb["segment"] = lb["user_id"].map(seg_map).fillna(0)
    seg_pop = lb.groupby(["segment", "category_id"])["total_interactions"].sum()
    seg_tot = lb.groupby("segment")["total_interactions"].sum().to_dict()
    segment_popularity = {}
    for (s, cid), val in seg_pop.items():
        segment_popularity[(s, cid)] = val / max(seg_tot.get(s, 1), 1)

    # Horizon labels: relevance grade: purchase=3, cart=2, view=1
    hz = con.execute(f"""
        SELECT
            user_id,
            category_id,
            MAX(CASE
                WHEN event_type = 'purchase' THEN 3
                WHEN event_type = 'cart' THEN 2
                ELSE 1
            END) AS relevance
        FROM events
        WHERE event_time >= '{l_start}' AND event_time < '{l_end}'
          AND category_id IN ({u_filter})
        GROUP BY user_id, category_id
    """).fetchdf()

    hz_labels = dict(
        zip(zip(hz["user_id"], hz["category_id"], strict=False), hz["relevance"], strict=False)
    )

    # Lookback user totals
    user_totals = lb.groupby("user_id")["total_interactions"].sum().to_dict()
    user_distinct_units = lb.groupby("user_id")["category_id"].nunique().to_dict()

    # User lookback unit set for novel checks
    user_lb_units = lb.groupby("user_id")["category_id"].apply(set).to_dict()

    # Sample users
    all_users = list(user_totals.keys())
    rng = np.random.RandomState(seed)
    if len(all_users) > sample_size:
        selected_users = rng.choice(all_users, size=sample_size, replace=False)
    else:
        selected_users = all_users

    # Quick lookups for interactions
    lb_dict = dict(
        zip(
            zip(lb["user_id"], lb["category_id"], strict=False),
            lb.to_dict(orient="records"),
            strict=False,
        )
    )

    X, y, groups = [], [], []
    user_row_indices = []
    unit_ids_per_row = []

    feat_names = [
        "user_unit_views",
        "user_unit_carts",
        "user_unit_purchases",
        "user_unit_recency_days",
        "user_unit_interaction_share",
        "unit_popularity",
        "segment_unit_popularity",
        "user_total_interactions",
        "user_distinct_units",
    ]

    for uid in selected_users:
        user_seg = seg_map.get(uid, 0)
        u_tot = user_totals.get(uid, 1)
        u_units = user_distinct_units.get(uid, 1)

        # Interacted in horizon (positives)
        pos_units = [cid for cid in unit_ids if (uid, cid) in hz_labels]

        # Negative sampling
        neg_candidates = [cid for cid in unit_ids if (uid, cid) not in hz_labels]
        n_neg = max(
            5, min(len(neg_candidates), len(pos_units) * neg_ratio if pos_units else neg_ratio)
        )
        if len(neg_candidates) > n_neg:
            sampled_neg = rng.choice(neg_candidates, size=n_neg, replace=False)
        else:
            sampled_neg = neg_candidates

        candidate_units = list(pos_units) + list(sampled_neg)
        if len(candidate_units) < 2:
            continue

        g_len = 0
        for cid in candidate_units:
            rec = lb_dict.get((uid, cid), {})
            views = rec.get("views", 0)
            carts = rec.get("carts", 0)
            purchases = rec.get("purchases", 0)
            recency = rec.get("recency_days", 99.0)
            share = (views + carts + purchases) / u_tot
            pop_score = unit_popularity.get(cid, 0.0)
            seg_score = segment_popularity.get((user_seg, cid), 0.0)

            feat = [views, carts, purchases, recency, share, pop_score, seg_score, u_tot, u_units]
            rel = hz_labels.get((uid, cid), 0)

            X.append(feat)
            y.append(rel)
            unit_ids_per_row.append(cid)
            g_len += 1

        groups.append(g_len)
        user_row_indices.append(uid)

    return {
        "X": np.array(X, dtype=np.float32),
        "y": np.array(y, dtype=np.float32),
        "groups": groups,
        "users": user_row_indices,
        "unit_ids_per_row": unit_ids_per_row,
        "unit_popularity": unit_popularity,
        "segment_popularity": segment_popularity,
        "user_lb_units": user_lb_units,
        "seg_map": seg_map,
        "feature_names": feat_names,
    }


def _evaluate_all_models(data, lgb_scores, all_unit_ids, seed):
    """Compute NDCG@10, Recall@10, HitRate@10 for LTR and baselines."""
    y = data["y"]
    groups = data["groups"]
    users = data["users"]
    unit_ids = data["unit_ids_per_row"]
    unit_pop = data["unit_popularity"]
    seg_pop = data["segment_popularity"]
    user_lb = data["user_lb_units"]
    seg_map = data["seg_map"]

    models = ["lightgbm_ltr", "global_popularity", "segment_popularity", "user_most_viewed"]
    metrics_per_model = {m: {"ndcg": [], "recall": [], "hit": []} for m in models}

    # Slices
    novel_ndcg = {m: [] for m in models}
    cold_ndcg = {m: [] for m in models}
    warm_ndcg = {m: [] for m in models}

    offset = 0
    rec_catalog = {m: set() for m in models}

    for i, g_size in enumerate(groups):
        uid = users[i]
        u_seg = seg_map.get(uid, 0)
        lb_set = user_lb.get(uid, set())
        is_cold = len(lb_set) <= 2

        g_y = y[offset : offset + g_size]
        g_cids = unit_ids[offset : offset + g_size]

        # 1. LightGBM score
        s_lgb = lgb_scores[offset : offset + g_size]

        # 2. Global popularity
        s_pop = np.array([unit_pop.get(cid, 0.0) for cid in g_cids])

        # 3. Segment popularity
        s_seg = np.array([seg_pop.get((u_seg, cid), 0.0) for cid in g_cids])

        # 4. User most viewed in lookback
        g_X = data["X"][offset : offset + g_size]
        s_views = g_X[:, 0]  # user_unit_views

        score_dict = {
            "lightgbm_ltr": s_lgb,
            "global_popularity": s_pop,
            "segment_popularity": s_seg,
            "user_most_viewed": s_views,
        }

        for m_name, s_vals in score_dict.items():
            k = min(10, len(g_y))
            nd = _calc_ndcg_at_k(g_y, s_vals, k)
            rc = _calc_recall_at_k(g_y, s_vals, k)
            ht = _calc_hit_at_k(g_y, s_vals, k)

            metrics_per_model[m_name]["ndcg"].append(nd)
            metrics_per_model[m_name]["recall"].append(rc)
            metrics_per_model[m_name]["hit"].append(ht)

            # Record recommended catalog items
            top_k_idx = np.argsort(s_vals)[::-1][:k]
            for idx in top_k_idx:
                rec_catalog[m_name].add(g_cids[idx])

            # Slice evaluations
            if is_cold:
                cold_ndcg[m_name].append(nd)
            else:
                warm_ndcg[m_name].append(nd)

            # Novel units slice (items not in lookback)
            novel_mask = np.array([cid not in lb_set for cid in g_cids])
            if novel_mask.any() and g_y[novel_mask].sum() > 0:
                k_nov = min(k, int(novel_mask.sum()))
                nd_nov = _calc_ndcg_at_k(g_y[novel_mask], s_vals[novel_mask], k_nov)
                novel_ndcg[m_name].append(nd_nov)

        offset += g_size

    # Aggregate results
    results = {}
    tot_units = max(len(all_unit_ids), 1)

    for m in models:
        results[m] = {
            "ndcg_at_10": round(float(np.mean(metrics_per_model[m]["ndcg"])), 4),
            "recall_at_10": round(float(np.mean(metrics_per_model[m]["recall"])), 4),
            "hit_rate_at_10": round(float(np.mean(metrics_per_model[m]["hit"])), 4),
            "catalog_coverage": round(float(len(rec_catalog[m]) / tot_units), 4),
            "cold_start_ndcg": round(float(np.mean(cold_ndcg[m])), 4) if cold_ndcg[m] else None,
            "warm_users_ndcg": round(float(np.mean(warm_ndcg[m])), 4) if warm_ndcg[m] else None,
            "novel_units_ndcg": round(float(np.mean(novel_ndcg[m])), 4) if novel_ndcg[m] else None,
        }

    # Relative improvement of LTR vs Popularity with Bootstrap CIs
    rng = np.random.RandomState(seed)
    n_users = len(metrics_per_model["lightgbm_ltr"]["ndcg"])
    lgb_arr = np.array(metrics_per_model["lightgbm_ltr"]["ndcg"])
    pop_arr = np.array(metrics_per_model["global_popularity"]["ndcg"])

    boot_rel_diffs = []
    for _ in range(1000):
        b_idx = rng.choice(n_users, size=n_users, replace=True)
        pop_m = pop_arr[b_idx].mean()
        if pop_m > 0:
            boot_rel_diffs.append((lgb_arr[b_idx].mean() - pop_m) / pop_m)

    results["lgbm_vs_popularity_uplift_ci"] = [
        round(float(np.percentile(boot_rel_diffs, 2.5)), 4),
        round(float(np.percentile(boot_rel_diffs, 97.5)), 4),
    ]

    return results


def _ips_evaluation(data, lgb_scores, seed):
    """Offline Counterfactual Evaluation using Self-Normalized Inverse Propensity Scoring (SNIPS)."""
    y = data["y"]
    cids = data["unit_ids_per_row"]
    unit_pop = data["unit_popularity"]

    # In logged bandit setup without explicit logs, exposure propensity is proportional to popularity
    # P(exposure | unit) ~ unit_popularity, clipped to prevent weight explosion
    pop_arr = np.array([unit_pop.get(c, 1e-5) for c in cids])
    propensities = np.clip(pop_arr / (pop_arr.max() or 1.0), 0.05, 1.0)

    # Click/Engagement event (view, cart, or purchase in horizon)
    clicks = (y > 0).astype(float)

    # Inverse propensity weights
    weights = clicks / propensities
    # Self-normalized estimator
    snips_ctr = float(weights.sum() / (1.0 / propensities).sum())
    naive_ctr = float(clicks.mean())

    # Bootstrap 95% CIs
    rng = np.random.RandomState(seed)
    n = len(weights)
    boot_snips = []
    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        denom = (1.0 / propensities[idx]).sum()
        if denom > 0:
            boot_snips.append(weights[idx].sum() / denom)

    ci = [
        round(float(np.percentile(boot_snips, 2.5)), 5),
        round(float(np.percentile(boot_snips, 97.5)), 5),
    ]

    return {
        "naive_ctr": round(naive_ctr, 5),
        "snips_ctr_estimate": round(snips_ctr, 5),
        "snips_ctr_ci": ci,
        "weight_clipping_min": 0.05,
        "note": (
            "Propensities estimated from observational category traffic frequency. "
            "Self-Normalized Inverse Propensity Scoring (SNIPS) adjusts for exposure/popularity bias."
        ),
    }


def _calc_ndcg_at_k(y_true, scores, k):
    if k <= 0 or y_true.sum() == 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    dcg = sum((2 ** y_true[i] - 1) / np.log2(pos + 2) for pos, i in enumerate(order))
    ideal = np.argsort(y_true)[::-1][:k]
    idcg = sum((2 ** y_true[i] - 1) / np.log2(pos + 2) for pos, i in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _calc_recall_at_k(y_true, scores, k):
    rel_count = (y_true > 0).sum()
    if rel_count == 0 or k <= 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    return float((y_true[order] > 0).sum() / rel_count)


def _calc_hit_at_k(y_true, scores, k):
    if k <= 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    return 1.0 if (y_true[order] > 0).any() else 0.0


def _save_app_ranking_sample(data, lgb_scores, unit_code_map, root):
    """Save ranking demonstration data for top users."""
    out_dir = root / "app_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    users = data["users"][:500]  # Sample 500 users
    sample_records = []
    offset = 0

    for i in range(len(users)):
        g_size = data["groups"][i]
        uid = users[i]
        g_cids = data["unit_ids_per_row"][offset : offset + g_size]
        g_scores = lgb_scores[offset : offset + g_size]
        g_y = data["y"][offset : offset + g_size]

        order = np.argsort(g_scores)[::-1][:5]
        for rank, idx in enumerate(order, 1):
            sample_records.append(
                {
                    "user_id": uid,
                    "rank": rank,
                    "category_id": g_cids[idx],
                    "category_code": unit_code_map.get(g_cids[idx], "unknown"),
                    "score": round(float(g_scores[idx]), 3),
                    "actual_relevance": int(g_y[idx]),
                }
            )
        offset += g_size

    pd.DataFrame(sample_records).to_parquet(out_dir / "sample_rankings.parquet", index=False)


def _plot_ranking_results(metrics, figures_dir):
    models = ["global_popularity", "segment_popularity", "user_most_viewed", "lightgbm_ltr"]
    labels = ["Global Pop", "Segment Pop", "User Viewed", "LightGBM LTR"]
    ndcgs = [metrics.get(m, {}).get("ndcg_at_10", 0) for m in models]
    recalls = [metrics.get(m, {}).get("recall_at_10", 0) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#888888", "#7570b3", "#1b9e77", "#2b5c8f"]

    ax1.bar(labels, ndcgs, color=colors, edgecolor="white")
    ax1.set_ylabel("NDCG@10")
    ax1.set_title("Ranking Quality: NDCG@10 by Model")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.bar(labels, recalls, color=colors, edgecolor="white")
    ax2.set_ylabel("Recall@10")
    ax2.set_title("Ranking Quality: Recall@10 by Model")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(figures_dir / "banner_ranking_ndcg.png", dpi=150)
    plt.close()


def _json_serialize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray, set)):
        return list(obj)
    return str(obj)
