# Verified Resume Metrics

All metrics recorded below are verified from end-to-end execution of the pipeline on the real REES46 eCommerce dataset (`2019-Oct.csv`) with fixed random seeds (`seed: 42`). No values are estimated, rounded up, or fabricated.

---

## 1. Verified Metrics Table

| Metric Category | Metric Description | Exact Real Value | Source File / Artifact | Generation Command |
|---|---|---|---|---|
| **Data Scale** | Raw clickstream events processed | `42,448,764` | `reports/data_profile.json` | `python -m custintel ingest` |
| **Data Scale** | Clean deduplicated events | `42,349,874` | `reports/data_profile.json` | `python -m custintel ingest` |
| **Data Scale** | Exact duplicate rows removed | `30,220` | `reports/data_profile.json` | `python -m custintel ingest` |
| **Data Scale** | Invalid prices filtered (`price <= 0`) | `68,673` | `reports/data_profile.json` | `python -m custintel ingest` |
| **Data Scale** | Unique customer users | `3,021,435` | `reports/data_profile.json` | `python -m custintel ingest` |
| **Data Scale** | Unique catalog products | `165,647` | `reports/data_profile.json` | `python -m custintel ingest` |
| **Data Scale** | Date range | `2019-10-01 00:00:00 UTC` to `2019-10-31 23:59:59 UTC` | `reports/data_profile.json` | `python -m custintel ingest` |
| **Data Scale** | Source dataset file | `2019-Oct.csv` (REES46 eCommerce multi-category store) | `data/raw/2019-Oct.csv` | Downloaded from Kaggle |
| **Features** | Total engineered behavioral features | `48` | `reports/feature_summary.json` | `python -m custintel features` |
| **Snapshots** | Train snapshot users (Oct 1–10 / 11–17) | `1,302,363` | `reports/feature_summary.json` | `python -m custintel features` |
| **Snapshots** | Validation snapshot users (Oct 8–17 / 18–24) | `1,397,275` | `reports/feature_summary.json` | `python -m custintel features` |
| **Snapshots** | Test snapshot users (Oct 15–24 / 25–31) | `1,457,672` | `reports/feature_summary.json` | `python -m custintel features` |
| **Segmentation** | Chosen cluster count ($k$) | `7` | `reports/segment_metrics.json` | `python -m custintel segment` |
| **Segmentation** | K-Means Silhouette score ($k=7$) | `0.2875` (sample `0.2560`) | `reports/segment_metrics.json` | `python -m custintel segment` |
| **Segmentation** | K-Means Davies-Bouldin index ($k=7$) | `1.2444` (sample `1.3848`) | `reports/segment_metrics.json` | `python -m custintel segment` |
| **Segmentation** | K-Means Calinski-Harabasz score ($k=7$) | `2333.6` (sample $n=10\text{k}$) | `scripts/audit_pipeline_and_clustering.py` | `python scripts/audit_pipeline_and_clustering.py` |
| **Segmentation** | GMM Silhouette score ($k=7$) | `0.2329` (sample $n=10\text{k}$) | `scripts/audit_pipeline_and_clustering.py` | `python scripts/audit_pipeline_and_clustering.py` |
| **Segmentation** | GMM Davies-Bouldin index ($k=7$) | `2.3720` (sample $n=10\text{k}$) | `scripts/audit_pipeline_and_clustering.py` | `python scripts/audit_pipeline_and_clustering.py` |
| **Segmentation** | GMM Calinski-Harabasz score ($k=7$) | `1579.5` (sample $n=10\text{k}$) | `scripts/audit_pipeline_and_clustering.py` | `python scripts/audit_pipeline_and_clustering.py` |
| **Segmentation** | GMM BIC ($k=7$) | `-636,115.3` | `reports/segment_metrics.json` | `python -m custintel segment` |
| **Segmentation** | Selected algorithm | `K-Means (k=7)` on log-standardized features | `reports/segment_metrics.json` | `python -m custintel segment` |
| **Segmentation** | Clustering determinism (fixed seed) | `ARI = 1.0000` (exact reproducibility) | `scripts/audit_pipeline_and_clustering.py` | `python scripts/audit_pipeline_and_clustering.py` |
| **Segmentation** | Clustering stability across 10 random seeds | `Mean ARI = 0.9214` (range: `0.8456` – `1.0000`) | `scripts/audit_pipeline_and_clustering.py` | `python scripts/audit_pipeline_and_clustering.py` |
| **Segmentation** | Forward purchase $R^2$ (Test snapshot) | `0.0503` (5.03% variance explained) | `reports/segment_metrics.json` | `python -m custintel segment` |
| **Segmentation** | RFM baseline forward purchase $R^2$ | `0.0091` (0.91% variance explained) | `reports/segment_metrics.json` | `python -m custintel segment` |
| **Segmentation** | Lift in variance explained over RFM | `5.5x lift` ($0.0503 / 0.0091$) | `reports/segment_metrics.json` | `python -m custintel segment` |
| **Propensity (Purchase)** | RFM baseline ROC-AUC (Test snapshot) | `0.6787` (PR-AUC: `0.1092`) | `reports/propensity_metrics.json` | `python -m custintel propensity` |
| **Propensity (Purchase)** | Logistic Regression ROC-AUC (Test snapshot) | `0.7806` (PR-AUC: `0.2096`) | `reports/propensity_metrics.json` | `python -m custintel propensity` |
| **Propensity (Purchase)** | LightGBM Calibrated ROC-AUC (Test snapshot) | `0.7783` (PR-AUC: `0.1569`) | `reports/propensity_metrics.json` | `python -m custintel propensity` |
| **Propensity (Purchase)** | LightGBM vs RFM ROC-AUC difference | `+0.0996` [95% CI: `+0.0862`, `+0.1152`] | `reports/propensity_metrics.json` | `python -m custintel propensity` |
| **Propensity (Purchase)** | LightGBM vs RFM relative improvement | `+14.68%` ($+0.0996 / 0.6787$) | `reports/propensity_metrics.json` | Calculated |
| **Propensity (Purchase)** | Brier score (Raw vs Calibrated LightGBM) | `0.0356` (Raw) $\to$ `0.0309` (Calibrated) | `reports/propensity_metrics.json` | `python -m custintel propensity` |
| **Propensity (Purchase)** | Top decile lift & precision | Lift: `4.68x`; Precision: `16.27%` | `reports/propensity_metrics.json` | `python -m custintel propensity` |
| **Propensity (Churn)** | RFM baseline ROC-AUC (Test snapshot) | `0.5189` (PR-AUC: `0.5473`) | `reports/propensity_metrics.json` | `python -m custintel propensity` |
| **Propensity (Churn)** | Logistic Regression ROC-AUC (Test snapshot) | `0.6748` (PR-AUC: `0.6698`) | `reports/propensity_metrics.json` | `python -m custintel propensity` |
| **Propensity (Churn)** | LightGBM Calibrated ROC-AUC (Test snapshot) | `0.6823` (PR-AUC: `0.6794`) | `reports/propensity_metrics.json` | `python -m custintel propensity` |
| **Propensity (Churn)** | LightGBM vs RFM ROC-AUC difference | `+0.1634` [95% CI: `+0.1530`, `+0.1716`] | `reports/propensity_metrics.json` | `python -m custintel propensity` |
| **Propensity (Churn)** | LightGBM vs RFM relative improvement | `+31.49%` ($+0.1634 / 0.5189$) | `reports/propensity_metrics.json` | Calculated |
| **Ranker** | Banner units evaluated | `239` category slots | `reports/ranking_metrics.json` | `python -m custintel rank` |
| **Ranker** | Global Popularity baseline NDCG@10 | `0.2322` | `reports/ranking_metrics.json` | `python -m custintel rank` |
| **Ranker** | User Most-Viewed baseline NDCG@10 | `0.1660` | `reports/ranking_metrics.json` | `python -m custintel rank` |
| **Ranker** | LightGBM LambdaRank NDCG@10 | `0.2382` | `reports/ranking_metrics.json` | `python -m custintel rank` |
| **Ranker** | Relative NDCG@10 lift over Popularity | `+2.58%` [95% CI: `+2.04%`, `+3.18%`] | `reports/ranking_metrics.json` | `python -m custintel rank` |
| **Ranker (IPS)** | Counterfactual Estimator | Self-Normalized Inverse Propensity Scoring (SNIPS) | `reports/ranking_metrics.json` | `python -m custintel rank` |
| **Ranker (IPS)** | Naive CTR vs SNIPS Debiased CTR Proxy | Naive: `0.07171` $\to$ SNIPS: `0.04796` [95% CI: `0.04651`, `0.04957`] | `reports/ranking_metrics.json` | `python -m custintel rank` |
| **A/B Test Plan** | Baseline purchase conversion rate | `0.0347` (3.47%) | `reports/experiment_report.json` | `python -m custintel experiment` |
| **A/B Test Plan** | Significance level ($\alpha$) & Power ($1 - \beta$) | $\alpha = 0.05$ (two-sided), Power $= 0.80$ | `reports/experiment_report.json` | `python -m custintel experiment` |
| **A/B Test Plan** | Minimum Detectable Effect (relative MDE) | `5.0% relative` (+0.174% absolute lift) | `reports/experiment_report.json` | `python -m custintel experiment` |
| **A/B Test Plan** | Sample size per arm (5% MDE) | `178,706 users` | `reports/experiment_report.json` | `python -m custintel experiment` |
| **A/B Test Plan** | Total sample size for 2 arms (5% MDE) | `357,412 users` | `reports/experiment_report.json` | `python -m custintel experiment` |
| **A/B Test Plan** | CUPED variance reduction (pre-spend covariate) | `17.11% reduction` ($\rho = 0.4136$) | `reports/experiment_report.json` | `python -m custintel experiment` |
| **A/B Test Plan** | Simulation validation (A/A FPR & A/B Power) | A/A FPR: `0.0580` (target 0.05); A/B Power: `0.8030` (target 0.80) | `reports/experiment_report.json` | `python -m custintel experiment` |
| **Testing** | Total unit and integration tests | `38` | `tests/` | `pytest` |
| **Testing** | Passing tests | `38 passed (100%)`, 0 failed, 0 skipped, 0 flaky | `tests/` | `pytest` |
| **Testing** | Code coverage on package library | `95%` | `.pytest_cache` / `pyproject.toml` | `pytest --cov=custintel` |

---

## 2. 48 Engineered Features Grouped

```text
RFM / Monetary (11 features):
  - days_since_last_event
  - days_since_last_purchase
  - total_events
  - active_days
  - total_spend
  - avg_order_value
  - purchase_count
  - purchase_count_nonzero
  - spend_per_purchase
  - log_total_events
  - log_total_spend

Recency & Temporal Dynamics (10 features):
  - last_3d_events
  - first_7d_events
  - recency_trend_ratio
  - pct_events_weekday
  - peak_hour
  - pct_events_peak_hour
  - pct_events_morning
  - pct_events_afternoon
  - pct_events_evening
  - pct_events_night

Session & Funnel Behavior (17 features):
  - total_sessions
  - log_total_sessions
  - avg_session_length_minutes
  - avg_session_gap_hours
  - events_per_session
  - events_per_active_day
  - view_count
  - cart_count
  - remove_count
  - view_to_cart_rate
  - cart_to_purchase_rate
  - view_to_purchase_rate
  - cart_abandonment_rate
  - has_open_cart
  - cart_churn_events
  - cart_churn_rate
  - purchase_event_share

Category & Brand Affinity / Price Sensitivity (10 features):
  - unique_categories
  - unique_brands
  - top_category_share
  - top_brand_share
  - events_per_category
  - events_per_brand
  - price_mean
  - price_std
  - price_median
  - price_relative_to_category_median
```

---

## 3. Resume Bullets (Strict 14–15 Words, Verb Start, No Trailing Period)

### Bullet A: Data Pipeline + Scale + Number of Features
```text
Engineered DuckDB pipeline processing 42.4M clickstream events across 3.02M users to extract 48 behavioral features
```
*(Word count: exactly 15 words. Starts with "Engineered".)*

### Bullet B: Segmentation + Propensity AUC (Before to After)
```text
Clustered customers into 7 K-Means segments and increased purchase propensity AUC from 0.68 to 0.78
```
*(Word count: exactly 15 words. Starts with "Clustered".)*

*Alternative for Churn Propensity (also verified)*:
```text
Clustered customers into 7 K-Means segments and increased churn propensity AUC from 0.52 to 0.68
```
*(Word count: exactly 15 words. Starts with "Clustered".)*

### Bullet C: Ranker NDCG@10 Lift + A/B Plan Sample Size
```text
Trained LambdaRank achieving 2.6% NDCG@10 lift over popularity with 178,706 users per arm experiment plan
```
*(Word count: exactly 15 words. Starts with "Trained".)*

---

## 4. Numbers to Flag & Avoid on Resume

1. **LightGBM vs Logistic Regression Raw AUC (+0.7783 vs 0.7806)**:
   - *Flag*: Do **NOT** claim LightGBM beat regularized logistic regression in raw AUC. Logistic regression achieved 0.7806 while LightGBM achieved 0.7783.
   - *Safe framing*: Compare LightGBM against the heuristic **RFM baseline** (0.6787 $\to$ 0.7783, +14.7% relative improvement) or cite LightGBM's superior **probability calibration** (Brier score 0.0309 vs 0.1850; log-loss 0.1292 vs 0.5653).
2. **Offline CTR Lift (Do NOT claim measured CTR lift)**:
   - *Flag*: The REES46 dataset contains **no banner impression logs**. The SNIPS metric of 0.04796 is an offline debiasing estimator of category engagement, not a measured A/B test CTR lift.
   - *Safe framing*: State "applied counterfactual SNIPS weighting to correct for observational popularity bias" or focus on the **+2.58% NDCG@10 ranking lift**.
3. **NBA Dollar Lift (Do NOT claim measured dollar revenue)**:
   - *Flag*: The $3.01M simulated expected value is calculated from assumed unit costs ($0.10 cart reminder, $1.00 coupon) and assumed uplifts (+5%), because no historical marketing interventions were logged.
   - *Safe framing*: Frame this as "built an expected-value optimization policy simulation demonstrating a 5.6x cost-benefit advantage over blanket discounting".
4. **LambdaRank Recall@10 (0.2576 vs 0.2582)**:
   - *Flag*: LambdaRank's recall is essentially flat compared to global popularity (0.2576 vs 0.2582). The gain is entirely in **ranking order at positions 1–3** (NDCG@10 lift of +2.58%).
   - *Safe framing*: Always cite **NDCG@10**, never Recall@10.
