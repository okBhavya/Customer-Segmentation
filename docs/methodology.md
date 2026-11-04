# Methodology & Engineering Documentation

## 1. System Overview & Problem Formulation
The **Customer Intelligence Platform** is an enterprise portfolio system designed to solve two core eCommerce personalization and growth challenges:
1. **Strategic Segmentation & Decisioning**: Identifying distinct behavioral customer archetypes, predicting forward 7-day propensity for purchase and churn, and recommending the Next-Best-Action (NBA) via expected value optimization.
2. **Onsite Personalization & Re-Ranking**: Dynamically personalizing promotional banner units (mapped to category-level slots) using Learning-to-Rank (LTR), evaluated counterfactually against strong heuristic baselines.

The system is built on the **REES46 eCommerce behavior dataset** (~42.4M raw events, October 2019).

---

## 2. Temporal Snapshot Design & Leakage Prevention
To ensure zero target leakage and realistic offline validation, data is structured into three strictly disjoint temporal evaluation snapshots:

| Snapshot | Lookback Window (Features) | Horizon Window (Labels) | Purpose |
|---|---|---|---|
| **Train** | Oct 01, 2019 – Oct 10, 2019 (10 days) | Oct 11, 2019 – Oct 17, 2019 (7 days) | Model fitting & feature calibration |
| **Validation** | Oct 08, 2019 – Oct 17, 2019 (10 days) | Oct 18, 2019 – Oct 24, 2019 (7 days) | Hyperparameter tuning & early stopping |
| **Test** | Oct 15, 2019 – Oct 24, 2019 (10 days) | Oct 25, 2019 – Oct 31, 2019 (7 days) | Final out-of-time evaluation |

### Leakage Guardrails
1. **Strict Temporal Partitioning**: Features are computed strictly using events timestamped before snapshot cut-off (`event_time < feature_end`). Labels use events strictly after (`event_time >= label_start`).
2. **Preprocessor Insulation**: All transformers, scalers, and encoders are fit strictly on the **Train** snapshot and applied out-of-sample to Validation and Test snapshots.
3. **Disjoint Horizons**: The test label horizon (Oct 25–31) is completely disjoint from the validation label horizon (Oct 18–24) and training label horizon (Oct 11–17).

---

## 3. Feature Dictionary (50+ Engineered Features)

### Group 1: Recency & Frequency
- `days_since_last_event`: Days elapsed from snapshot end to the user's latest event (recency).
- `days_since_last_purchase`: Days elapsed from snapshot end to latest purchase (999.0 if no purchase).
- `active_days`: Count of distinct calendar days with activity in lookback.
- `total_sessions`: Count of distinct user sessions.
- `avg_session_length_minutes`: Average duration between first and last event in a session.
- `avg_session_gap_hours`: Mean elapsed time between consecutive sessions.
- `total_events`: Total interaction count across lookback.
- `events_per_active_day`: Total events divided by active days.
- `events_per_session`: Total events divided by total sessions.

### Group 2: Funnel & Conversion Dynamics
- `view_count`: Total product page views in lookback.
- `cart_count`: Total cart addition events.
- `remove_count`: Total remove from cart events.
- `purchase_count`: Total purchase transactions completed.
- `view_to_cart_rate`: Ratio of cart additions to views (`cart_count / view_count`).
- `cart_to_purchase_rate`: Ratio of purchases to cart additions (`purchase_count / cart_count`).
- `cart_abandonment_rate`: Abandoned cart fraction (`(cart - purchase) / cart`).
- `has_open_cart`: Binary flag (1 if cart additions exceed purchases).
- `cart_churn_events`: Sum of cart additions and removals (`cart_count + remove_count`).
- `cart_churn_rate`: Fraction of overall activity spent managing carts.
- `purchase_event_share`: Proportion of total events that were purchases.
- `view_to_purchase_rate`: End-to-end direct conversion rate from views to orders.

### Group 3: Monetary & Pricing
- `total_spend`: Cumulative gross merchandise value (GMV) purchased in lookback.
- `avg_order_value`: Average spend per purchase transaction (AOV).
- `purchase_count_nonzero`: Indicator if user completed at least 1 purchase.
- `spend_per_purchase`: Average transaction spend.
- `log_total_spend`: Log-transformed monetary volume (`log2(total_spend + 1)`).
- `price_mean`: Average item price viewed, carted, or bought.
- `price_std`: Standard deviation of viewed item prices.
- `price_median`: Median item price viewed.
- `price_relative_to_category_median`: User's average item price normalized against each category's median price level.

### Group 4: Breadth & Preferences
- `unique_categories`: Distinct product categories interacted with.
- `unique_brands`: Distinct brands interacted with.
- `top_category_share`: Concentration ratio in user's most active category.
- `top_brand_share`: Concentration ratio in user's most active brand.
- `events_per_category`: Mean interaction density per category.
- `events_per_brand`: Mean interaction density per brand.
- `top_category`: Modal category code.
- `top_brand`: Modal brand name.

### Group 5: Temporal & Momentum Patterns
- `pct_events_weekday`: Fraction of events occurring Monday through Friday.
- `peak_hour`: Hour of day (0-23 UTC) with maximum interaction frequency.
- `pct_events_peak_hour`: Concentration of activity in user's peak hour.
- `pct_events_morning`: Proportion of events between 06:00 and 11:59 UTC.
- `pct_events_afternoon`: Proportion of events between 12:00 and 17:59 UTC.
- `pct_events_evening`: Proportion of events between 18:00 and 22:59 UTC.
- `pct_events_night`: Proportion of events between 23:00 and 05:59 UTC.
- `last_3d_events`: Event volume in the final 3 days of lookback.
- `first_7d_events`: Event volume in the initial 7 days of lookback.
- `recency_trend_ratio`: Velocity momentum metric (`last_3d_events / first_7d_events`).
- `log_total_events`: Log-transformed event frequency.
- `log_total_sessions`: Log-transformed session frequency.

### Target Labels (Horizon Window)
- `label_purchase`: Binary indicator (1 if user purchased in next 7 days, else 0).
- `label_churn`: Binary churn indicator (1 if user had 0 events in horizon; defined strictly for users with `active_days >= 2` in lookback).
- `label_revenue`: Total monetary spend in the 7-day horizon.

---

## 4. Modeling & Algorithmic Strategy
- **Segmentation**: RFM rule-based baseline vs. K-Means and GMM clustering on log-transformed, standardized features. Cluster count $k$ is selected via Silhouette, Davies-Bouldin, BIC, and bootstrap ARI stability.
- **Propensity Scoring**: Class-imbalanced LightGBM with Optuna hyperparameter optimization against regularized Logistic Regression and RFM heuristics. Post-hoc probability calibration is conducted via Isotonic Regression.
- **Banner Ranking**: LightGBM LambdaRank trained with pairwise/listwise objectives and negative sampling on category promotional units. Graded relevance reflects conversion depth (Purchase=3, Cart=2, View=1).
- **Counterfactual Offline Evaluation**: Evaluated using Self-Normalized Inverse Propensity Scoring (SNIPS) with weight clipping to mitigate observational exposure bias.
