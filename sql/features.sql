-- Feature engineering: one row per user from the lookback window.
-- Placeholders: {{feature_start}}, {{feature_end}}, {{label_start}}, {{label_end}}, {{min_events_for_churn_label}}
-- Expects a table called 'events' with the cleaned parquet schema.

-- ============================================================
-- 1. Scope to lookback window
-- ============================================================
CREATE OR REPLACE TABLE lookback AS
SELECT
    CAST(event_time AS TIMESTAMP) AS event_time,
    event_type,
    product_id,
    category_id,
    category_code,
    brand,
    price,
    user_id,
    user_session
FROM events
WHERE CAST(event_time AS TIMESTAMP) >= TIMESTAMP '{{feature_start}}'
  AND CAST(event_time AS TIMESTAMP) <  TIMESTAMP '{{feature_end}}';

-- ============================================================
-- 2. Session-level aggregates (ordered events per user+session)
-- ============================================================
CREATE OR REPLACE TABLE session_stats AS
SELECT
    user_id,
    user_session,
    MIN(event_time) AS session_start,
    MAX(event_time) AS session_end,
    COUNT(*)        AS session_events,
    EXTRACT(EPOCH FROM (MAX(event_time) - MIN(event_time))) / 60.0 AS session_length_min
FROM lookback
GROUP BY user_id, user_session;

-- ============================================================
-- 3. Category median prices (for relative price feature)
-- ============================================================
CREATE OR REPLACE TABLE category_medians AS
SELECT
    category_code,
    MEDIAN(price) AS cat_median_price
FROM lookback
WHERE category_code != 'unknown'
GROUP BY category_code;

-- ============================================================
-- 4. Peak hour per user (optimized without correlated subqueries)
-- ============================================================
CREATE OR REPLACE TABLE user_peak_hour AS
SELECT
    user_id,
    hour AS peak_hour,
    hour_cnt AS peak_hour_events
FROM (
    SELECT
        user_id,
        EXTRACT(HOUR FROM event_time)::INT AS hour,
        COUNT(*) AS hour_cnt,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, EXTRACT(HOUR FROM event_time)::INT ASC) AS rn
    FROM lookback
    GROUP BY user_id, EXTRACT(HOUR FROM event_time)::INT
) sub
WHERE rn = 1;

-- ============================================================
-- 5. Top category and brand per user
-- ============================================================
CREATE OR REPLACE TABLE user_top_category AS
SELECT
    user_id,
    category_code AS top_category,
    cnt AS top_category_events
FROM (
    SELECT
        user_id,
        category_code,
        COUNT(*) AS cnt,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC) AS rn
    FROM lookback
    GROUP BY user_id, category_code
) sub
WHERE rn = 1;

CREATE OR REPLACE TABLE user_top_brand AS
SELECT
    user_id,
    brand AS top_brand,
    cnt AS top_brand_events
FROM (
    SELECT
        user_id,
        brand,
        COUNT(*) AS cnt,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC) AS rn
    FROM lookback
    GROUP BY user_id, brand
) sub
WHERE rn = 1;

-- ============================================================
-- 6. User sessions aggregation (average session length and gap)
-- ============================================================
CREATE OR REPLACE TABLE user_session_aggs AS
SELECT
    user_id,
    COUNT(*) AS total_sessions,
    AVG(session_length_min) AS avg_session_length_minutes,
    AVG(session_gap_hours) AS avg_session_gap_hours
FROM (
    SELECT
        user_id,
        session_length_min,
        EXTRACT(EPOCH FROM (
            session_start - LAG(session_end) OVER (PARTITION BY user_id ORDER BY session_start)
        )) / 3600.0 AS session_gap_hours
    FROM session_stats
) sg
GROUP BY user_id;

-- ============================================================
-- 7. Open cart detection
-- ============================================================
CREATE OR REPLACE TABLE user_open_cart AS
SELECT
    user_id,
    CASE WHEN cart_adds > purchases THEN 1 ELSE 0 END AS has_open_cart
FROM (
    SELECT
        user_id,
        SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS cart_adds,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases
    FROM lookback
    GROUP BY user_id
) sub;

-- ============================================================
-- 8. Relative price vs category median
-- ============================================================
CREATE OR REPLACE TABLE user_relative_price AS
SELECT
    l.user_id,
    AVG(l.price / NULLIF(cm.cat_median_price, 0)) AS price_relative_to_category_median
FROM lookback l
JOIN category_medians cm ON l.category_code = cm.category_code
GROUP BY l.user_id;

-- ============================================================
-- 9. Base event aggregation and assembly of 50+ features
-- ============================================================
CREATE OR REPLACE TABLE user_features AS
WITH event_agg AS (
    SELECT
        user_id,
        EXTRACT(EPOCH FROM (TIMESTAMP '{{feature_end}}' - MAX(event_time))) / 86400.0 AS days_since_last_event,
        EXTRACT(EPOCH FROM (
            TIMESTAMP '{{feature_end}}' - MAX(CASE WHEN event_type = 'purchase' THEN event_time END)
        )) / 86400.0 AS days_since_last_purchase,
        COUNT(DISTINCT CAST(event_time AS DATE)) AS active_days,
        COUNT(*) AS total_events,
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
        SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS cart_count,
        SUM(CASE WHEN event_type = 'remove_from_cart' THEN 1 ELSE 0 END) AS remove_count,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,
        SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS total_spend,
        AVG(CASE WHEN event_type = 'purchase' THEN price END) AS avg_order_value,
        COUNT(DISTINCT category_code) AS unique_categories,
        COUNT(DISTINCT brand) AS unique_brands,
        AVG(price) AS price_mean,
        STDDEV_SAMP(price) AS price_std,
        MEDIAN(price) AS price_median,
        SUM(CASE WHEN EXTRACT(DOW FROM event_time) BETWEEN 1 AND 5 THEN 1 ELSE 0 END) AS weekday_events,
        SUM(CASE WHEN EXTRACT(HOUR FROM event_time) BETWEEN 6 AND 11 THEN 1 ELSE 0 END) AS morning_events,
        SUM(CASE WHEN EXTRACT(HOUR FROM event_time) BETWEEN 12 AND 17 THEN 1 ELSE 0 END) AS afternoon_events,
        SUM(CASE WHEN EXTRACT(HOUR FROM event_time) BETWEEN 18 AND 22 THEN 1 ELSE 0 END) AS evening_events,
        SUM(CASE WHEN EXTRACT(HOUR FROM event_time) >= 23 OR EXTRACT(HOUR FROM event_time) < 6 THEN 1 ELSE 0 END) AS night_events,
        SUM(CASE WHEN event_time >= TIMESTAMP '{{feature_end}}' - INTERVAL '3 DAYS' THEN 1 ELSE 0 END) AS last_3d_events,
        SUM(CASE WHEN event_time < TIMESTAMP '{{feature_start}}' + INTERVAL '7 DAYS' THEN 1 ELSE 0 END) AS first_7d_events
    FROM lookback
    GROUP BY user_id
)
SELECT
    e.user_id,
    COALESCE(e.days_since_last_event, 999.0) AS days_since_last_event,
    COALESCE(e.days_since_last_purchase, 999.0) AS days_since_last_purchase,
    e.active_days,
    COALESCE(us.total_sessions, 0) AS total_sessions,
    COALESCE(us.avg_session_length_minutes, 0.0) AS avg_session_length_minutes,
    COALESCE(us.avg_session_gap_hours, 0.0) AS avg_session_gap_hours,
    e.view_count,
    e.cart_count,
    e.remove_count,
    e.purchase_count,
    e.total_events,
    CASE WHEN e.view_count > 0 THEN e.cart_count * 1.0 / e.view_count ELSE 0.0 END AS view_to_cart_rate,
    CASE WHEN e.cart_count > 0 THEN e.purchase_count * 1.0 / e.cart_count ELSE 0.0 END AS cart_to_purchase_rate,
    CASE WHEN e.cart_count > 0 THEN (e.cart_count - e.purchase_count) * 1.0 / e.cart_count ELSE 0.0 END AS cart_abandonment_rate,
    COALESCE(oc.has_open_cart, 0) AS has_open_cart,
    e.total_spend,
    COALESCE(e.avg_order_value, 0.0) AS avg_order_value,
    CASE WHEN e.purchase_count > 0 THEN 1 ELSE 0 END AS purchase_count_nonzero,
    e.unique_categories,
    e.unique_brands,
    CASE WHEN e.total_events > 0 THEN COALESCE(tc.top_category_events, 0) * 1.0 / e.total_events ELSE 0.0 END AS top_category_share,
    CASE WHEN e.total_events > 0 THEN COALESCE(tb.top_brand_events, 0) * 1.0 / e.total_events ELSE 0.0 END AS top_brand_share,
    COALESCE(e.price_mean, 0.0) AS price_mean,
    COALESCE(e.price_std, 0.0) AS price_std,
    COALESCE(e.price_median, 0.0) AS price_median,
    COALESCE(ucp.price_relative_to_category_median, 1.0) AS price_relative_to_category_median,
    CASE WHEN e.total_events > 0 THEN e.weekday_events * 1.0 / e.total_events ELSE 0.0 END AS pct_events_weekday,
    COALESCE(ph.peak_hour, 12) AS peak_hour,
    CASE WHEN e.total_events > 0 THEN COALESCE(ph.peak_hour_events, 0) * 1.0 / e.total_events ELSE 0.0 END AS pct_events_peak_hour,
    CASE WHEN e.total_events > 0 THEN e.morning_events * 1.0 / e.total_events ELSE 0.0 END AS pct_events_morning,
    CASE WHEN e.total_events > 0 THEN e.afternoon_events * 1.0 / e.total_events ELSE 0.0 END AS pct_events_afternoon,
    CASE WHEN e.total_events > 0 THEN e.evening_events * 1.0 / e.total_events ELSE 0.0 END AS pct_events_evening,
    CASE WHEN e.total_events > 0 THEN e.night_events * 1.0 / e.total_events ELSE 0.0 END AS pct_events_night,
    e.last_3d_events,
    e.first_7d_events,
    CASE WHEN e.first_7d_events > 0 THEN e.last_3d_events * 1.0 / e.first_7d_events ELSE 0.0 END AS recency_trend_ratio,
    COALESCE(tc.top_category, 'unknown') AS top_category,
    COALESCE(tb.top_brand, 'unknown') AS top_brand,
    CASE WHEN e.active_days > 0 THEN e.total_events * 1.0 / e.active_days ELSE 0.0 END AS events_per_active_day,
    CASE WHEN COALESCE(us.total_sessions, 0) > 0 THEN e.total_events * 1.0 / us.total_sessions ELSE 0.0 END AS events_per_session,
    CASE WHEN e.purchase_count > 0 THEN e.total_spend / e.purchase_count ELSE 0.0 END AS spend_per_purchase,
    CASE WHEN e.view_count > 0 THEN e.purchase_count * 1.0 / e.view_count ELSE 0.0 END AS view_to_purchase_rate,
    CASE WHEN e.unique_categories > 0 THEN e.total_events * 1.0 / e.unique_categories ELSE 0.0 END AS events_per_category,
    CASE WHEN e.unique_brands > 0 THEN e.total_events * 1.0 / e.unique_brands ELSE 0.0 END AS events_per_brand,
    e.cart_count + e.remove_count AS cart_churn_events,
    CASE WHEN e.total_events > 0 THEN (e.cart_count + e.remove_count) * 1.0 / e.total_events ELSE 0.0 END AS cart_churn_rate,
    CASE WHEN e.total_events > 0 THEN e.purchase_count * 1.0 / e.total_events ELSE 0.0 END AS purchase_event_share,
    LOG2(e.total_events + 1.0) AS log_total_events,
    LOG2(e.total_spend + 1.0) AS log_total_spend,
    LOG2(COALESCE(us.total_sessions, 0) + 1.0) AS log_total_sessions
FROM event_agg e
LEFT JOIN user_top_category tc ON e.user_id = tc.user_id
LEFT JOIN user_top_brand tb ON e.user_id = tb.user_id
LEFT JOIN user_peak_hour ph ON e.user_id = ph.user_id
LEFT JOIN user_session_aggs us ON e.user_id = us.user_id
LEFT JOIN user_open_cart oc ON e.user_id = oc.user_id
LEFT JOIN user_relative_price ucp ON e.user_id = ucp.user_id;

-- ============================================================
-- 10. Labels from horizon window
-- ============================================================
CREATE OR REPLACE TABLE horizon AS
SELECT
    CAST(event_time AS TIMESTAMP) AS event_time,
    event_type,
    product_id,
    category_id,
    category_code,
    brand,
    price,
    user_id,
    user_session
FROM events
WHERE CAST(event_time AS TIMESTAMP) >= TIMESTAMP '{{label_start}}'
  AND CAST(event_time AS TIMESTAMP) <  TIMESTAMP '{{label_end}}';

CREATE OR REPLACE TABLE user_labels AS
SELECT
    user_id,
    MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS label_purchase,
    SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0.0 END) AS label_revenue,
    COUNT(*) AS horizon_events
FROM horizon
GROUP BY user_id;

-- ============================================================
-- 11. Final snapshot features and labels
-- ============================================================
CREATE OR REPLACE TABLE snapshot_features AS
SELECT
    uf.*,
    COALESCE(ul.label_purchase, 0) AS label_purchase,
    CASE
        WHEN uf.active_days >= {{min_events_for_churn_label}} THEN
            CASE WHEN ul.user_id IS NULL OR ul.horizon_events = 0 THEN 1 ELSE 0 END
        ELSE NULL
    END AS label_churn,
    COALESCE(ul.label_revenue, 0.0) AS label_revenue
FROM user_features uf
LEFT JOIN user_labels ul ON uf.user_id = ul.user_id;
