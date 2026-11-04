# Customer Segmentation Documentation

## 1. Segmentation Strategy & Objectives
E-commerce customer bases are structurally heterogeneous: a small minority of power buyers drives the vast majority of revenue, while casual browsers constitute the bulk of top-of-funnel traffic. 

This engine implements and evaluates two complementary segmentation paradigms:
1. **RFM Heuristic Baseline**: A deterministic, rule-based approach partitioning customers into documented tiers based on Recency, Frequency, and Monetary spend.
2. **Behavioral Machine Learning (K-Means & GMM)**: Multi-dimensional clustering across 14 normalized behavioral signals (activity intensity, catalog breadth, conversion velocity, price point affinity, session duration, and weekend preferences).

---

## 2. RFM Thresholds & Implementation Safeguards
Monetary spend in eCommerce is heavily zero-inflated (over 80% of window users complete zero purchases). Classical quartile binning (`qcut`) fails due to non-unique bin edges.

### Safeguard Design
- **Recency (R)**: Partitioned into 4 bins over `days_since_last_event`. Lower recency = higher rank (4 = active in last 2 days; 1 = inactive for >8 days).
- **Frequency (F)**: Partitioned into 4 bins over `total_events` using `pd.cut` with `duplicates='drop'`.
- **Monetary (M)**: Evaluated conditionally. If `total_spend == 0`, score is explicitly 0. For users with positive spend, non-zero values are binned into ranks 1–4.

### Canonical Segment Definitions
| Segment Name | Typical Score Profile | Business Archetype |
|---|---|---|
| **Champions** | `R >= 3, F >= 3, M >= 3` | Recent high-frequency buyers with high lifetime spend. |
| **Loyal Customers** | `R >= 3, F >= 2, M >= 1` | Steady active purchasers who buy regularly. |
| **New / Promising** | `R >= 3, F < 2` | Recently arrived users with low event counts but high immediate potential. |
| **At Risk** | `R == 2, F >= 2` | Formerly active users showing declining interaction over recent days. |
| **Hibernating** | `R == 1, F >= 2` | Unengaged users with long dormant periods since last session. |
| **Window Shoppers** | `M == 0, F < 2` | Zero purchases and low view counts; high bounce probability. |
| **Need Attention** | All other profiles | Moderate activity but inconsistent conversion path. |

---

## 3. Unsupervised Machine Learning Clustering (k=7)
- **Feature Space**: Log-transformed (`log1p`) behavioral attributes to compress long tails in spend and event counts, followed by `StandardScaler` fitted strictly on the training snapshot.
- **Cluster Count Selection ($k$)**: Evaluated over $k \in [4, 5, 6, 7]$.
  - $k=4$: Silhouette = 0.2418, Davies-Bouldin = 1.5938, BIC = -600,325, ARI = 0.6122
  - $k=5$: Silhouette = 0.2756, Davies-Bouldin = 1.1625, BIC = -644,259, ARI = 0.6738
  - $k=6$: Silhouette = 0.2827, Davies-Bouldin = 1.1516, BIC = -623,221, ARI = 0.7544
  - $k=7$: Silhouette = 0.2875, Davies-Bouldin = 1.2444, BIC = -636,115, ARI = 0.6115
- **Selected $k=7$**: Optimal silhouette coefficient (0.2875) isolating clear behavioral archetypes (including separating direct 1-click purchasers from traditional cart-and-checkout buyers).

---

## 4. Segment Profiles & Action Plan (Evaluated on Test Snapshot)

All figures below are evaluated out-of-time on the **Test Snapshot** (1,457,672 users; Oct 15–24 lookback, Oct 25–31 forward horizon):

| Cluster ID | Segment Name | % Users | Test Count | Lookback Avg Spend | Forward Purchase Rate (95% CI) | Forward Churn Rate | Test Revenue Share | Recommended Intervention | Target KPI |
|---|---|---|---|---|---|---|---|---|---|
| **Cluster 2** | Power Buyers (Cart & Checkout) | 6.36% | 92,745 | $691.83 | 16.48% [16.24%, 16.72%] | 51.96% | **47.43%** | VIP loyalty perk, premium cross-sell, concierge service | Net Revenue, LTV |
| **Cluster 3** | Direct 1-Click Purchasers | 3.18% | 46,377 | $350.37 | 12.28% [11.98%, 12.58%] | 55.20% | **11.39%** | Fast-checkout category re-engagement, post-purchase perks | Order Frequency |
| **Cluster 5** | Catalog Explorers (Heavy Browsers) | 7.84% | 114,281 | $9.50 | 6.90% [6.76%, 7.05%] | **35.29%** | 9.57% | Personalized category banners, price-drop alerts, urgency nudges | View-to-Cart Conversion |
| **Cluster 1** | Deep Comparison Shoppers | 21.82% | 318,045 | $0.00 | 2.97% [2.91%, 3.02%] | 56.60% | 12.54% | Side-by-side comparison highlights, reviews, first-order incentive | First-Time Conversion |
| **Cluster 4** | Recent Casual Visitors | 24.56% | 357,999 | $0.01 | 1.52% [1.48%, 1.56%] | 64.14% | 9.02% | Trending items category banner, popular products showcase | Session Depth / Views |
| **Cluster 6** | Mid-Recency Bouncers | 13.69% | 199,579 | $0.07 | 1.34% [1.29%, 1.39%] | 67.72% | 3.81% | Exit intent re-engagement, top browsed category reminder | Return Visit Rate |
| **Cluster 0** | Dormant Window Shoppers | 22.55% | 328,646 | $0.02 | 1.28% [1.25%, 1.32%] | 71.96% | 6.23% | Automated low-cost win-back email (strict contact frequency cap) | Reactivation Rate |

---

## 5. Segment Migration & Out-of-Time Predictive Validity

### Migration Dynamics
- **K-Means Migration (Train $\to$ Val)**: 71.81% of users transition segments between consecutive 10-day lookback windows, reflecting high velocity in short-term e-commerce intent.
- **K-Means Migration (Val $\to$ Test)**: 70.56% of users transition.
- **GMM Soft Cluster Migration**: 48.16% (Train $\to$ Val) and 49.39% (Val $\to$ Test), indicating smoother transitions when probabilistic density is modeled.

### Outcome Variance Explained ($R^2$ on Test Snapshot)
- **Forward Purchase Rate ($R^2$)**:
  - RFM Baseline: $0.0091$ ($0.91\%$)
  - **K-Means ($k=7$)**: **$0.0503$ ($5.03\%$)** $\to$ **$5.5\times$ lift in predictive variance over RFM**
  - Gaussian Mixture: $0.0557$ ($5.57\%$) $\to$ $6.1\times$ lift over RFM
- **Forward Revenue ($R^2$)**:
  - RFM Baseline: $0.0021$ ($0.21\%$)
  - **K-Means ($k=7$)**: **$0.0172$ ($1.72\%$)** $\to$ **$8.2\times$ lift in predictive variance over RFM**
  - Gaussian Mixture: $0.0215$ ($2.15\%$) $\to$ $10.2\times$ lift over RFM
