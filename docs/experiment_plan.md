# Experimentation & Causal Evaluation Plan

## 1. Executive Summary & Hypotheses
Online randomized controlled trials (A/B testing) are the gold standard for validating whether machine learning models and personalized interventions deliver genuine incremental business value.

### Primary Hypotheses
- **$H_1$ (Onsite Banner Personalization)**: Presenting dynamic category banners ranked via LightGBM LambdaRank will increase overall 7-day user purchase conversion by at least $\ge 3\%$ relative to a popularity-ordered baseline banner.
- **$H_2$ (Next-Best-Action Optimization)**: Targeting customers using the calibrated Model Expected Value (EV) policy will generate higher net incremental revenue per user than uniform blanket promotions or rule-based routing.

---

## 2. Experimental Design Architecture
- **Unit of Randomization**: `user_id` (cookie-stable persistent identifier).
- **Randomization Strategy**: Hash-based salted modular bucketing:
  $$\text{Bucket}(u) = \text{MD5}(\text{user\_id} + \text{"\_exp\_salt\_2019"})\pmod{100}$$
- **Holdout Structure**:
  - **Universal Holdout (10%)**: Strictly receives `no_action` and global popularity ranking to measure long-term system drift and baseline conversion.
  - **Treatment Arm 1 (45%)**: Heuristic baseline (RFM rules + Category Popularity ranking).
  - **Treatment Arm 2 (45%)**: Algorithmic policy (Calibrated LightGBM Propensity EV + LambdaRank).
- **Stratification**: Pre-experiment stratification across behavioral segments (Champions, Loyal, At-Risk, Window Shoppers) to ensure balanced segment proportions across arms.

---

## 3. Metric Hierarchy & Guardrails

| Metric Classification | Metric Name | Definition | Measurement Window |
|---|---|---|---|
| **Primary Metric** | Purchase Conversion Rate | Fraction of users completing $\ge 1$ purchase | 7 days post-exposure |
| **Co-Primary Metric** | Revenue Per User (ARPU) | Total spend / total assigned users | 7 days post-exposure |
| **Secondary Metric** | Category Click-Through Rate | Clicks on top-10 ranked banner slots / impressions | Real-time session |
| **Secondary Metric** | Cart-to-Purchase Rate | Completed purchases / users who added to cart | 7 days |
| **Guardrail Metric** | Communication Opt-Out Rate | Unsubscribe / notification block rate | Real-time / Daily |
| **Guardrail Metric** | Gross Margin Per Order | Net revenue minus promotional discount and delivery cost | Per transaction |
| **Technical Guardrail**| P99 Onsite Latency | Time elapsed to compute ranking recommendations | $< 50\text{ ms}$ |

---

## 4. Power Analysis & Minimum Detectable Effect (MDE)
Evaluated using two-sided two-sample proportion tests at significance level $\alpha = 0.05$ and statistical power $1 - \beta = 0.80$:

- Baseline Purchase Conversion Rate: $\approx 5.0\%$
- Required Sample Sizes (per arm):
  - **$\text{MDE} = 10\%$ relative** ($\Delta = 0.005$): $\approx 31,000$ users per arm.
  - **$\text{MDE} = 5\%$ relative** ($\Delta = 0.0025$): $\approx 125,000$ users per arm.
  - **$\text{MDE} = 2\%$ relative** ($\Delta = 0.001$): $\approx 780,000$ users per arm.

Given weekly traffic exceeding 1.2M active users, a 14-day experiment provides adequate statistical power to detect relative conversion lifts as small as **$2.5\%$ to $3.0\%$**.

---

## 5. Advanced Statistical Controls

### CUPED (Controlled-experiment Using Pre-Experiment Data)
To reduce variance in metric estimation and accelerate experiment runtime, we apply CUPED using pre-period lookback spend ($X$) as the covariate for post-period revenue ($Y$):
$$\hat{Y}_{\text{CUPED}} = Y - \hat{\theta}(X - E[X]), \quad \text{where } \hat{\theta} = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
With pre-post correlation $\rho \approx 0.50 - 0.60$, CUPED achieves **$25\% - 35\%$ variance reduction**, effectively reducing required sample sizes and test duration by roughly one-third.

### Sample Ratio Mismatch (SRM) Monitoring
Continuous verification using Chi-Square goodness-of-fit testing against the intended 50/50 allocation ratio:
$$\chi^2 = \sum_{j} \frac{(O_j - E_j)^2}{E_j}, \quad \text{threshold: } p < 0.001$$
Any experiment exhibiting $p < 0.001$ is immediately halted for assignment bias, redirect failures, or tracking drops.

### Multiple Comparison Correction
When evaluating secondary metrics across multiple segment cuts (e.g. testing for heterogeneous treatment effects across 5 segments), we apply the **Holm-Bonferroni step-down procedure** to strictly control the Family-Wise Error Rate (FWER) at $5\%$.
