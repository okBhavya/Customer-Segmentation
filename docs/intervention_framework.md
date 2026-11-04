# Next-Best-Action (NBA) & Intervention Planning Framework

## 1. Executive Summary & Policy Objective
The Next-Best-Action (NBA) engine operationalizes predictive intelligence by determining the optimal intervention for each customer at any given decision point. 

Because the observational eCommerce dataset contains no historically logged marketing interventions or randomized treatment logs, **all policy evaluations and revenue impacts in this framework represent simulated offline estimates under stated unit cost and uplift assumptions**. They are never framed as measured experimental lift.

---

## 2. Action Catalogue & Unit Economics
The engine balances 6 distinct treatment channels with varying unit marginal costs and assumed uplift potentials:

| Action Name | Target Trigger / Eligibility | Assumed Uplift ($\Delta p$) | Unit Marginal Cost ($c$) | Placement Channel |
|---|---|---|---|---|
| **No Action (Control)** | Universal fallback | 0.0% | $0.00 | None (Holdout) |
| **Cart Reminder** | `has_open_cart == 1` | +5.0% | $0.10 | Push notification / Email trigger |
| **Category Cross-Sell** | `unique_categories >= 2` or $p_{\text{purchase}} > 0.15$ | +3.0% | $0.50 | Onsite homepage personalized banner |
| **Price-Led Nudge** | Price-sensitive or low relative price | +4.0% | $1.00 | Onsite coupon / modal discount |
| **Loyalty Early Access** | `Champions` or $p_{\text{purchase}} > 0.25$ | +2.0% | $0.25 | Direct VIP email / In-app badge |
| **Win-Back Message** | $p_{\text{churn}} > 0.40$ or recency $\ge 4$ days | +6.0% (churn save) | $0.75 | SMS / Re-engagement email |

---

## 3. Expected Value (EV) Decision Formulation
For each eligible user $i$ and candidate action $a \in \mathcal{A}_i$:
$$\text{EV}(i, a) = \Delta p_a \times \hat{p}_{\text{purchase}, i} \times \text{AOV}_i - c_a$$

For churn prevention actions (e.g. `win_back_message`):
$$\text{EV}(i, \text{win\_back}) = \Delta p_{\text{churn\_save}} \times \hat{p}_{\text{churn}, i} \times \left(0.5 \times \text{AOV}_i\right) - c_{\text{win\_back}}$$

### Decision Rule
Select action $a^* = \arg\max_{a \in \mathcal{A}_i} \text{EV}(i, a)$ subject to:
1. $\text{EV}(i, a^*) > \text{EV}(i, \text{no\_action}) = 0.0$
2. Contact frequency caps (maximum 3 interventions per user per 7-day period).

If no active intervention produces positive net expected value, the policy strictly defaults to `no_action`.

---

## 4. Segment Playbook & Operational Rules

### Segment: Champions & Loyal Customers
- **Audience Definition**: High historical frequency and spend; high predicted purchase probability.
- **Message Strategy**: Recognition, VIP early catalog access, product drops, loyalty point statements.
- **Timing & Delivery**: Align with modal peak hour (typically 17:00–20:00 UTC) and active weekday patterns.
- **Primary KPI**: Net customer lifetime value (LTV) and 30-day repeat purchase rate.
- **Guardrails**: Avoid heavy discounting or margin dilution; monitor opt-out rates.

### Segment: Window Shoppers & New Promising
- **Audience Definition**: High top-of-funnel view count, 0 lifetime spend, low relative price affinity.
- **Message Strategy**: Category education, social proof, low-barrier first-purchase incentives.
- **Timing & Delivery**: Immediate real-time onsite banner placement during active session.
- **Primary KPI**: View-to-cart rate and first-time buyer conversion.
- **Guardrails**: Frequency cap on intrusive popups; prioritize non-intrusive banner re-ranking.

### Segment: At-Risk & Hibernating
- **Audience Definition**: Prior activity followed by 4+ days of inactivity and elevated churn score ($p_{\text{churn}} > 0.40$).
- **Message Strategy**: "We saved your favorites" / limited-time reactivation coupon.
- **Timing & Delivery**: Weekend morning push (09:00 UTC) when browse re-engagement is highest.
- **Primary KPI**: 7-day reactivation rate and churn reduction.
- **Guardrails**: Cease messaging after 2 unopened communications to protect sender reputation.
