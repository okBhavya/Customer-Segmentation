"""Generate clean, executable EDA notebook notebooks/01_eda.ipynb."""

from pathlib import Path

import nbformat as nbf


def build_notebook(out_path: Path):
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell("""# 01 - Exploratory Data Analysis & Funnel Diagnostics

**Project**: Customer Segmentation, Next-Best-Action & Banner Ranking Engine
**Dataset**: REES46 eCommerce Behavior Data (October 2019, 42M+ events)
**Author**: Bhavya Gupta

---
### Objective
This notebook explores the cleaned behavior dataset, analyzes conversion funnel drop-offs, and quantifies onsite optimization opportunities across categories, hours of day, and customer behavior tiers.
"""),
        nbf.v4.new_code_cell("""import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load consolidated data profile and funnel metrics
reports_dir = Path("../reports")
figures_dir = Path("../reports/figures")

with open(reports_dir / "data_profile.json") as f:
    profile = json.load(f)

with open(reports_dir / "funnel_report.json") as f:
    funnel = json.load(f)
"""),
        nbf.v4.new_markdown_cell("""## 1. Data Cleaning & Profile Summary
The raw dataset contains 42,448,764 events. Cleaning steps included:
- Normalizing timestamps to UTC
- Filtering rows with price <= 0
- Deduplicating exact duplicate records
- Handling missing category codes and brands
"""),
        nbf.v4.new_code_cell("""print(f"Total Raw Events:        {profile['total_raw_events']:,}")
print(f"Total Clean Events:      {profile['total_clean_events']:,}")
print(f"Duplicates Removed:      {profile['duplicates_removed']:,}")
print(f"Invalid Price Removed:   {profile['price_invalid_removed']:,}")
print(f"Unique Customers:        {profile['unique_users']:,}")
print(f"Unique Catalog Products: {profile['unique_products']:,}")
print(f"Observation Window:      {profile['date_range']['min']} to {profile['date_range']['max']}")
"""),
        nbf.v4.new_markdown_cell("""## 2. Event Type Distribution
E-commerce user interactions are heavily skewed toward page views, followed by cart additions and purchases.
"""),
        nbf.v4.new_code_cell("""event_mix = pd.Series(profile["event_type_counts"])
fig, ax = plt.subplots(figsize=(7, 4))
event_mix.plot(kind="bar", ax=ax, color=["#2b5c8f", "#d95f02", "#1b9e77"], edgecolor="white")
ax.set_title("Event Distribution (October 2019)", fontsize=11)
ax.set_ylabel("Event Count")
ax.set_yscale("log")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""),
        nbf.v4.new_markdown_cell("""## 3. Conversion Funnel & Drop-Off Dynamics
We evaluate the progression of unique users from initial browsing to cart addition and final transaction completion.
"""),
        nbf.v4.new_code_cell("""ov = funnel["overall_funnel"]
print(f"Unique Viewers:            {ov['viewers']:,}")
print(f"Unique Carters:            {ov['carters']:,}  (View->Cart: {ov['view_to_cart_rate']:.2%})")
print(f"Unique Purchasers:          {ov['purchasers']:,}  (Cart->Purchase: {ov['cart_to_purchase_rate']:.2%})")
print(f"Overall User Conversion:   {ov['overall_conversion']:.2%}")
print(f"Cart Abandonment Rate:     {funnel['cart_abandonment']['abandonment_rate']:.2%}")
if 'time_to_purchase' in funnel:
    print(f"Median Time to Purchase:   {funnel['time_to_purchase']['median_minutes']} minutes")
"""),
        nbf.v4.new_markdown_cell("""## 4. Top Onsite Opportunity Categories
Categories with high traffic volume but below-average cart-to-purchase conversion represent prime targets for promotional banner re-ranking and cart recovery interventions.
"""),
        nbf.v4.new_code_cell("""opps = pd.DataFrame(funnel.get("opportunities", []))
if not opps.empty:
    display_cols = ["category_code", "viewers", "carters", "cart_to_purchase", "potential_incremental_orders"]
    print(opps[display_cols].head(10).to_string(index=False))
"""),
        nbf.v4.new_markdown_cell("""## 5. Conclusions & Next Steps
1. Over 80% of users who add items to cart abandon before completing a purchase, highlighting huge potential for cart reminders and price-led nudges.
2. Conversion velocity is rapid (median time to purchase is within 30 minutes of session start), necessitating low-latency real-time onsite recommendations.
3. In subsequent notebooks and pipeline stages, we model customer segments and deploy calibrated propensity scoring and category banner re-ranking.
"""),
    ]

    nb.cells = cells
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook at {out_path}")


if __name__ == "__main__":
    p = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"
    build_notebook(p)
