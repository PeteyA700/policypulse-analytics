import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

np.random.seed(42)

# --------------------------------------------------
# 1. CREATE SYNTHETIC INSURANCE CUSTOMER DATASET
# --------------------------------------------------

n = 5000

customer_id = np.arange(1, n + 1)

age = np.random.randint(18, 75, n)
income = np.random.normal(65000, 22000, n).clip(20000, 160000)
tenure_months = np.random.randint(1, 120, n)

policy_type = np.random.choice(
    ["Auto", "Home", "Renters", "Life"],
    size=n,
    p=[0.45, 0.25, 0.20, 0.10]
)

monthly_premium = np.round(
    np.random.normal(135, 45, n).clip(40, 350),
    2
)

claim_count = np.random.poisson(0.7, n)
customer_service_calls = np.random.poisson(1.2, n)
late_payments = np.random.poisson(0.4, n)

satisfaction_score = np.random.randint(1, 11, n)

discount_received = np.random.choice([0, 1], size=n, p=[0.65, 0.35])

# Churn probability logic
churn_risk = (
    -2.2
    + 0.018 * monthly_premium
    + 0.35 * customer_service_calls
    + 0.45 * late_payments
    - 0.035 * tenure_months
    - 0.28 * satisfaction_score
    - 0.40 * discount_received
)

prob_churn = 1 / (1 + np.exp(-churn_risk))
churned = np.random.binomial(1, prob_churn)

df = pd.DataFrame({
    "customer_id": customer_id,
    "age": age,
    "income": income.round(2),
    "tenure_months": tenure_months,
    "policy_type": policy_type,
    "monthly_premium": monthly_premium,
    "claim_count": claim_count,
    "customer_service_calls": customer_service_calls,
    "late_payments": late_payments,
    "satisfaction_score": satisfaction_score,
    "discount_received": discount_received,
    "churned": churned
})

df["annual_revenue"] = df["monthly_premium"] * 12

df.to_csv("data/insurance_customers.csv", index=False)

# --------------------------------------------------
# 2. LOAD INTO SQLITE DATABASE
# --------------------------------------------------

conn = sqlite3.connect("data/insurance.db")
df.to_sql("customers", conn, if_exists="replace", index=False)

# --------------------------------------------------
# 3. BUSINESS KPI ANALYSIS USING SQL
# --------------------------------------------------

query = """
SELECT
    policy_type,
    COUNT(*) AS customers,
    ROUND(AVG(monthly_premium), 2) AS avg_monthly_premium,
    ROUND(AVG(annual_revenue), 2) AS avg_annual_revenue,
    ROUND(AVG(churned) * 100, 2) AS churn_rate_percent,
    ROUND(SUM(annual_revenue), 2) AS total_annual_revenue
FROM customers
GROUP BY policy_type
ORDER BY churn_rate_percent DESC;
"""

policy_summary = pd.read_sql(query, conn)
policy_summary.to_csv("outputs/policy_summary.csv", index=False)

print("\nPOLICY SUMMARY")
print(policy_summary)

# --------------------------------------------------
# 4. CUSTOMER SEGMENTATION
# --------------------------------------------------

df["value_segment"] = pd.qcut(
    df["annual_revenue"],
    q=3,
    labels=["Low Value", "Medium Value", "High Value"]
)

df["risk_segment"] = pd.cut(
    df["satisfaction_score"],
    bins=[0, 4, 7, 10],
    labels=["Low Satisfaction", "Medium Satisfaction", "High Satisfaction"]
)

segment_summary = df.groupby(
    ["value_segment", "risk_segment"],
    observed=True
).agg(
    customers=("customer_id", "count"),
    churn_rate=("churned", "mean"),
    annual_revenue=("annual_revenue", "sum")
).reset_index()

segment_summary["churn_rate"] = (segment_summary["churn_rate"] * 100).round(2)
segment_summary["annual_revenue"] = segment_summary["annual_revenue"].round(2)

segment_summary.to_csv("outputs/segment_summary.csv", index=False)

print("\nSEGMENT SUMMARY")
print(segment_summary)

# --------------------------------------------------
# 5. CHURN PREDICTION MODEL
# --------------------------------------------------

model_features = [
    "age",
    "income",
    "tenure_months",
    "monthly_premium",
    "claim_count",
    "customer_service_calls",
    "late_payments",
    "satisfaction_score",
    "discount_received"
]

X = df[model_features]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=8
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
prediction_probs = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, prediction_probs)

print("\nMODEL PERFORMANCE")
print("ROC AUC:", round(auc, 3))
print(classification_report(y_test, predictions))

feature_importance = pd.DataFrame({
    "feature": model_features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_importance.to_csv("outputs/feature_importance.csv", index=False)

# --------------------------------------------------
# 6. PREDICT CHURN RISK FOR ALL CUSTOMERS
# --------------------------------------------------

df["predicted_churn_probability"] = model.predict_proba(X)[:, 1]

df["churn_risk_group"] = pd.cut(
    df["predicted_churn_probability"],
    bins=[0, 0.30, 0.60, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

df.to_csv("outputs/customer_risk_scores.csv", index=False)

# --------------------------------------------------
# 7. BUSINESS SCENARIO:
# SHOULD THE COMPANY RAISE PRICES BY 8%?
# --------------------------------------------------

current_revenue = df.loc[df["churned"] == 0, "annual_revenue"].sum()

price_increase = 0.08
new_premium = df["monthly_premium"] * (1 + price_increase)

# Assumption: price increase raises churn risk
price_sensitivity_factor = 0.06
df["new_churn_probability"] = (
    df["predicted_churn_probability"] + price_sensitivity_factor
).clip(0, 1)

df["expected_revenue_after_price_increase"] = (
    new_premium * 12 * (1 - df["new_churn_probability"])
)

expected_revenue_after_price_increase = df[
    "expected_revenue_after_price_increase"
].sum()

revenue_difference = expected_revenue_after_price_increase - current_revenue

# --------------------------------------------------
# 8. RETENTION OFFER STRATEGY
# --------------------------------------------------

high_risk_high_value = df[
    (df["churn_risk_group"] == "High Risk") &
    (df["annual_revenue"] >= df["annual_revenue"].quantile(0.66))
].copy()

retention_offer_cost = 75
retention_success_rate = 0.25

saved_revenue = (
    high_risk_high_value["annual_revenue"]
    * retention_success_rate
).sum()

campaign_cost = len(high_risk_high_value) * retention_offer_cost
net_campaign_value = saved_revenue - campaign_cost

# --------------------------------------------------
# 9. VISUALIZATIONS
# --------------------------------------------------

plt.figure(figsize=(8, 5))
policy_summary.plot(
    kind="bar",
    x="policy_type",
    y="churn_rate_percent",
    legend=False
)
plt.title("Churn Rate by Policy Type")
plt.xlabel("Policy Type")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("outputs/churn_by_policy_type.png")
plt.close()

plt.figure(figsize=(8, 5))
feature_importance.plot(
    kind="bar",
    x="feature",
    y="importance",
    legend=False
)
plt.title("Top Drivers of Customer Churn")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

risk_counts = df["churn_risk_group"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
risk_counts.plot(kind="bar")
plt.title("Customers by Churn Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/customers_by_risk_group.png")
plt.close()

# --------------------------------------------------
# 10. EXECUTIVE SUMMARY
# --------------------------------------------------

summary = f"""
# Insurance Customer Retention & Pricing Analytics

## Business Question
Should the company raise monthly premiums by 8%, or would it make more sense to focus on retaining high-risk, high-value customers?

## Key Findings
- Current estimated annual revenue from active customers: ${current_revenue:,.2f}
- Expected revenue after 8% price increase: ${expected_revenue_after_price_increase:,.2f}
- Expected revenue difference: ${revenue_difference:,.2f}
- High-risk, high-value customers identified: {len(high_risk_high_value)}
- Estimated retention campaign cost: ${campaign_cost:,.2f}
- Estimated revenue saved from campaign: ${saved_revenue:,.2f}
- Net campaign value: ${net_campaign_value:,.2f}
- Churn prediction ROC AUC: {auc:.3f}

## Recommendation
The company should avoid a broad price increase if the revenue gain is small or negative after churn risk is considered.

Instead, the company should target high-risk, high-value customers with retention offers, especially customers with:
- High monthly premiums
- Low satisfaction scores
- Multiple customer service calls
- Late payment history
- Shorter tenure

## Business Impact
This project shows how analytics can move beyond charts and answer a real business decision:
Should the company raise prices, protect retention, or target specific customers?

## Tools Used
- Python
- Pandas
- SQLite
- Scikit-learn
- Matplotlib
- Business KPI analysis
- Churn prediction
- Customer segmentation
"""

with open("outputs/executive_summary.md", "w", encoding="utf-8") as file:
    file.write(summary)

conn.close()

print("\nProject complete. Check the outputs folder.")