"""
Customer Retention Prioritisation and Its Distributional Consequences
=====================================================================
A pilot study on the IBM Telco Customer Churn dataset (n = 7,043).

Most churn analyses stop at predictive accuracy. This one goes one step
further and asks the question that motivates the doctoral proposal:

    When a firm allocates a limited retention budget by predicted
    customer value, who does the allocation leave out, and are those
    customers systematically different from those it serves?

Three stages:
  1. Predict churn (logistic regression, random forest, gradient boosting)
  2. Rank customers by expected revenue at risk
  3. Examine who falls below a realistic budget threshold

Data: Telco Customer Churn, IBM. Public.
Usage: python -I churn_prioritisation.py
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "churn_data", "telco.csv")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

SEED = 42
LINE = "=" * 70


def banner(m):
    print(f"\n{LINE}\n  {m}\n{LINE}")


# ══════════════════════════════════════════════════════════════════
# 1. LOAD AND PREPARE
# ══════════════════════════════════════════════════════════════════
banner("DATA")

df = pd.read_csv(DATA)
print(f"  {df.shape[0]:,} customers, {df.shape[1]} variables")

# TotalCharges is stored as text and is blank for customers with
# tenure 0. Coerce and fill with 0, which is the correct value.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

df["churn"] = (df["Churn"] == "Yes").astype(int)
print(f"  churn rate: {df['churn'].mean():.1%}  "
      f"({df['churn'].sum():,} of {len(df):,})")
print(f"  class ratio: 1 churner per {len(df) / df['churn'].sum():.1f} customers")

TARGET = "churn"
DROP = ["customerID", "Churn", "churn"]

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
# NB: pandas 3 reports string columns as StringDtype, not object, so a
# dtype == object test silently yields an empty list. Select by exclusion.
CATEGORICAL = [c for c in df.columns if c not in DROP + NUMERIC]

print(f"\n  numeric features    : {len(NUMERIC)}")
print(f"  categorical features: {len(CATEGORICAL)}")

X = df.drop(columns=DROP)
y = df[TARGET].values


# ══════════════════════════════════════════════════════════════════
# 2. PREDICT CHURN
# ══════════════════════════════════════════════════════════════════
banner("CHURN PREDICTION — 5-fold stratified cross-validation")

pre = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
])

models = {
    "Logistic Regression": Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ]),
    "Random Forest": Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED)),
    ]),
    "Gradient Boosting": Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=SEED)),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
base_rate = y.mean()
results, probs = {}, {}

for name, model in models.items():
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    probs[name] = p
    roc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)
    results[name] = (roc, pr)
    print(f"  {name:<22} ROC-AUC {roc:.4f}   PR-AUC {pr:.4f}  "
          f"({pr / base_rate:.2f}x base rate)")

best = max(results, key=lambda k: results[k][1])
print(f"\n  best by PR-AUC: {best}")
df["p_churn"] = probs[best]


# ══════════════════════════════════════════════════════════════════
# 3. EXPECTED REVENUE AT RISK
# ══════════════════════════════════════════════════════════════════
banner("RETENTION PRIORITISATION")

# A firm allocating a retention budget does not target the customers
# most likely to churn. It targets those whose departure costs most:
# probability of churn multiplied by the revenue that would be lost.
HORIZON_MONTHS = 12

df["revenue_at_risk"] = df["p_churn"] * df["MonthlyCharges"] * HORIZON_MONTHS

print(f"  expected revenue at risk over {HORIZON_MONTHS} months")
print(f"      total  £{df['revenue_at_risk'].sum():,.0f}")
print(f"      median £{df['revenue_at_risk'].median():,.0f} per customer")

# Rank and allocate. A realistic campaign reaches a fraction of the base.
BUDGETS = [0.05, 0.10, 0.20]

df["rank_value"] = df["revenue_at_risk"].rank(ascending=False, method="first")
df["rank_risk"] = df["p_churn"].rank(ascending=False, method="first")

print(f"\n  Coverage of actual churners under two allocation rules:\n")
print(f"      {'budget':>8}  {'by value':>12}  {'by risk alone':>14}")

for b in BUDGETS:
    n = int(len(df) * b)
    by_value = df.nsmallest(n, "rank_value")
    by_risk = df.nsmallest(n, "rank_risk")
    cov_v = by_value["churn"].sum() / df["churn"].sum()
    cov_r = by_risk["churn"].sum() / df["churn"].sum()
    print(f"      {b:>7.0%}  {cov_v:>11.1%}  {cov_r:>13.1%}")

print("\n  Targeting by value captures fewer churners than targeting by")
print("  risk alone. That gap is the firm's deliberate trade: it accepts")
print("  losing more customers in exchange for keeping more revenue.")


# ══════════════════════════════════════════════════════════════════
# 4. WHO IS LEFT OUT
# ══════════════════════════════════════════════════════════════════
BUDGET = 0.10
n = int(len(df) * BUDGET)
banner(f"WHO THE ALLOCATION LEAVES OUT  (budget = top {BUDGET:.0%} by value)")

df["prioritised"] = (df["rank_value"] <= n).astype(int)

# The group that matters: customers the model says are likely to leave,
# who are not selected because their revenue is too small to justify it.
at_risk = df["p_churn"] > df["p_churn"].quantile(0.75)
df["at_risk_not_served"] = (at_risk & (df["prioritised"] == 0)).astype(int)

n_served = int(df["prioritised"].sum())
n_abandoned = int(df["at_risk_not_served"].sum())

print(f"  customers served    : {n_served:,}")
print(f"  high-risk, not served: {n_abandoned:,}  "
      f"({n_abandoned / at_risk.sum():.1%} of all high-risk customers)")

print("\n  Comparison of high-risk customers, served versus not served:\n")

served_hr = df[at_risk & (df["prioritised"] == 1)]
unserved_hr = df[at_risk & (df["prioritised"] == 0)]

print(f"      {'characteristic':<28} {'served':>10} {'not served':>12}")
print(f"      {'-' * 28} {'-' * 10} {'-' * 12}")

for label, col in [
    ("Monthly charges (£)", "MonthlyCharges"),
    ("Tenure (months)", "tenure"),
    ("Actual churn rate", "churn"),
    ("Predicted churn prob.", "p_churn"),
]:
    s, u = served_hr[col].mean(), unserved_hr[col].mean()
    fmt = ".1%" if col in ("churn", "p_churn") else ".1f"
    print(f"      {label:<28} {s:>10{fmt}} {u:>12{fmt}}")

print(f"\n      {'Senior citizens':<28} "
      f"{served_hr['SeniorCitizen'].mean():>10.1%} "
      f"{unserved_hr['SeniorCitizen'].mean():>12.1%}")

for col in ["Contract", "PaymentMethod", "InternetService"]:
    print(f"\n      {col}:")
    s = served_hr[col].value_counts(normalize=True)
    u = unserved_hr[col].value_counts(normalize=True)
    for level in sorted(set(s.index) | set(u.index)):
        print(f"          {level:<24} {s.get(level, 0):>10.1%} "
              f"{u.get(level, 0):>12.1%}")


# ══════════════════════════════════════════════════════════════════
# 5. THE FINDING
# ══════════════════════════════════════════════════════════════════
banner("INTERPRETATION")

churn_served = served_hr["churn"].mean()
churn_unserved = unserved_hr["churn"].mean()
spend_ratio = served_hr["MonthlyCharges"].mean() / unserved_hr["MonthlyCharges"].mean()

print(f"""
  Among customers the model flags as high churn risk, those excluded
  from the retention budget spend {1 / spend_ratio:.0%} as much per month as those
  included, and actually churn at {churn_unserved:.1%} against {churn_served:.1%}.

  The exclusion is not a prediction failure. The model identified
  these customers correctly. They were deprioritised because the
  revenue they represent did not justify the cost of keeping them.

  This is customer prioritisation working exactly as intended, and it
  is the point at which the doctoral question arises: what happens to
  these customers next, and does their deprioritisation feed back into
  the behaviour that produced it.

  This pilot cannot answer that. It has no post-allocation outcome
  data, the allocation is simulated rather than observed, and a single
  cross-section cannot identify a feedback loop. What it establishes
  is that the group exists, is large, and is systematically different.
""")

summary = df[["customerID", "p_churn", "MonthlyCharges", "tenure",
              "revenue_at_risk", "prioritised", "at_risk_not_served",
              "churn"]].copy()
summary.to_csv(os.path.join(OUT, "customer_prioritisation.csv"), index=False)

pd.DataFrame([
    {"model": k, "roc_auc": v[0], "pr_auc": v[1]} for k, v in results.items()
]).to_csv(os.path.join(OUT, "model_performance.csv"), index=False)

print(f"  written: {OUT}/customer_prioritisation.csv")
print(f"  written: {OUT}/model_performance.csv")
print(f"\n{LINE}\n  COMPLETE\n{LINE}\n")
