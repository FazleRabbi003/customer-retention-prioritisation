"""
Figure for the retention prioritisation pilot.

Panel A shows the mechanism: the budget boundary is a revenue curve, so a
whole region of high-risk, low-spend customers sits permanently below it.
Panel B shows the consequence: who those excluded customers are.

Usage: python -I make_figure.py   (run churn_prioritisation.py first)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "churn_data", "telco.csv")
OUT = os.path.join(HERE, "outputs")

# Validated categorical pair: CVD separation dE 28.0 (protan), 33.3 normal.
SERVED = "#3B6FD4"
EXCLUDED = "#D97706"
INK = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#E3E3E3"

HORIZON = 12
BUDGET = 0.10

scored = pd.read_csv(os.path.join(OUT, "customer_prioritisation.csv"))
raw = pd.read_csv(DATA)[["customerID", "InternetService", "PaymentMethod"]]
df = scored.merge(raw, on="customerID", how="left")

n_served = int(df["prioritised"].sum())
threshold = df.nlargest(n_served, "revenue_at_risk")["revenue_at_risk"].min()

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), width_ratios=[1.25, 1])

# ── Panel A: the boundary ────────────────────────────────────────
served = df[df["prioritised"] == 1]
excluded = df[df["prioritised"] == 0]

ax1.scatter(excluded["p_churn"], excluded["MonthlyCharges"],
            s=7, c=EXCLUDED, alpha=0.30, linewidths=0, label="Not served")
ax1.scatter(served["p_churn"], served["MonthlyCharges"],
            s=7, c=SERVED, alpha=0.45, linewidths=0, label="Retention budget")

# The allocation boundary: p x charges x horizon = threshold
p = np.linspace(0.02, 1.0, 400)
boundary = threshold / (HORIZON * p)
ax1.plot(p, boundary, color=INK, lw=1.6, ls="--", zorder=5)

ax1.annotate("budget boundary", xy=(0.93, threshold / (HORIZON * 0.93)),
             xytext=(0.60, 44), fontsize=9, color=INK,
             arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))

ax1.set_xlim(0, 1.0)
ax1.set_ylim(15, 125)
ax1.set_xlabel("Predicted probability of churn")
ax1.set_ylabel("Monthly charges (£)")
ax1.set_title("Exclusion is set by revenue, not by risk",
              fontsize=11.5, fontweight="bold", loc="left", pad=12)
ax1.grid(axis="both", color=GRID, lw=0.7)
ax1.set_axisbelow(True)

ax1.text(0.60, 92, "served", fontsize=9.5, color=SERVED, fontweight="bold")

# ── Panel B: who is excluded ─────────────────────────────────────
at_risk = df["p_churn"] > df["p_churn"].quantile(0.75)
hr_served = df[at_risk & (df["prioritised"] == 1)]
hr_excluded = df[at_risk & (df["prioritised"] == 0)]

metrics = [
    ("DSL internet", (hr_served["InternetService"] == "DSL").mean(),
     (hr_excluded["InternetService"] == "DSL").mean()),
    ("Pays by mailed cheque", (hr_served["PaymentMethod"] == "Mailed check").mean(),
     (hr_excluded["PaymentMethod"] == "Mailed check").mean()),
    ("Actually churned", hr_served["churn"].mean(), hr_excluded["churn"].mean()),
]

y = np.arange(len(metrics))
h = 0.34

for i, (label, s_val, e_val) in enumerate(metrics):
    ax2.barh(i + h / 2, s_val, height=h, color=SERVED, zorder=3)
    ax2.barh(i - h / 2, e_val, height=h, color=EXCLUDED, zorder=3)
    ax2.text(s_val + 0.012, i + h / 2, f"{s_val:.0%}",
             va="center", fontsize=9, color=INK)
    ax2.text(e_val + 0.012, i - h / 2, f"{e_val:.0%}",
             va="center", fontsize=9, color=INK)

ax2.set_yticks(y)
ax2.set_yticklabels([m[0] for m in metrics], fontsize=9.5)
ax2.set_xlim(0, 0.92)
ax2.set_xticks([0, 0.25, 0.5, 0.75])
ax2.set_xticklabels(["0%", "25%", "50%", "75%"])
ax2.set_title("Among high-risk customers, who is left out",
              fontsize=11.5, fontweight="bold", loc="left", pad=12)
ax2.grid(axis="x", color=GRID, lw=0.7)
ax2.set_axisbelow(True)
ax2.invert_yaxis()

fig.legend(handles=[
    Line2D([], [], marker="o", ls="", ms=7, color=SERVED, label="Receives retention budget"),
    Line2D([], [], marker="o", ls="", ms=7, color=EXCLUDED, label="Not served"),
], loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=2,
   frameon=False, fontsize=9.5)

fig.text(0.5, -0.13,
         f"IBM Telco Customer Churn, n = 7,043. Retention budget = top "
         f"{BUDGET:.0%} by expected revenue at risk over {HORIZON} months. "
         f"High risk = top quartile of predicted churn probability.",
         ha="center", fontsize=8.5, color=MUTED)

plt.tight_layout()
path = os.path.join(OUT, "exclusion.png")
plt.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
print("written", path)

print(f"\n  DSL among served high-risk   : {(hr_served['InternetService'] == 'DSL').mean():.1%}")
print(f"  DSL among excluded high-risk : {(hr_excluded['InternetService'] == 'DSL').mean():.1%}")
print(f"  budget threshold             : £{threshold:,.0f} revenue at risk")
