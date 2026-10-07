"""
eda.py
------
Stage 3: exploratory data analysis. Prints the numbers a credit manager would
ask for and saves four PNG charts into outputs/.

matplotlib only (no seaborn) so the project runs with the smallest possible
dependency list.
"""

import matplotlib

matplotlib.use("Agg")            # headless backend: works on servers / Colab
import matplotlib.pyplot as plt

import config

PALETTE = {"approve": "#1b7f4d", "reject": "#b3261e", "accent": "#1f4e79", "grey": "#8a8a8a"}


def print_summary(df):
    """Console EDA: shape, approval rate, group averages."""
    print("\n" + "=" * 68)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 68)

    print(f"Records            : {len(df)}")
    print(f"Columns            : {len(df.columns)}")

    approved = (df[config.TARGET] == "Y").sum()
    print(f"Approved / Rejected: {approved} / {len(df) - approved}")
    print(f"Approval rate      : {approved / len(df):.1%}")

    print("\n-- Averages by approval status --")
    grouped = df.groupby(config.TARGET)[
        ["credit_score", "annual_income", "dti_ratio", "loan_to_income", "previous_defaults"]
    ].mean().round(2)
    print(grouped.to_string())

    print("\n-- Approval rate by employment type --")
    emp = df.groupby("employment_type")[config.TARGET].apply(
        lambda s: (s == "Y").mean()
    ).sort_values(ascending=False)
    for name, rate in emp.items():
        print(f"  {name:<15} {rate:.0%}")

    print("\n-- Correlation of numeric drivers with approval --")
    numeric = df[["credit_score", "annual_income", "dti_ratio", "loan_to_income",
                  "previous_defaults", "risk_score"]].copy()
    numeric["approved_flag"] = (df[config.TARGET] == "Y").astype(int)
    corr = numeric.corr(numeric_only=True)["approved_flag"].drop("approved_flag").round(3)
    print(corr.sort_values(ascending=False).to_string())


def _save(fig, filename):
    path = config.OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[eda] saved {path.name}")


def build_charts(df):
    """Four charts that together tell the whole data story."""

    # 1. Approval split -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[config.TARGET].value_counts()
    ax.bar(["Approved" if k == "Y" else "Rejected" for k in counts.index], counts.values,
           color=[PALETTE["approve"] if k == "Y" else PALETTE["reject"] for k in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.15, str(v), ha="center", fontweight="bold")
    ax.set_title("Loan approval distribution")
    ax.set_ylabel("Applications")
    _save(fig, "01_approval_distribution.png")

    # 2. Credit score vs outcome ---------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [df.loc[df[config.TARGET] == "N", "credit_score"],
            df.loc[df[config.TARGET] == "Y", "credit_score"]]
    bp = ax.boxplot(data, tick_labels=["Rejected", "Approved"], patch_artist=True)
    for patch, colour in zip(bp["boxes"], [PALETTE["reject"], PALETTE["approve"]]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.55)
    ax.set_title("Credit score by outcome")
    ax.set_ylabel("CIBIL score")
    _save(fig, "02_credit_score_boxplot.png")

    # 3. Income vs loan amount ------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label, colour, name in [("Y", PALETTE["approve"], "Approved"),
                                ("N", PALETTE["reject"], "Rejected")]:
        sub = df[df[config.TARGET] == label]
        ax.scatter(sub["annual_income"] / 1e5, sub["loan_amount"] / 1e5,
                   s=sub["risk_score"] * 4 + 20, c=colour, alpha=0.65,
                   edgecolors="white", label=name)
    ax.set_xlabel("Annual income (lakh INR)")
    ax.set_ylabel("Loan amount (lakh INR)")
    ax.set_title("Income vs loan size (bubble = risk score)")
    ax.legend()
    _save(fig, "03_income_vs_loan.png")

    # 4. Risk band distribution ----------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    order = ["LOW", "MEDIUM", "HIGH"]
    values = [int((df["risk_band"] == b).sum()) for b in order]
    ax.bar(order, values, color=[PALETTE["approve"], "#c8860d", PALETTE["reject"]])
    for i, v in enumerate(values):
        ax.text(i, v + 0.15, str(v), ha="center", fontweight="bold")
    ax.set_title("Applications per risk band (rule engine)")
    ax.set_ylabel("Applications")
    _save(fig, "04_risk_bands.png")


def run(df):
    print_summary(df)
    build_charts(df)


if __name__ == "__main__":
    import data_loader
    import risk_engine

    run(risk_engine.score_dataframe(data_loader.get_training_data()))
