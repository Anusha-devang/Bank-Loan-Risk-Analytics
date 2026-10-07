"""
data_loader.py
--------------
Stage 1 of the pipeline: read the CSV, validate it, clean it and derive the
extra columns that both the rule engine and the ML model depend on.
"""

import pandas as pd

import config


def load_csv(path):
    """Read a CSV file into a DataFrame and fail loudly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    df = pd.read_csv(path)
    print(f"[data_loader] Loaded {len(df)} rows x {len(df.columns)} columns from {path.name}")
    return df


def validate(df, expect_target=True):
    """Basic data-quality gate. Returns a dict describing what was found."""
    report = {
        "rows": len(df),
        "duplicate_ids": int(df["application_id"].duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "negative_income": int((df["annual_income"] <= 0).sum()),
    }

    if expect_target and config.TARGET not in df.columns:
        raise ValueError(f"Target column '{config.TARGET}' is missing")

    if report["duplicate_ids"]:
        df = df.drop_duplicates(subset="application_id", keep="first")

    # Numeric columns: fill any gap with the column median (robust to outliers)
    for col in config.NUMERIC_RAW:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns: fill gaps with the most frequent value
    for col in config.CATEGORICAL:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"[data_loader] Validation -> {report}")
    return df, report


def _emi(principal, annual_rate, months):
    """
    Standard reducing-balance EMI formula.

        EMI = P * r * (1+r)^n / ((1+r)^n - 1)

    P = principal, r = monthly interest rate, n = number of months.
    """
    r = annual_rate / 12
    growth = (1 + r) ** months
    return principal * r * growth / (growth - 1)


def engineer_features(df):
    """Create the derived columns listed in config.NUMERIC_ENGINEERED."""
    df = df.copy()

    # Lenders think in months, the CSV stores yearly income.
    df["monthly_income"] = df["annual_income"] / 12

    # EMI the applicant would pay if this loan is sanctioned.
    df["estimated_emi"] = _emi(
        df["loan_amount"], config.ANNUAL_INTEREST_RATE, df["loan_term_months"]
    ).round(0)

    # Total monthly obligation = new EMI + EMIs already running.
    df["total_emi"] = df["estimated_emi"] + df["existing_emi"]

    # Debt-to-Income: the single most important affordability metric.
    df["dti_ratio"] = (df["total_emi"] / df["monthly_income"]).round(3)

    # Loan-to-Income: how many years of gross salary the loan represents.
    df["loan_to_income"] = (df["loan_amount"] / df["annual_income"]).round(3)

    print(f"[data_loader] Engineered {len(config.NUMERIC_ENGINEERED)} new features")
    return df


def get_training_data():
    """Convenience wrapper used by main.py -> returns a model-ready DataFrame."""
    df = load_csv(config.TRAINING_CSV)
    df, _ = validate(df, expect_target=True)
    return engineer_features(df)


def get_stream_data():
    """Same as above but for the unlabelled incoming applications."""
    df = load_csv(config.STREAM_CSV)
    df, _ = validate(df, expect_target=False)
    return engineer_features(df)


if __name__ == "__main__":
    data = get_training_data()
    print(data[["application_id", "dti_ratio", "loan_to_income", "estimated_emi"]].head())
