"""
risk_engine.py
--------------
Stage 2: a transparent, rule-based credit risk score.

Why a rule engine when we also train an ML model?
Banks are regulated. A model can output "reject" but the officer must still be
able to tell the customer *why*. The rule engine gives that explanation, and it
also works on day one, before enough history exists to train anything.

Score = 0 (perfect borrower) ... 100 (worst borrower). Penalties are additive.
"""

import config


def _band_penalty(value, bands, higher_is_better):
    """
    Walk a list of (threshold, penalty) tuples and return the first match.

    higher_is_better=True  -> bands are (minimum, penalty), e.g. credit score
    higher_is_better=False -> bands are (maximum, penalty), e.g. DTI ratio
    """
    for threshold, penalty in bands:
        if higher_is_better and value >= threshold:
            return penalty
        if not higher_is_better and value <= threshold:
            return penalty
    return bands[-1][1]


def score_application(row):
    """
    Score one application (a pandas Series or a plain dict).

    Returns a dict with the total score, the risk band, the suggested decision
    and a human-readable breakdown of every penalty that was applied.
    """
    reasons = []
    score = 0

    # --- 1. Credit history -------------------------------------------------
    p = _band_penalty(row["credit_score"], config.CREDIT_SCORE_BANDS, True)
    if p:
        reasons.append(f"credit score {int(row['credit_score'])} (+{p})")
    score += p

    # --- 2. Affordability (debt-to-income) ---------------------------------
    p = _band_penalty(row["dti_ratio"], config.DTI_BANDS, False)
    if p:
        reasons.append(f"DTI {row['dti_ratio']:.0%} (+{p})")
    score += p

    # --- 3. Loan size relative to income -----------------------------------
    p = _band_penalty(row["loan_to_income"], config.LTI_BANDS, False)
    if p:
        reasons.append(f"loan is {row['loan_to_income']:.1f}x income (+{p})")
    score += p

    # --- 4. Past behaviour --------------------------------------------------
    defaults = int(row["previous_defaults"])
    if defaults:
        p = defaults * config.DEFAULT_PENALTY
        reasons.append(f"{defaults} previous default(s) (+{p})")
        score += p

    # --- 5. Income stability ------------------------------------------------
    employment = row["employment_type"]
    if employment == "Unemployed":
        reasons.append(f"unemployed (+{config.UNEMPLOYED_PENALTY})")
        score += config.UNEMPLOYED_PENALTY
    elif employment == "Self-Employed":
        reasons.append(f"self-employed income (+{config.SELF_EMPLOYED_PENALTY})")
        score += config.SELF_EMPLOYED_PENALTY

    score = min(score, 100)          # cap so the scale stays 0-100

    # --- 6. Translate the number into a band + action -----------------------
    band, action = config.RISK_BANDS[-1][1], config.RISK_BANDS[-1][2]
    for limit, label, hint in config.RISK_BANDS:
        if score <= limit:
            band, action = label, hint
            break

    return {
        "risk_score": score,
        "risk_band": band,
        "rule_decision": action,
        "risk_reasons": "; ".join(reasons) if reasons else "no risk flags",
    }


def score_dataframe(df):
    """Vectorised-by-loop version: adds four risk columns to a whole DataFrame."""
    scored = df.apply(score_application, axis=1, result_type="expand")
    out = df.join(scored)
    counts = out["risk_band"].value_counts().to_dict()
    print(f"[risk_engine] Scored {len(out)} applications -> {counts}")
    return out


if __name__ == "__main__":
    import data_loader

    demo = score_dataframe(data_loader.get_training_data())
    print(demo[["application_id", "risk_score", "risk_band", "rule_decision"]].head())
