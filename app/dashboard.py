"""
dashboard.py  (OPTIONAL BONUS MODULE)
-------------------------------------
A live Streamlit dashboard on top of the same pipeline.
Run it only after `python src/main.py` has produced outputs/loan_model.joblib.

    pip install streamlit
    streamlit run app/dashboard.py

Everything here reuses src/ - no analytics logic is duplicated.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import config                      # noqa: E402
import data_loader                 # noqa: E402
import model as model_module       # noqa: E402
import risk_engine                 # noqa: E402
from stream_simulator import RollingKPI, combine_decision   # noqa: E402

st.set_page_config(page_title="Loan Risk Control Tower", layout="wide")
st.title("Bank Loan Approval & Risk Analytics — Live Control Tower")

pipe = model_module.load()

tab_stream, tab_manual = st.tabs(["Live stream", "Score a single applicant"])

# ---------------------------------------------------------------- live stream
with tab_stream:
    delay = st.slider("Seconds between applications", 0.0, 3.0, 1.0, 0.5)
    if st.button("Start stream", type="primary"):
        stream = data_loader.get_stream_data()
        kpi = RollingKPI()
        c1, c2, c3, c4 = st.columns(4)
        m_proc, m_rate, m_risk, m_exp = (c1.empty(), c2.empty(), c3.empty(), c4.empty())
        table_slot = st.empty()
        rows = []

        for _, row in stream.iterrows():
            time.sleep(delay)
            risk = risk_engine.score_application(row)
            enriched = pd.concat([row, pd.Series(risk)])
            prob, _ = model_module.predict_one(pipe, enriched)
            decision, why = combine_decision(risk["rule_decision"], prob)
            kpi.update(decision, risk["risk_score"], risk["risk_band"], row["loan_amount"])
            snap = kpi.snapshot()

            m_proc.metric("Processed", snap["processed"])
            m_rate.metric("Approval rate", f"{snap['approval_rate']:.0%}")
            m_risk.metric("Avg risk", f"{snap['avg_risk']:.1f}")
            m_exp.metric("Exposure", f"Rs.{snap['exposure']:,}")

            rows.append({
                "ID": row["application_id"], "Name": row["applicant_name"],
                "CIBIL": row["credit_score"], "DTI": f"{row['dti_ratio']:.0%}",
                "Risk": risk["risk_score"], "Band": risk["risk_band"],
                "P(approve)": round(prob, 2), "Decision": decision, "Why": why,
            })
            table_slot.dataframe(pd.DataFrame(rows), width="stretch")

# ------------------------------------------------------------- manual scoring
with tab_manual:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 21, 70, 32)
        income = st.number_input("Annual income (Rs.)", 100000, 10000000, 900000, 50000)
        cibil = st.slider("Credit score", 300, 900, 720)
        existing = st.number_input("Existing EMI (Rs./month)", 0, 200000, 5000, 1000)
        defaults = st.slider("Previous defaults", 0, 5, 0)
    with col2:
        amount = st.number_input("Loan amount (Rs.)", 100000, 10000000, 1200000, 50000)
        term = st.selectbox("Term (months)", [24, 36, 48, 60, 72, 84, 96, 120, 144, 180, 240], 7)
        employment = st.selectbox("Employment", ["Salaried", "Self-Employed", "Unemployed"])
        purpose = st.selectbox("Purpose", ["Home", "Personal", "Vehicle", "Education", "Business"])
        area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

    if st.button("Evaluate application", type="primary"):
        raw = pd.DataFrame([{
            "application_id": "MANUAL", "applicant_name": "Manual entry",
            "age": age, "gender": "Male", "marital_status": "Single",
            "employment_type": employment, "annual_income": income,
            "credit_score": cibil, "existing_emi": existing, "loan_amount": amount,
            "loan_term_months": term, "loan_purpose": purpose,
            "property_area": area, "previous_defaults": defaults,
        }])
        row = data_loader.engineer_features(raw).iloc[0]
        risk = risk_engine.score_application(row)
        prob, _ = model_module.predict_one(pipe, pd.concat([row, pd.Series(risk)]))
        decision, why = combine_decision(risk["rule_decision"], prob)

        st.metric("Final decision", decision)
        a, b, c = st.columns(3)
        a.metric("Risk score", f"{risk['risk_score']}/100")
        b.metric("Risk band", risk["risk_band"])
        c.metric("P(approve)", f"{prob:.2f}")
        st.info(f"Justification: {why}")
        st.caption(f"Risk factors: {risk['risk_reasons']}")
