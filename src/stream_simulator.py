"""
stream_simulator.py
-------------------
Stage 5: the "real-time" part of the project.

A real bank receives applications one at a time from a queue (Kafka, RabbitMQ,
an API endpoint). We reproduce that behaviour by reading incoming_stream.csv
row by row with a delay, and for every arriving record we:

    1. engineer the same features used at training time
    2. run the rule engine  -> explainable risk score + band
    3. run the ML model     -> approval probability
    4. combine both into a final decision
    5. update rolling KPIs and append the result to a decision log

Because the loop is event driven, swapping the CSV reader for a real Kafka
consumer would not change any other line of this file.
"""

import time
from datetime import datetime

import pandas as pd

import config
import data_loader
import model as model_module
import risk_engine


def combine_decision(rule_action, probability):
    """
    Champion/challenger logic used by most retail banks.

    * The rule engine has veto power on the extremes (regulatory safety net).
    * In the grey zone the model's probability decides.
    """
    if rule_action == "AUTO-REJECT":
        return "REJECTED", "rule veto: high risk band"
    if rule_action == "AUTO-APPROVE" and probability >= 0.40:
        return "APPROVED", "low risk + model agrees"
    if probability >= config.APPROVAL_THRESHOLD:
        return "APPROVED", f"model probability {probability:.2f}"
    return "REJECTED", f"model probability {probability:.2f}"


class RollingKPI:
    """Keeps the live counters that a monitoring dashboard would show."""

    def __init__(self):
        self.total = 0
        self.approved = 0
        self.risk_sum = 0
        self.high_risk = 0
        self.exposure = 0          # rupees sanctioned

    def update(self, decision, risk_score, band, loan_amount):
        self.total += 1
        self.risk_sum += risk_score
        if decision == "APPROVED":
            self.approved += 1
            self.exposure += loan_amount
        if band == "HIGH":
            self.high_risk += 1

    def snapshot(self):
        return {
            "processed": self.total,
            "approval_rate": self.approved / self.total if self.total else 0,
            "avg_risk": self.risk_sum / self.total if self.total else 0,
            "high_risk": self.high_risk,
            "exposure": self.exposure,
        }


def run(pipe=None, delay=None):
    """Process the whole incoming stream and return the decision log."""
    pipe = pipe or model_module.load()
    delay = config.STREAM_DELAY_SECONDS if delay is None else delay

    stream = data_loader.get_stream_data()
    kpi = RollingKPI()
    log = []

    print("\n" + "=" * 68)
    print("REAL-TIME DECISION STREAM")
    print("=" * 68)

    for _, row in stream.iterrows():
        time.sleep(delay)                                   # simulate arrival gap
        arrived = datetime.now().strftime("%H:%M:%S")

        risk = risk_engine.score_application(row)           # explainable layer
        enriched = pd.concat([row, pd.Series(risk)])
        probability, _ = model_module.predict_one(pipe, enriched)   # statistical layer
        decision, why = combine_decision(risk["rule_decision"], probability)

        kpi.update(decision, risk["risk_score"], risk["risk_band"], row["loan_amount"])
        snap = kpi.snapshot()

        print(f"\n[{arrived}] {row['application_id']} | {row['applicant_name']}")
        print(f"    income Rs.{row['annual_income']:,}/yr | loan Rs.{row['loan_amount']:,}"
              f" | CIBIL {row['credit_score']} | DTI {row['dti_ratio']:.0%}")
        print(f"    risk {risk['risk_score']}/100 ({risk['risk_band']})"
              f" | model P(approve)={probability:.2f}")
        print(f"    DECISION: {decision}  <- {why}")
        print(f"    reasons : {risk['risk_reasons']}")
        print(f"    LIVE KPI: processed={snap['processed']}"
              f" approval={snap['approval_rate']:.0%}"
              f" avg-risk={snap['avg_risk']:.1f}"
              f" high-risk={snap['high_risk']}"
              f" exposure=Rs.{snap['exposure']:,}")

        log.append({
            "timestamp": arrived,
            "application_id": row["application_id"],
            "applicant_name": row["applicant_name"],
            "credit_score": row["credit_score"],
            "dti_ratio": row["dti_ratio"],
            "risk_score": risk["risk_score"],
            "risk_band": risk["risk_band"],
            "model_probability": round(probability, 3),
            "final_decision": decision,
            "justification": why,
            "risk_reasons": risk["risk_reasons"],
        })

    log_df = pd.DataFrame(log)
    log_df.to_csv(config.DECISION_LOG, index=False)
    print(f"\n[stream] decision log written to {config.DECISION_LOG.name}")

    final = kpi.snapshot()
    print("\n--- SESSION SUMMARY ---")
    print(f"Applications processed : {final['processed']}")
    print(f"Approval rate          : {final['approval_rate']:.1%}")
    print(f"Average risk score     : {final['avg_risk']:.1f}/100")
    print(f"High-risk applications : {final['high_risk']}")
    print(f"Sanctioned exposure    : Rs.{final['exposure']:,}")

    return log_df, final


if __name__ == "__main__":
    run()
