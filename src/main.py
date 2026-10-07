"""
main.py
-------
Entry point. Runs the five pipeline stages end to end:

    ingest -> risk scoring -> EDA -> model training -> real-time stream

Usage:
    python main.py                # full run, 1 second between stream events
    python main.py --fast         # no delay, useful while debugging
    python main.py --no-charts    # skip matplotlib, console output only
"""

import argparse
from datetime import datetime

import config
import data_loader
import eda
import model as model_module
import risk_engine
import stream_simulator


def banner(step, title):
    print("\n" + "#" * 68)
    print(f"# STEP {step}: {title}")
    print("#" * 68)


def write_summary(metrics, drivers, stream_kpi, scored):
    """Persist a plain-text report so results survive after the console clears."""
    lines = [
        "BANK LOAN APPROVAL & RISK ANALYTICS - RUN SUMMARY",
        f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "1. DATASET",
        f"   Historical records : {len(scored)}",
        f"   Approval rate      : {(scored[config.TARGET] == 'Y').mean():.1%}",
        f"   Risk band split    : {scored['risk_band'].value_counts().to_dict()}",
        "",
        "2. MODEL",
        f"   Train / test rows  : {metrics['train_rows']} / {metrics['test_rows']}",
        f"   Accuracy           : {metrics['accuracy']}",
        f"   ROC-AUC            : {metrics['roc_auc']}",
        f"   Confusion matrix   : {metrics['confusion_matrix']}",
        "",
        "   Top drivers:",
    ]
    for _, r in drivers.iterrows():
        lines.append(f"     {r['feature']:<40} {r['coefficient']:>8}")

    lines += [
        "",
        "3. REAL-TIME STREAM",
        f"   Applications processed : {stream_kpi['processed']}",
        f"   Approval rate          : {stream_kpi['approval_rate']:.1%}",
        f"   Average risk score     : {stream_kpi['avg_risk']:.1f}/100",
        f"   High-risk applications : {stream_kpi['high_risk']}",
        f"   Sanctioned exposure    : Rs.{stream_kpi['exposure']:,}",
        "",
        "Artefacts written to outputs/: charts (*.png), loan_model.joblib,",
        "realtime_decisions.csv, summary_report.txt",
    ]

    config.SUMMARY_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[main] summary report written to {config.SUMMARY_REPORT.name}")


def main():
    parser = argparse.ArgumentParser(description="Bank Loan Approval & Risk Analytics")
    parser.add_argument("--fast", action="store_true", help="no delay in the stream")
    parser.add_argument("--no-charts", action="store_true", help="skip PNG generation")
    args = parser.parse_args()

    print("\nBANK LOAN APPROVAL & RISK ANALYTICS")
    print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S}")

    # ---- STEP 1: ingest + feature engineering ------------------------------
    banner(1, "DATA INGESTION & FEATURE ENGINEERING")
    df = data_loader.get_training_data()

    # ---- STEP 2: rule-based risk scoring -----------------------------------
    banner(2, "RULE-BASED RISK SCORING")
    scored = risk_engine.score_dataframe(df)
    print(scored[["application_id", "credit_score", "dti_ratio",
                  "risk_score", "risk_band", "rule_decision",
                  config.TARGET]].to_string(index=False))

    # ---- STEP 3: EDA --------------------------------------------------------
    banner(3, "EXPLORATORY DATA ANALYSIS")
    if args.no_charts:
        eda.print_summary(scored)
    else:
        eda.run(scored)

    # ---- STEP 4: model training --------------------------------------------
    banner(4, "MODEL TRAINING & EVALUATION")
    pipe, metrics = model_module.train(scored)
    drivers = model_module.top_drivers(pipe)

    # ---- STEP 5: real-time stream ------------------------------------------
    banner(5, "REAL-TIME APPLICATION STREAM")
    _, stream_kpi = stream_simulator.run(pipe, delay=0 if args.fast else None)

    write_summary(metrics, drivers, stream_kpi, scored)
    print("\nPipeline finished successfully.\n")


if __name__ == "__main__":
    main()
