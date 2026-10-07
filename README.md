# Bank Loan Approval & Risk Analytics

A real-time data analytics project. Historical loan applications are analysed,
a transparent rule-based risk engine and a Logistic Regression model are built
on top of them, and new applications are then scored one by one as they
"arrive" — exactly the way a bank's underwriting queue works.

Read `Bank_Loan_Approval_Risk_Analytics.pdf` first. It contains the full
requirement specification, the architecture, and a line-by-line explanation of
every file in `src/`.

---

## 1. Quick start (3 commands)

```bash
pip install -r requirements.txt
cd src
python main.py
```

That is the whole project. It runs all five stages and writes every artefact
into `outputs/`.

Useful flags:

| Command | What it does |
|---|---|
| `python main.py` | Full run, 1 second pause between streamed applications |
| `python main.py --fast` | Same, but no pause (good while debugging) |
| `python main.py --no-charts` | Console output only, skips matplotlib |

Each module also runs standalone for testing:

```bash
python data_loader.py      # check ingestion + feature engineering
python risk_engine.py      # check the scoring rules
python eda.py              # regenerate charts only
python model.py            # retrain and print coefficients
python stream_simulator.py # replay the live stream (needs a trained model)
```

---

## 2. Folder structure

```
BankLoanRiskAnalytics/
├── README.md
├── requirements.txt
├── Bank_Loan_Approval_Risk_Analytics.pdf   <- the handbook, start here
├── data/
│   ├── loan_applications.csv      20 labelled historical applications
│   └── incoming_stream.csv         8 unlabelled applications for the live feed
├── src/
│   ├── config.py            all paths, thresholds and business rules
│   ├── data_loader.py       ingestion, validation, feature engineering
│   ├── risk_engine.py       explainable rule-based risk score (0-100)
│   ├── eda.py               console statistics + 4 matplotlib charts
│   ├── model.py             scikit-learn pipeline, training, evaluation
│   ├── stream_simulator.py  real-time decision loop with rolling KPIs
│   └── main.py              orchestrator (run this)
├── app/
│   └── dashboard.py         OPTIONAL Streamlit dashboard
├── sample_outputs/          reference results so you can check your run
└── outputs/                 empty until you run the project
```

---

## 3. What gets generated in `outputs/`

| File | Contents |
|---|---|
| `01_approval_distribution.png` | Approved vs rejected counts |
| `02_credit_score_boxplot.png` | Credit score spread per outcome |
| `03_income_vs_loan.png` | Income vs loan size, bubble = risk score |
| `04_risk_bands.png` | Low / Medium / High band counts |
| `loan_model.joblib` | The trained scikit-learn pipeline |
| `realtime_decisions.csv` | One row per streamed application with its decision |
| `summary_report.txt` | Plain-text run summary |

---

## 3b. Checking your run

`sample_outputs/` contains the exact files a correct run produces, plus
`console_output.txt` with the full reference console log. Compare your
`outputs/` against it — only the timestamps should differ.

---

## 4. Optional dashboard

```bash
pip install streamlit
python src/main.py          # must run once so the model file exists
streamlit run app/dashboard.py
```

---

## 5. Requirements

Python 3.9 or newer. Dependencies are in `requirements.txt`
(pandas, numpy, scikit-learn, matplotlib, joblib). No internet access,
no database and no API key is required.
