"""
config.py
---------
Single place for every path, constant and business rule threshold used by the
project. Nothing else in the codebase should hard-code a number or a file path.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# 1. FOLDER LAYOUT
# ----------------------------------------------------------------------------
# __file__ -> .../BankLoanRiskAnalytics/src/config.py
# .parent  -> .../src        .parent.parent -> project root
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

TRAINING_CSV = DATA_DIR / "loan_applications.csv"
STREAM_CSV = DATA_DIR / "incoming_stream.csv"

MODEL_FILE = OUTPUT_DIR / "loan_model.joblib"
DECISION_LOG = OUTPUT_DIR / "realtime_decisions.csv"
SUMMARY_REPORT = OUTPUT_DIR / "summary_report.txt"

# Create the outputs folder on import so no module has to worry about it.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 2. COLUMN GROUPS
# ----------------------------------------------------------------------------
TARGET = "loan_approved"

# Raw numeric columns coming straight from the CSV
NUMERIC_RAW = [
    "age",
    "annual_income",
    "credit_score",
    "existing_emi",
    "loan_amount",
    "loan_term_months",
    "previous_defaults",
]

# Columns created by feature engineering (see data_loader.engineer_features)
NUMERIC_ENGINEERED = [
    "monthly_income",
    "estimated_emi",
    "total_emi",
    "dti_ratio",
    "loan_to_income",
]

CATEGORICAL = [
    "gender",
    "marital_status",
    "employment_type",
    "loan_purpose",
    "property_area",
]

FEATURES = NUMERIC_RAW + NUMERIC_ENGINEERED + CATEGORICAL

# ----------------------------------------------------------------------------
# 3. BUSINESS RULES (used by risk_engine.py)
# ----------------------------------------------------------------------------
ANNUAL_INTEREST_RATE = 0.105          # 10.5 % p.a. flat assumption for EMI maths

CREDIT_SCORE_BANDS = [                # (minimum score, penalty points)
    (750, 0),
    (700, 8),
    (650, 18),
    (600, 30),
    (0, 45),
]

DTI_BANDS = [                         # (maximum DTI, penalty points)
    (0.30, 0),
    (0.40, 10),
    (0.50, 22),
    (1.00, 35),
]

LTI_BANDS = [                         # (maximum loan-to-income, penalty points)
    (1.5, 0),
    (2.5, 6),
    (3.5, 12),
    (99.0, 20),
]

DEFAULT_PENALTY = 12                  # points per previous default
UNEMPLOYED_PENALTY = 15
SELF_EMPLOYED_PENALTY = 5

RISK_BANDS = [                        # (maximum score, label, decision hint)
    (25, "LOW", "AUTO-APPROVE"),
    (45, "MEDIUM", "MANUAL REVIEW"),
    (100, "HIGH", "AUTO-REJECT"),
]

# ----------------------------------------------------------------------------
# 4. MODEL / STREAM SETTINGS
# ----------------------------------------------------------------------------
TEST_SIZE = 0.30
RANDOM_STATE = 42
APPROVAL_THRESHOLD = 0.50             # probability above which the model says "Y"
STREAM_DELAY_SECONDS = 1.0            # pause between two simulated applications
