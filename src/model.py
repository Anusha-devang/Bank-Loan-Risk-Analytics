"""
model.py
--------
Stage 4: train a Logistic Regression classifier that predicts whether a loan
will be approved, and expose a predict() helper the real-time engine can call.

Logistic Regression is chosen deliberately:
  * it outputs a calibrated probability, not just a label
  * its coefficients are readable, which matters for credit regulation
  * it is stable on very small datasets, unlike deep trees
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def build_pipeline():
    """
    A Pipeline bundles preprocessing + estimator into one object, so the exact
    same transformations are applied at training time and at prediction time.
    This is what prevents 'training/serving skew'.
    """
    numeric = config.NUMERIC_RAW + config.NUMERIC_ENGINEERED

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000,
                                              class_weight="balanced",
                                              random_state=config.RANDOM_STATE)),
        ]
    )


def train(df):
    """Split, fit, evaluate and persist. Returns (pipeline, metrics dict)."""
    X = df[config.FEATURES]
    y = (df[config.TARGET] == "Y").astype(int)      # Y/N -> 1/0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,                                  # keep class balance in both sets
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 3) if y_test.nunique() > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\n" + "=" * 68)
    print("MODEL TRAINING")
    print("=" * 68)
    print(f"Train rows : {metrics['train_rows']}   Test rows : {metrics['test_rows']}")
    print(f"Accuracy   : {metrics['accuracy']}")
    print(f"ROC-AUC    : {metrics['roc_auc']}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"],
                                zero_division=0))

    joblib.dump(pipe, config.MODEL_FILE)
    print(f"[model] pipeline saved to {config.MODEL_FILE.name}")

    return pipe, metrics


def top_drivers(pipe, n=8):
    """Read the fitted coefficients back out and rank them by absolute weight."""
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    coefs = pipe.named_steps["classifier"].coef_[0]
    table = (pd.DataFrame({"feature": names, "coefficient": coefs.round(3)})
             .assign(strength=lambda d: d["coefficient"].abs())
             .sort_values("strength", ascending=False)
             .head(n)
             .drop(columns="strength")
             .reset_index(drop=True))
    print("\nTop model drivers (positive = pushes towards approval):")
    print(table.to_string(index=False))
    return table


def load():
    """Load the persisted pipeline; train first if it does not exist yet."""
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError("Model not trained yet. Run main.py first.")
    return joblib.load(config.MODEL_FILE)


def predict_one(pipe, row):
    """
    Score a single application.
    `row` is a pandas Series; we wrap it in a one-row DataFrame because
    scikit-learn always expects 2-D input.
    """
    frame = row.to_frame().T[config.FEATURES]
    probability = float(pipe.predict_proba(frame)[0, 1])
    decision = "APPROVE" if probability >= config.APPROVAL_THRESHOLD else "REJECT"
    return probability, decision


if __name__ == "__main__":
    import data_loader
    import risk_engine

    frame = risk_engine.score_dataframe(data_loader.get_training_data())
    model, _ = train(frame)
    top_drivers(model)
