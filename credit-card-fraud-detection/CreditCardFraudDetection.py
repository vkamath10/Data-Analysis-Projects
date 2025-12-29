# ============================================================
# Credit Card Fraud Detection – IEEE-CIS
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
train_transaction = pd.read_csv("train_transaction.csv")
train_identity = pd.read_csv("train_identity.csv")

print("Transaction shape:", train_transaction.shape)
print("Identity shape:", train_identity.shape)

# ------------------------------------------------------------
# 2. Merge
# ------------------------------------------------------------
train = train_transaction.merge(
    train_identity,
    on="TransactionID",
    how="left"
)

print("Merged shape:", train.shape)

# ------------------------------------------------------------
# 3. Fraud distribution
# ------------------------------------------------------------
print("\nFraud distribution:")
print(train["isFraud"].value_counts(normalize=True))

# ------------------------------------------------------------
# 4. Feature selection
# ------------------------------------------------------------
features = [
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
    "DeviceInfo",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain"
]

X = train[features]
y = train["isFraud"]

# ------------------------------------------------------------
# 5. Column types
# ------------------------------------------------------------
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# ------------------------------------------------------------
# 6. Preprocessing
# ------------------------------------------------------------
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_pipeline, categorical_features),
        ("num", numerical_pipeline, numerical_features)
    ]
)

# ------------------------------------------------------------
# 7. Model
# ------------------------------------------------------------
model = LogisticRegression(
    solver="saga",
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)

# ------------------------------------------------------------
# 8. Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# 9. Train
# ------------------------------------------------------------
print("\nTraining Logistic Regression (FINAL, memory-safe)...")
pipeline.fit(X_train, y_train)

# ------------------------------------------------------------
# 10. Default evaluation
# ------------------------------------------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression Results (Default Threshold 0.5) ---")
print(classification_report(y_test, y_pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ------------------------------------------------------------
# 11. FRAUD ALERT SYSTEM – SAFE THRESHOLD SELECTION
# ------------------------------------------------------------
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

pr_df = pd.DataFrame({
    "threshold": thresholds,
    "precision": precision[:-1],
    "recall": recall[:-1]
})

# Business preferences 
min_recall = 0.75
min_precision = 0.10

candidates = pr_df[
    (pr_df["recall"] >= min_recall) &
    (pr_df["precision"] >= min_precision)
]

if not candidates.empty:
    # Ideal case
    alert_row = candidates.sort_values(
        by="recall", ascending=False
    ).iloc[0]
    decision_note = "Used balanced recall + precision threshold"
else:
    # Fallback: maximise F1-score
    pr_df["f1"] = 2 * (
        pr_df["precision"] * pr_df["recall"]
    ) / (pr_df["precision"] + pr_df["recall"] + 1e-9)

    alert_row = pr_df.sort_values(
        by="f1", ascending=False
    ).iloc[0]
    decision_note = "Fallback: best F1-score (constraints not jointly achievable)"

alert_threshold = alert_row["threshold"]

print("\n--- Fraud Alert Threshold Selection ---")
print(alert_row)
print("Decision:", decision_note)

# ------------------------------------------------------------
# 12. Apply alert threshold
# ------------------------------------------------------------
y_alert = (y_prob >= alert_threshold).astype(int)

print("\n--- Fraud Alert System Results ---")
print(classification_report(y_test, y_alert, zero_division=0))

print("\n FINAL PIPELINE EXECUTED SUCCESSFULLY")



