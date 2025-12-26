# ============================================================
# Customer Churn Prediction - Exploratory Data Analysis (EDA)
# Dataset: Telco Customer Churn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# Display settings
# ------------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------------------------
# Basic Info
# ------------------------------------------------------------
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ------------------------------------------------------------
# Data Cleaning
# ------------------------------------------------------------
# Convert TotalCharges to numeric (contains blank spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop rows with missing TotalCharges
df.dropna(inplace=True)

print("\nShape after dropping missing values:", df.shape)

# ------------------------------------------------------------
# Target Variable Analysis (Churn)
# ------------------------------------------------------------
print("\nChurn Distribution:")
print(df['Churn'].value_counts())
print("\nChurn Percentage:")
print(df['Churn'].value_counts(normalize=True) * 100)

plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# ------------------------------------------------------------
# Tenure vs Churn
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True)
plt.title("Tenure Distribution by Churn")
plt.show()

# ------------------------------------------------------------
# Monthly Charges vs Churn
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# ------------------------------------------------------------
# Contract Type vs Churn
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.show()

# ------------------------------------------------------------
# Payment Method vs Churn
# ------------------------------------------------------------
"""plt.figure(figsize=(9,5))
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.title("Payment Method vs Churn")
plt.xticks(rotation=10, ha='right')
plt.show()
"""
# Calculate churn rate
payment_churn = (
    df.groupby('PaymentMethod')['Churn']
      .value_counts(normalize=True)
      .rename('rate')
      .reset_index()
)

payment_churn = payment_churn[payment_churn['Churn'] == 'Yes']

plt.figure(figsize=(10,6))
sns.barplot(
    x='PaymentMethod',
    y='rate',
    data=payment_churn
)
plt.title("Churn Rate by Payment Method")
plt.ylabel("Churn Rate")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Tech Support (Complaints Proxy) vs Churn
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.countplot(x='TechSupport', hue='Churn', data=df)
plt.title("Tech Support vs Churn")
plt.show()

# ------------------------------------------------------------
# Correlation Heatmap (Numeric Features)
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(
    df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------------------------
# EDA Summary (Printed for reference)
# ------------------------------------------------------------
print("\nEDA SUMMARY:")
print("- Customers with shorter tenure are more likely to churn")
print("- Month-to-month contracts show the highest churn rate")
print("- Higher monthly charges correlate with higher churn")
print("- Electronic check payment method has higher churn")
print("- Lack of tech support strongly increases churn risk")

# ============================================================
# Feature Selection
# ============================================================

# Drop customerID (not useful for prediction)
df_model = df.drop('customerID', axis=1)

# Target variable
y = df_model['Churn'].map({'Yes': 1, 'No': 0})

# Features
X = df_model.drop('Churn', axis=1)
# One-Hot Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

print("Feature matrix shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = log_model.predict(X_test_scaled)
y_pred_prob = log_model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy_lr = log_model.score(X_test_scaled, y_test)
# ROC-AUC
roc_auc_lr = roc_auc_score(y_test, y_pred_prob)
print("ROC-AUC Score:", roc_auc_lr)

# Create DataFrame for coefficients
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_model.coef_[0]
})

# Sort by absolute importance
feature_importance['abs_coef'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='abs_coef', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

"""
Month-to-month contracts increase churn probability
* Short tenure strongly increases churn risk
* Higher monthly charges increase churn
* Lack of tech support increases churn
* Long-term contracts reduce churn

 Example explanation:

Logistic Regression coefficients show that customers on month-to-month contracts with high monthly charges and
short tenure are significantly more likely to churn.

"""
# Train Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=50,
    random_state=42
)

dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)
y_pred_dt_prob = dt_model.predict_proba(X_test)[:, 1]

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))
accuracy_dt = dt_model.score(X_test, y_test)
roc_auc_dt = roc_auc_score(y_test, y_pred_dt_prob)
print("ROC-AUC Score:",roc_auc_dt)

feature_importance_dt = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Decision Tree Features:")
print(feature_importance_dt.head(10))

# ============================================================
# Random Forest - Customer Churn Prediction
# ============================================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ------------------------------------------------------------
# Train Random Forest Model
# ------------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# ------------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------------

y_pred_rf = rf_model.predict(X_test)
y_pred_rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\nRANDOM FOREST RESULTS")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

roc_auc_rf = roc_auc_score(y_test, y_pred_rf_prob)
accuracy_rf = rf_model.score(X_test, y_test)

print("ROC-AUC Score:", roc_auc_rf)
print("Accuracy:", accuracy_rf)

# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------

feature_importance_rf = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features (Random Forest):")
print(feature_importance_rf.head(10))

# ------------------------------------------------------------
# Model Comparison Table
# ------------------------------------------------------------

model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_lr, accuracy_dt, accuracy_rf],
    'ROC-AUC': [roc_auc_lr, roc_auc_dt, roc_auc_rf]
})

print("\nMODEL COMPARISON:")
print(model_comparison)

"""
FINAL CONCLUSION:
- Logistic Regression offers interpretability
- Decision Tree provides clear churn rules
- Random Forest gives best overall performance
"""

