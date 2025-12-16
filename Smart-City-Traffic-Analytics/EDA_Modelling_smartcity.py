# ============================================================
# Smart City Traffic — Step 3 (EDA) & Step 4 (ML)
# Run in VS Code: python EDA_and_Modeling.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)

# -------------------------------
# 1) Load data from MySQL
# -------------------------------
ENGINE_URL = "mysql+mysqlconnector://root:Vkamath1982!@localhost/smartcity"  
engine = create_engine(ENGINE_URL)

print("⏳ Loading data from MySQL...")
df = pd.read_sql("SELECT * FROM master_dataset", con=engine)
print("✅ Data loaded:", df.shape)
print(df.head(), "\n")

# -------------------------------
# 2) Preprocess / features
# -------------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day_name()
df['total_vehicles'] = df[['cars','trucks','buses']].sum(axis=1)

# quick sanity
print("Columns:", list(df.columns))
print(df.describe(include='all'), "\n")

# -------------------------------
# 3) EDA — Trend analysis
# -------------------------------

# Peak vs off-peak hours
hourly = df.groupby('hour')['total_vehicles'].mean().reset_index()

plt.figure(figsize=(10,5))
plt.plot(hourly['hour'], hourly['total_vehicles'], marker='o')
plt.title("Average Traffic Volume by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Average Vehicles")
plt.grid(True)
plt.tight_layout()
plt.show()

# Most congested areas (lower avg speed)
by_loc = df.groupby('location')['avg_speed'].mean().sort_values().reset_index()

plt.figure(figsize=(8,4))
plt.bar(by_loc['location'], by_loc['avg_speed'])
plt.title("Average Speed by Location (Lower = More Congested)")
plt.xlabel("Location")
plt.ylabel("Avg Speed (km/h)")
plt.tight_layout()
plt.show()

# Weather impact on speed
weather_speed = df.groupby('weather_condition')['avg_speed'].mean().sort_values().reset_index()

plt.figure(figsize=(8,4))
plt.bar(weather_speed['weather_condition'], weather_speed['avg_speed'])
plt.title("Impact of Weather on Traffic Speed")
plt.xlabel("Weather Condition")
plt.ylabel("Avg Speed (km/h)")
plt.tight_layout()
plt.show()

# Correlation heatmap
num_cols = ['cars','trucks','buses','avg_speed','temperature','precipitation','accident_count','active_constructions']
corr = df[num_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap — Traffic, Weather, Accidents")
plt.tight_layout()
plt.show()

# -------------------------------
# 4) ML — Predictive Analysis
# -------------------------------

# 4a) Linear Regression — predict avg_speed (congestion proxy)
X_reg = df[['temperature', 'precipitation', 'cars', 'trucks', 'buses', 'active_constructions']]
y_reg = df['avg_speed']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(Xr_train, yr_train)
yr_pred = lr.predict(Xr_test)

print("📊 Linear Regression (predict avg_speed)")
print("  MAE:", round(mean_absolute_error(yr_test, yr_pred), 3))
print("  R² :", round(r2_score(yr_test, yr_pred), 4))
print("  Coefficients:\n", pd.Series(lr.coef_, index=X_reg.columns).sort_values(), "\n")

# 4b) Logistic Regression — classify accident risk (rare events)
df['accident_risk'] = (df['accident_count'] > 0).astype(int)

X_clf = df[['avg_speed', 'temperature', 'precipitation', 'active_constructions']]
y_clf = df['accident_risk']

# Use class_weight='balanced' to handle imbalance without manual resampling
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(Xc_train, yc_train)
yc_pred = clf.predict(Xc_test)

print("📈 Logistic Regression (accident risk)")
print("  Accuracy:", round(accuracy_score(yc_test, yc_pred), 3))
print("  Confusion matrix:\n", confusion_matrix(yc_test, yc_pred))
print("  Classification report:\n", classification_report(yc_test, yc_pred, zero_division=0))

# Feature influence (coefficients)
coef_series = pd.Series(clf.coef_[0], index=X_clf.columns).sort_values()
plt.figure(figsize=(6,4))
coef_series.plot(kind='barh')
plt.title("Feature Influence on Accident Risk (Logistic Coefficients)")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.show()

# -------------------------------
# (Optional) Alternative: Upsample positives for comparison
# -------------------------------
# from sklearn.utils import resample
# majority = df[df['accident_risk'] == 0]
# minority = df[df['accident_risk'] == 1]
# minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
# df_bal = pd.concat([majority, minority_up])
# Xb = df_bal[['avg_speed','temperature','precipitation','active_constructions']]
# yb = df_bal['accident_risk']
# Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, random_state=42, stratify=yb)
# clf_bal = LogisticRegression(max_iter=1000)
# clf_bal.fit(Xb_train, yb_train)
# yb_pred = clf_bal.predict(Xb_test)
# print("\n🔁 Logistic (with upsampling)")
# print("  Accuracy:", round(accuracy_score(yb_test, yb_pred), 3))
# print("  Confusion matrix:\n", confusion_matrix(yb_test, yb_pred))
# print("  Classification report:\n", classification_report(yb_test, yb_pred, zero_division=0))
