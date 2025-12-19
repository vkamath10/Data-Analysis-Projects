# =====================================================
# FUTURE FLIGHT DELAY RISK PREDICTION (FINAL VERSION)
# =====================================================

import pandas as pd
import joblib
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# -----------------------------------------------
# 1 CONFIGURATION
# -----------------------------------------------
MODEL_PATH = "flight_delay_model.pkl"
DB_USER = "root"
DB_PASSWORD = "******"
DB_HOST = "localhost"
DB_NAME = "flight_data"
TARGET_TABLE = "predicted_delays_future"

# -----------------------------------------------
# 2 CREATE SIMULATED FUTURE FLIGHTS
# -----------------------------------------------
print("Creating simulated future flight data...")

future_flights = pd.DataFrame({
    "flight_id": [90001, 90002, 90003, 90004, 90005],
    "operating_airline": ["AA", "DL", "UA", "B6", "WN"],
    "origin": ["JFK", "ATL", "ORD", "BOS", "LAX"],
    "dest": ["LAX", "DFW", "DEN", "MIA", "SEA"],
    "depdelay": [0, 5, 10, 3, 7],
    "arrdelay": [0, 0, 0, 0, 0],          # assume on-time departure for simulation
    "weatherdelay": [2, 0, 1, 3, 0],      # simulate possible weather impacts
    "flightdate": [datetime.now() + timedelta(days=i) for i in [1, 2, 3, 4, 5]]
})

print(f"Simulated {len(future_flights)} future flights.")

# -----------------------------------------------
# 3 LOAD TRAINED MODEL
# -----------------------------------------------
print("Loading trained model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# -----------------------------------------------
# 4 PREDICT TOTAL DELAY
# -----------------------------------------------
print(" Predicting total delay...")

# Select only the 6 feature columns used in training
features = future_flights[['operating_airline', 'origin', 'dest',
                           'depdelay', 'arrdelay', 'weatherdelay']]

# Run model prediction
future_flights['predicted_total_delay'] = model.predict(features)
future_flights['prediction_timestamp'] = datetime.now()

# Categorize risk for Power BI visualization
def risk_level(delay):
    if delay <= 5:
        return "Low"
    elif delay <= 20:
        return "Medium"
    elif delay <= 40:
        return "High"
    else:
        return "Severe"

future_flights['delay_risk'] = future_flights['predicted_total_delay'].apply(risk_level)

print("Predictions and risk levels generated.")

# -----------------------------------------------
# 5 UPLOAD TO MYSQL
# -----------------------------------------------
print("Uploading results to MySQL...")
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
future_flights.to_sql(TARGET_TABLE, con=engine, if_exists="replace", index=False)
print(f" Data uploaded successfully to MySQL table: {TARGET_TABLE}")

# -----------------------------------------------
# 6 SUMMARY
# -----------------------------------------------
print("\nFuture Flight Delay Prediction Pipeline completed successfully!")
print(f" Power BI can now read from: {DB_NAME}.{TARGET_TABLE}")
print(f"Columns available: {', '.join(future_flights.columns)}")
