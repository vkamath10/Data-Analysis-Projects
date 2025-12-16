import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Folder to save datasets
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# File paths
traffic_file = os.path.join(DATA_DIR, "traffic.csv")
weather_file = os.path.join(DATA_DIR, "weather.csv")
accidents_file = os.path.join(DATA_DIR, "accidents.csv")
construction_file = os.path.join(DATA_DIR, "construction.csv")


# -------------------------------
# Function to generate synthetic data
# -------------------------------

def generate_traffic_data(file, n=1000):
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=15 * i) for i in range(n)]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "location": np.random.choice(["Downtown", "Highway", "Suburb", "City Center"], n),
        "cars": np.random.randint(50, 500, n),
        "trucks": np.random.randint(5, 50, n),
        "buses": np.random.randint(1, 20, n),
        "avg_speed": np.random.randint(20, 100, n)
    })
    df.to_csv(file, index=False)
    print(f" Generated {file}")

def generate_weather_data(file, n=1000):
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=15 * i) for i in range(n)]
    conditions = ["Clear", "Rain", "Fog", "Snow", "Cloudy"]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_C": np.random.randint(-5, 35, n),
        "rainfall_mm": np.random.choice([0, 0, 0, 2, 5, 10, 20], n),
        "condition": np.random.choice(conditions, n)
    })
    df.to_csv(file, index=False)
    print(f" Generated {file}")

def generate_accidents_data(file, n=1000):
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=15 * i) for i in range(n)]
    df = pd.DataFrame({
        "timestamp": np.random.choice(timestamps, n),
        "location": np.random.choice(["Downtown", "Highway", "Suburb", "City Center"], n),
        "severity": np.random.choice(["Minor", "Moderate", "Severe"], n),
        "cause": np.random.choice(["Speeding", "Weather", "Distracted Driving", "Mechanical Failure"], n)
    })
    df.to_csv(file, index=False)
    print(f" Generated {file}")

def generate_construction_data(file, n=200):
    timestamps = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    df = pd.DataFrame({
        "timestamp": np.random.choice(timestamps, n),
        "location": np.random.choice(["Downtown", "Highway", "Suburb", "City Center"], n),
        "status": np.random.choice(["Active", "Completed"], n)
    })
    df.to_csv(file, index=False)
    print(f" Generated {file}")

# -------------------------------
# Generate datasets if missing
# -------------------------------
if not os.path.exists(traffic_file):
    generate_traffic_data(traffic_file, 1000)
if not os.path.exists(weather_file):
    generate_weather_data(weather_file, 1000)
if not os.path.exists(accidents_file):
    generate_accidents_data(accidents_file, 1000)
if not os.path.exists(construction_file):
    generate_construction_data(construction_file, 200)

# -------------------------------
# Load datasets
# -------------------------------
print(" Loading datasets...")
traffic_df = pd.read_csv(traffic_file)
weather_df = pd.read_csv(weather_file)
accidents_df = pd.read_csv(accidents_file)
construction_df = pd.read_csv(construction_file)

# Convert timestamps to datetime
for df in [traffic_df, weather_df, accidents_df, construction_df]:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")


print("\n All CSVs exported! Use these for MySQL import.")
