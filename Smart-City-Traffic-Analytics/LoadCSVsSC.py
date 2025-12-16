import pandas as pd
from sqlalchemy import create_engine, text

# -------------------------------
# 🔧 MySQL connection settings
# -------------------------------
USER = "root"
PASSWORD = "Vkamath1982!"   # ← change to your actual MySQL password
HOST = "localhost"
DB_NAME = "smartcity"

# -------------------------------
# 📁 CSV file paths
# -------------------------------
traffic_csv = "data/traffic.csv"
weather_csv = "data/weather.csv"
accidents_csv = "data/accidents.csv"
construction_csv = "data/construction.csv"

# -------------------------------
# ⚙️ Connect to MySQL (default port 3306 used automatically)
# -------------------------------
engine = create_engine(f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}")
with engine.connect() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};"))
    conn.execute(text(f"USE {DB_NAME};"))

engine = create_engine(f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}/{DB_NAME}")

# -------------------------------
# 🧱 Table creation SQL
# -------------------------------
schema_sql = """
DROP TABLE IF EXISTS master_dataset;
DROP TABLE IF EXISTS accident_report;
DROP TABLE IF EXISTS road_construction;
DROP TABLE IF EXISTS weather;
DROP TABLE IF EXISTS traffic;

CREATE TABLE traffic (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    location VARCHAR(100) NOT NULL,
    cars INT,
    trucks INT,
    buses INT,
    avg_speed DECIMAL(5,2),
    INDEX (timestamp),
    INDEX (location)
);

CREATE TABLE weather (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    temperature DECIMAL(4,1),
    precipitation DECIMAL(4,1),
    weather_condition ENUM('Clear','Cloudy','Rain','Storm','Fog','Snow'),
    INDEX (timestamp)
);

CREATE TABLE accident_report (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    location VARCHAR(100) NOT NULL,
    severity ENUM('Minor','Moderate','Severe'),
    cause VARCHAR(100),
    status ENUM('Open','Closed') NOT NULL DEFAULT 'Open',
    INDEX (timestamp),
    INDEX (location)
);

CREATE TABLE road_construction (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    location VARCHAR(100) NOT NULL,
    status ENUM('Active','Completed') NOT NULL,
    INDEX (timestamp),
    INDEX (location)
);
"""

# Execute schema creation
with engine.begin() as conn:
    for statement in schema_sql.strip().split(";"):
        if statement.strip():
            conn.execute(text(statement))

print("✅ Tables created successfully.")

# -------------------------------
# 📥 Load CSVs into DataFrames
# -------------------------------
def load_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

traffic_df = load_csv(traffic_csv)
weather_df = load_csv(weather_csv)
accidents_df = load_csv(accidents_csv)
construction_df = load_csv(construction_csv)

# -------------------------------
# 🧹 Data cleaning / normalization
# -------------------------------

# Accident report: add status if missing
if 'status' not in accidents_df.columns:
    accidents_df['status'] = 'Open'

# Normalize all column names for weather
weather_df.columns = weather_df.columns.str.lower().str.strip()

# ✅ Rename columns to match MySQL table schema
rename_map = {
    'temperature_c': 'temperature',
    'rainfall_mm': 'precipitation',
    'condition': 'weather_condition'
}
weather_df = weather_df.rename(columns=rename_map)

print("✅ Weather columns normalized to:", list(weather_df.columns))

# Convert timestamps (dayfirst=True because your CSV uses DD-MM-YYYY)
for df in [traffic_df, weather_df, accidents_df, construction_df]:
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True)

# -------------------------------
# 🚀 Upload to MySQL
# -------------------------------
traffic_df.to_sql('traffic', con=engine, if_exists='append', index=False)
weather_df.to_sql('weather', con=engine, if_exists='append', index=False)
accidents_df.to_sql('accident_report', con=engine, if_exists='append', index=False)
construction_df.to_sql('road_construction', con=engine, if_exists='append', index=False)

# -------------------------------
# 🧩 Build master_dataset table
# -------------------------------
create_master_sql = """
DROP TABLE IF EXISTS master_dataset;

CREATE TABLE master_dataset AS
WITH
traffic_base AS (
  SELECT
    t.timestamp,
    t.location,
    t.cars,
    t.trucks,
    t.buses,
    t.avg_speed
  FROM traffic t
),
accident_summary AS (
  SELECT
    a.timestamp,
    a.location,
    COUNT(*) AS accident_count
  FROM accident_report a
  GROUP BY a.timestamp, a.location
),
construction_summary AS (
  SELECT
    c.timestamp,
    c.location,
    SUM(c.status = 'Active') AS active_constructions,
    SUM(c.status = 'Completed') AS completed_constructions
  FROM road_construction c
  GROUP BY c.timestamp, c.location
)
SELECT
  tb.timestamp,
  tb.location,
  tb.cars,
  tb.trucks,
  tb.buses,
  tb.avg_speed,
  w.temperature,
  w.precipitation,
  w.weather_condition,
  COALESCE(a.accident_count, 0) AS accident_count,
  COALESCE(cs.active_constructions, 0) AS active_constructions,
  COALESCE(cs.completed_constructions, 0) AS completed_constructions
FROM traffic_base tb
LEFT JOIN weather w
  ON w.timestamp = tb.timestamp
LEFT JOIN accident_summary a
  ON a.timestamp = tb.timestamp AND a.location = tb.location
LEFT JOIN construction_summary cs
  ON cs.timestamp = tb.timestamp AND cs.location = tb.location;

CREATE INDEX idx_master_time_loc ON master_dataset (timestamp, location);
"""

with engine.begin() as conn:
    print("⚙️ Building master_dataset table...")
    for stmt in create_master_sql.strip().split(";"):
        if stmt.strip():
            conn.execute(text(stmt))

print("✅ master_dataset table created successfully.")


print("✅ All CSVs imported successfully into MySQL database.")
