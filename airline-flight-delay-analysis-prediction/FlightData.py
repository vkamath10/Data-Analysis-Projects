
import pandas as pd
import glob
"""
# Folder where your CSVs are stored
path = r"C:\Users\Public\VK\DataAnalysis1\data\raw"

# Get all 2022 CSV files
all_files = glob.glob(path + "/flights_2022_*.csv")

# Read and append them together safely
df_list = []
for file in all_files:
    df = pd.read_csv(file, low_memory=False)  # prevents dtype warnings
    df_list.append(df)

df_2022 = pd.concat(df_list, ignore_index=True)

# Save as one combined file
output_file = path + "/flights_2022.csv"
df_2022.to_csv(output_file, index=False)

print(f"Combined {len(all_files)} files into {output_file}")
print("Total rows:", len(df_2022))
"""
# Step 1: Load data
file_path = r"C:\Users\Public\VK\DataAnalysis1\data\raw\flights_2022.csv"  # your raw CSV
df = pd.read_csv(file_path, low_memory=False)

# Step 2: Keep only important columns
keep_cols = [
    'FlightDate', 'Operating_Airline ', 'Flight_Number_Operating_Airline',
    'Origin', 'Dest', 'CRSDepTime', 'DepTime', 'DepDelay',
    'CRSArrTime', 'ArrTime', 'ArrDelay',
    'Cancelled', 'CancellationCode', 'Diverted',
    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
    'Tail_Number', 'Distance'
]
df = df[keep_cols].copy()

# Step 3: Rename columns (strip spaces + lowercase)
df.columns = [col.strip().lower() for col in df.columns]

# Step 4: Handle missing or invalid data
df = df.fillna({
    'cancelled': 0,
    'diverted': 0,
    'depdelay': 0,
    'arrdelay': 0
})

# Step 5: Convert to correct data types
df['flightdate'] = pd.to_datetime(df['flightdate'], errors='coerce')
df['cancelled'] = df['cancelled'].astype(int)
df['diverted'] = df['diverted'].astype(int)
df['depdelay'] = pd.to_numeric(df['depdelay'], errors='coerce').fillna(0).astype(int)
df['arrdelay'] = pd.to_numeric(df['arrdelay'], errors='coerce').fillna(0).astype(int)

# Step 6: Convert delay cause columns to integers (0 if missing)
delay_cols = ['carrierdelay', 'weatherdelay', 'nasdelay', 'securitydelay', 'lateaircraftdelay']
for col in delay_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Step 7: Save cleaned dataset
output_path = r"C:\Users\Public\VK\DataAnalysis1\data\raw\flights_2022_clean.csv"
df.to_csv(output_path, index=False)

print("Cleaned dataset saved as 'flights_2022_clean.csv'")
print("Rows:", len(df))
print(df.head())



