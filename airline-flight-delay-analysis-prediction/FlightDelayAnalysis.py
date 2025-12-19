# 2 tables flights and delay causes were created in MySql and uploaded the data to the tables.
# ========================================
# STEP 1 — CONNECT TO MYSQL & LOAD DATA
# ========================================

import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

user = "root"
password = "******" #Enter your mysql password here
host = "localhost"      
database = "flight_data"
# -----------------------

# Create SQLAlchemy connection
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

# Query: join flights & delay_causes for January 2022
query = """
SELECT 
    f.flight_id,
    f.flightdate,
    f.operating_airline,
    f.flight_number_operating_airline,
    f.origin,
    f.dest,
    f.depdelay,
    f.arrdelay,
    COALESCE(dc.carrierdelay, 0) AS carrierdelay,
    COALESCE(dc.weatherdelay, 0) AS weatherdelay,
    COALESCE(dc.nasdelay, 0) AS nasdelay,
    COALESCE(dc.securitydelay, 0) AS securitydelay,
    COALESCE(dc.lateaircraftdelay, 0) AS lateaircraftdelay,
    (f.depdelay + f.arrdelay +
     COALESCE(dc.carrierdelay, 0) + 
     COALESCE(dc.weatherdelay, 0) + 
     COALESCE(dc.nasdelay, 0) + 
     COALESCE(dc.securitydelay, 0) + 
     COALESCE(dc.lateaircraftdelay, 0)) AS total_delay
FROM flights f
LEFT JOIN delay_causes dc ON f.flight_id = dc.flight_id
WHERE f.cancelled = 0
  AND f.flightdate BETWEEN '2022-01-01' AND '2022-01-31';
"""

# Load into a DataFrame
flights = pd.read_sql(query, engine)
print(" Data loaded:", flights.shape)
flights.head()

# Fill missing delays with 0
delay_cols = ['depdelay', 'arrdelay', 'carrierdelay', 'weatherdelay',
              'nasdelay', 'securitydelay', 'lateaircraftdelay']
flights[delay_cols] = flights[delay_cols].fillna(0)

# Drop missing target values if any
flights = flights.dropna(subset=['total_delay'])

# Optional: show quick stats
print(flights.describe())

# --- Average delay by airline ---
airline_delay = flights.groupby('operating_airline')['total_delay'].mean().sort_values(ascending=False)
airline_delay.plot(kind='bar', figsize=(10,5), title='Average Delay by Airline', ylabel='Minutes')
plt.show()

# --- Average delay by airport ---
airport_delay = flights.groupby('origin')['total_delay'].mean().sort_values(ascending=False).head(15)
airport_delay.plot(kind='bar', figsize=(10,5), color='orange', title='Top 15 Airports by Average Delay', ylabel='Minutes')
plt.show()

# --- Weather impact ---
flights['weather_impact'] = flights['weatherdelay'] > 0
avg_delay_weather = flights.groupby('weather_impact')['total_delay'].mean()
avg_delay_weather.plot(kind='bar', title='Average Delay: Weather vs No Weather')
plt.xticks([0,1], ['No Weather Delay', 'Weather Delay'], rotation=0)
plt.ylabel("Average Delay (minutes)")
plt.show()

# --- Correlation between delay causes ---
corr = flights[delay_cols + ['total_delay']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Between Delay Factors')
plt.show()

# Select features & target
features = ['operating_airline', 'origin', 'dest', 'depdelay', 'arrdelay', 'weatherdelay']
target = 'total_delay'

X = flights[features]
y = flights[target]

# Define categorical & numerical columns
cat_features = ['operating_airline', 'origin', 'dest']
num_features = ['depdelay', 'arrdelay', 'weatherdelay']

# OneHotEncode categorical variables
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
], remainder='passthrough')

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build regression pipeline
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(" Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

encoded_features = list(model.named_steps['preprocess'].transformers_[0][1].get_feature_names_out(cat_features))
all_features = encoded_features + num_features
coefficients = model.named_steps['regressor'].coef_

importance = pd.DataFrame({
    'feature': all_features,
    'coefficient': coefficients
}).sort_values(by='coefficient', ascending=False)

print("Top 10 features influencing total delay:")
print(importance.head(10))

new_flight = pd.DataFrame([{
    'operating_airline': 'AA',
    'origin': 'JFK',
    'dest': 'LAX',
    'depdelay': 15,
    'arrdelay': 10,
    'weatherdelay': 5
}])

predicted_delay = model.predict(new_flight)
print(f"Predicted total delay: {predicted_delay[0]:.2f} minutes")

# Save model (pipeline + preprocessing)
joblib.dump(model, 'flight_delay_model.pkl')

print(" Model saved successfully as flight_delay_model.pkl")

""" Insights and Recommendations:
Area	Key Insight	Recommendation
Airline Ops	Certain regional airports (DEC, PIB, BGM, CGI) drive chronic delays.Audit ground-handling and scheduling buffers at these stations; review connection timing.
Weather Impact	Flights with any weatherdelay > 0 show much higher total_delay.	Integrate real-time weather alerts into scheduling; pre-position crews and de-icing equipment.
Network Design	Late-aircraft and NAS delays still contribute even with low averages.	Strengthen rotation planning; stagger departures from congested hubs.
Predictive Use	Model can flag high-risk flights ~47 min before takeoff.	Feed risk level into Ops dashboards to trigger proactive re-routing or passenger notifications.
Continuous Improvement	Extreme outliers (>99 th percentile) distort averages.	Cap for analytics; investigate individually for systemic issues (maintenance, ATC holds).
"""