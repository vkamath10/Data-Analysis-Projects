import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "Data/M5Forecasting/"

# =====================================================
# LOAD RAW DATA
# =====================================================
sales = pd.read_csv(DATA_PATH + "sales_train_validation.csv")
calendar = pd.read_csv(DATA_PATH + "calendar.csv")
prices = pd.read_csv(DATA_PATH + "sell_prices.csv")

print("✅ Raw files loaded")

# =====================================================
# SELECT VALID FMCG ITEM (FOODS + PRICES)
# =====================================================
sales_foods = sales[sales["cat_id"] == "FOODS"]
priced_items = prices[["item_id", "store_id"]].drop_duplicates()

valid_items = sales_foods.merge(
    priced_items,
    on=["item_id", "store_id"],
    how="inner"
)

sample = valid_items.iloc[0]
ITEM_ID = sample["item_id"]
STORE_ID = sample["store_id"]

print(f"Using ITEM_ID={ITEM_ID}, STORE_ID={STORE_ID}")

# =====================================================
# FILTER SALES
# =====================================================
sales = sales[
    (sales["item_id"] == ITEM_ID) &
    (sales["store_id"] == STORE_ID)
]

# =====================================================
# WIDE → LONG
# =====================================================
sales_long = sales.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    var_name="d",
    value_name="units_sold"
)

# =====================================================
# MERGE CALENDAR
# =====================================================
df = sales_long.merge(calendar, on="d", how="left")
df["date"] = pd.to_datetime(df["date"])

# =====================================================
# MERGE PRICES
# =====================================================
df = df.merge(
    prices,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)

df["sell_price"] = df["sell_price"].ffill()

# =====================================================
# FEATURE ENGINEERING
# =====================================================
df["is_holiday"] = df["event_name_1"].notna().astype(int)
df["price_change"] = df["sell_price"].diff().fillna(0)
df["day_of_week"] = df["date"].dt.dayofweek
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month

# =====================================================
# FINAL DAILY DATASET
# =====================================================
final_df = (
    df[[
        "date", "units_sold", "sell_price", "price_change",
        "is_holiday", "day_of_week", "week", "month"
    ]]
    .sort_values("date")
    .reset_index(drop=True)
)

print("✅ Daily dataset ready | Rows:", len(final_df))

# =====================================================
# TRAIN / TEST SPLIT (DAILY)
# =====================================================
split_date = final_df["date"].quantile(0.8)

train_df = final_df[final_df["date"] <= split_date]
test_df  = final_df[final_df["date"] > split_date]

# =====================================================
# 1️⃣ LINEAR REGRESSION (DAILY)
# =====================================================
features = [
    "sell_price", "price_change", "is_holiday",
    "day_of_week", "week", "month"
]

lr = LinearRegression()
lr.fit(train_df[features], train_df["units_sold"])
lr_pred = lr.predict(test_df[features])

print("\n📊 Linear Regression (Daily)")
print("MAE :", mean_absolute_error(test_df["units_sold"], lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(test_df["units_sold"], lr_pred)))
print("R²  :", r2_score(test_df["units_sold"], lr_pred))

# =====================================================
# TIME SERIES (DAILY) — EXPLICIT FREQUENCY (NO WARNINGS)
# =====================================================
ts_daily = (
    final_df
    .set_index("date")
    .asfreq("D")["units_sold"]
)

train_ts = ts_daily[ts_daily.index <= split_date]
test_ts  = ts_daily[ts_daily.index > split_date]

# =====================================================
# 2️⃣ ARIMA (DAILY)
# =====================================================
arima = ARIMA(train_ts, order=(1, 1, 1))
arima_fit = arima.fit()
arima_forecast = arima_fit.forecast(steps=len(test_ts))

print("\n📊 ARIMA (Daily)")
print("MAE :", mean_absolute_error(test_ts, arima_forecast))
print("RMSE:", np.sqrt(mean_squared_error(test_ts, arima_forecast)))

# =====================================================
# 3️⃣ SARIMA / SARIMAX (DAILY)
# =====================================================
exog_cols = [
    "sell_price", "price_change", "is_holiday",
    "day_of_week", "week", "month"
]

exog = final_df.set_index("date").asfreq("D")[exog_cols]
exog_train = exog[exog.index <= split_date]
exog_test  = exog[exog.index > split_date]

sarimax = SARIMAX(
    train_ts,
    exog=exog_train,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_fit = sarimax.fit(disp=False)
sarimax_forecast = sarimax_fit.forecast(
    steps=len(test_ts),
    exog=exog_test
)

print("\n📊 SARIMA / SARIMAX (Daily)")
print("MAE :", mean_absolute_error(test_ts, sarimax_forecast))
print("RMSE:", np.sqrt(mean_squared_error(test_ts, sarimax_forecast)))

# =====================================================
# WEEKLY AGGREGATION
# =====================================================
weekly_df = (
    final_df
    .set_index("date")
    .resample("W")
    .agg({
        "units_sold": "sum",
        "sell_price": "mean",
        "price_change": "sum",
        "is_holiday": "max"
    })
)

print("\n✅ Weekly dataset ready | Rows:", len(weekly_df))

weekly_ts = weekly_df.asfreq("W")["units_sold"]

split_week = weekly_ts.index[int(len(weekly_ts) * 0.8)]

train_w = weekly_ts[weekly_ts.index <= split_week]
test_w  = weekly_ts[weekly_ts.index > split_week]

# =====================================================
# 4️⃣ ARIMA (WEEKLY)
# =====================================================
arima_w = ARIMA(train_w, order=(1, 1, 1))
arima_w_fit = arima_w.fit()
arima_w_forecast = arima_w_fit.forecast(steps=len(test_w))

print("\n📊 ARIMA (Weekly)")
print("MAE :", mean_absolute_error(test_w, arima_w_forecast))
print("RMSE:", np.sqrt(mean_squared_error(test_w, arima_w_forecast)))

# =====================================================
# 5️⃣ SARIMAX (WEEKLY)
# =====================================================
weekly_exog = weekly_df.asfreq("W")[["sell_price", "price_change", "is_holiday"]]

sarimax_w = SARIMAX(
    train_w,
    exog=weekly_exog.loc[train_w.index],
    order=(1, 1, 1),
    seasonal_order=(1, 0, 0, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_w_fit = sarimax_w.fit(disp=False)
sarimax_w_forecast = sarimax_w_fit.forecast(
    steps=len(test_w),
    exog=weekly_exog.loc[test_w.index]
)

print("\n📊 SARIMAX (Weekly)")
print("MAE :", mean_absolute_error(test_w, sarimax_w_forecast))
print("RMSE:", np.sqrt(mean_squared_error(test_w, sarimax_w_forecast)))

# =====================================================
# PLOTS
# =====================================================
plt.figure(figsize=(12, 5))
plt.plot(test_ts.index, test_ts, label="Actual (Daily)", color="black")
plt.plot(test_ts.index, arima_forecast, label="ARIMA (Daily)")
plt.plot(test_ts.index, sarimax_forecast, label="SARIMAX (Daily)")
plt.legend()
plt.title("Daily Demand Forecast")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(test_w.index, test_w, label="Actual (Weekly)", color="black")
plt.plot(test_w.index, arima_w_forecast, label="ARIMA (Weekly)")
plt.plot(test_w.index, sarimax_w_forecast, label="SARIMAX (Weekly)")
plt.legend()
plt.title("Weekly Demand Forecast")
plt.tight_layout()
plt.show()

















