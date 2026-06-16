import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv(r"F:\pypy3\4336502.csv")

# Step 2: Clean the dataset
df = df[["DATE", "TAVG"]]
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

print(df.head())
print(f"\nDate range: {df['DATE'].min()} → {df['DATE'].max()}")

# Step 3: Prepare the data
# Original lags
df["lag1"] = df["TAVG"].shift(1)
df["lag2"] = df["TAVG"].shift(2)
df["lag3"] = df["TAVG"].shift(3)

# Longer lags
df["lag7"]   = df["TAVG"].shift(7)    # same day last week
df["lag14"]  = df["TAVG"].shift(14)   # two weeks ago
df["lag365"] = df["TAVG"].shift(365)  # same day last year — captures yearly seasonality

# Rolling averages
df["rolling_7"]  = df["TAVG"].rolling(window=7).mean()
df["rolling_30"] = df["TAVG"].rolling(window=30).mean()
df["rolling_7_std"] = df["TAVG"].rolling(window=7).std()

# Calendar features
df["month"]      = df["DATE"].dt.month
df["dayofyear"]  = df["DATE"].dt.dayofyear

df = df.dropna().reset_index(drop=True)

print(df.head())
print(f"Rows after cleanup: {len(df)}")

# Step 4: Split into train/test
split = int(len(df) * 0.8)

train = df.iloc[:split]
test  = df.iloc[split:]

features = [
    "lag1", "lag2", "lag3",
    "lag7", "lag14", "lag365",
    "rolling_7", "rolling_30", "rolling_7_std",
    "month", "dayofyear"
]
target = "TAVG"

X_train, y_train = train[features], train[target]
X_test,  y_test  = test[features],  test[target]

print(f"Training: {train['DATE'].min().date()} → {train['DATE'].max().date()}  ({len(train)} rows)")
print(f"Testing:  {test['DATE'].min().date()} → {test['DATE'].max().date()}  ({len(test)} rows)")

# Step 5: Train & Evaluate models
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# XGBoost
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

print(f"\nLinear Regression MAE: {mean_absolute_error(y_test, lr_preds):.2f}°C")
print(f"Random Forest MAE:     {mean_absolute_error(y_test, rf_preds):.2f}°C")
print(f"XGBoost MAE:           {mean_absolute_error(y_test, xgb_preds):.2f}°C")

# Step 6: Visualize predictions
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: Full test period
axes[0].plot(test["DATE"].values, y_test.values,  label="Actual",            color="black",     linewidth=1.5)
axes[0].plot(test["DATE"].values, lr_preds,        label="Linear Regression", color="steelblue", linestyle="--", alpha=0.8)
axes[0].plot(test["DATE"].values, rf_preds,        label="Random Forest",     color="tomato",    linestyle="--", alpha=0.8)
axes[0].plot(test["DATE"].values, xgb_preds,       label="XGBoost",           color="seagreen",  linestyle="--", alpha=0.8)
axes[0].set_title("HCMC Temperature Forecast — Full Test Period (Improved)")
axes[0].set_ylabel("Avg Temp (°C)")
axes[0].legend()

# Plot 2: Zoom into last 60 days for detail
axes[1].plot(test["DATE"].values[-60:], y_test.values[-60:],  label="Actual",            color="black",     linewidth=2)
axes[1].plot(test["DATE"].values[-60:], lr_preds[-60:],        label="Linear Regression", color="steelblue", linestyle="--")
axes[1].plot(test["DATE"].values[-60:], rf_preds[-60:],        label="Random Forest",     color="tomato",    linestyle="--")
axes[1].plot(test["DATE"].values[-60:], xgb_preds[-60:],       label="XGBoost",           color="seagreen",  linestyle="--")
axes[1].set_title("Zoomed In — Last 60 Days (Improved)")
axes[1].set_ylabel("Avg Temp (°C)")
axes[1].legend()

plt.tight_layout()
plt.show()