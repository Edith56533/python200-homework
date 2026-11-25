# Week 2 Mini-Project
# City: Eagle Pass, Texas

import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 28.7091,
    "longitude": -100.4995,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Chicago",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
print("Dataset shape:", df.shape)
print(df.head())


# Task 2: Explore the Data

print(df.describe())
print("Missing values:")
print(df.isnull().sum())
correlations = df.corr(numeric_only=True)["temperature_2m_max"]
print(correlations.sort_values())

import matplotlib.pyplot as plt

plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"])
plt.title("Daily High vs Low - Eagle Pass, TX")
plt.xlabel("Daily Low Temperature (°C)")
plt.ylabel("Daily High Temperature (°C)")
plt.savefig("assignments_02/outputs/weather_high_vs_low.png")
#plt.show()
# Daily high and low temperatures have a strong positive relationship. Higher daily lows mostly correspond to higher daily highs, although some variation still exists.


# Task 3: Engineer a Feature

df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)
print(df["is_summer"].value_counts())
print("Summer days:", df["is_summer"].sum())

# Summer may help predict higher temperatures, but Eagle Pass is hot throughout most of the year, so I'm unsure how much additional help this feature will provide.

# Task 4: Baseline Model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

X = df[["temperature_2m_min"]]
y = df["temperature_2m_max"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)
print("Slope:", model.coef_[0])
print("R-squared:", r2)
print("Intercept:", model.intercept_)
print("RMSE:", rmse)

# The baseline model has an RMSE of about 3.14°C, meaning its predictions have errors of roughly 3 degrees Celsius.

# Task 5: Full Model

feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer"
]

X_full = df[feature_cols].values
y_full = df["temperature_2m_max"].values
X_train_full, X_test_full, y_train_full, y_test_full= train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    random_state=42
)
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
y_pred_full = model_full.predict(X_test_full)
rmse_full = np.sqrt(mean_squared_error(y_test_full, y_pred_full))
train_r2 = model_full.score(X_train_full, y_train_full)
test_r2 = model_full.score(X_test_full, y_test_full)
print("Training R²:", train_r2)
print("Testing R²:", test_r2)
print("RMSE:", rmse_full)
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# Adding more features slightly reduced the model's test R².Minimum temperature has a strong positive relationship with daily highs. Rainfall lowers predicted highs, while wind speed and summer increase them. Training and testing R² are close, suggesting no major overfitting. I expected the full model to perform better, but it didn't.


# Task 6: Evaluate and Summarize

plt.figure()
plt.scatter(y_pred_full, y_test_full)
limits = [
    min(y_pred_full.min(), y_test_full.min()),
    max(y_pred_full.max(), y_test_full.max())
]
plt.plot(limits, limits, "k--")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Temperature (°C)")
plt.ylabel("Actual Temperature (°C)")
plt.savefig("assignments_02/outputs/predicted_vs_actual_high.png")
plt.show()

# Task 6 Summary

# The dataset contains 365 days of weather observations from Eagle Pass, Texas, with 73 of those used for testing. The full model achieved an RMSE of 3.17°C and an R² of 0.8648. This means prediction errors are roughly 3.17°C, and the model explains about 86.5% of the variation in daily highs. Minimum temperature has the strongest correlation with daily highs (0.92). Its coefficient is +0.904, meaning a 1°C increase in the daily low raises the predicted high by about 0.90°C, holding other features constant.The predicted vs actual plot shows errors across the temperature range, with some larger deviations in the middle and lower ranges. I was surprised that adding precipitation, wind speed, and summer slightly reduced the model's accuracy compared to using minimum temperature alone.