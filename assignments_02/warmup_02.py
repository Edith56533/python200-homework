import numpy as np
from sklearn.linear_model import LinearRegression

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

#Q1
model = LinearRegression()
model.fit(temp_min, temp_max)
predictions = model.predict([[6], [18]])
print("Slope:",model.coef_[0])
print("Intercept:",model.intercept_)
print("Predictions for 6 and 18:", predictions)

#Q2
x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)
x = x.reshape(-1, 1)
print("Reshaped shape:", x.shape)
# Scikit-learn requires 2D arrays to organize data into rows and columns.

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature (°C)")
plt.ylabel("Daily High Temperature (°C)")
plt.savefig("assignments_02/outputs/high_vs_low.png")
#plt.show()
#Rainy days tend to have lower daily high temperatures compared to non-rainy days, as indicated by the color coding in the scatter plot.
#Both show a positive correlation, as the daily low increases, the daily high generally increases, too.

#  Linear Regression Q2 
X = temp_min.reshape(-1, 1)
y = temp_max
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Linear Regression Q3
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)
print("Slope:", model.coef_[0])
print("R-squared:", r2)
print("Intercept:", model.intercept_)
print("RMSE:", rmse)
# plt.show()

# For every 1°C increase in the daily low,the model predicts the daily high will increase by approximately 0.944°C.


# Linear Regression Q4

X_full = np.column_stack([temp_min, is_rainy])
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42
)
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
r2_full = model_full.score(X_test_full, y_test_full)
print("Original R²:", r2)
print("Full model R²:", r2_full)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

#3For every 1°C increase in the daily low, the predicted high increases approximately 0.98°C. Rainy days are predicted to be approximately 2.92°C cooler than dry days with the same daily low temperature.


# Linear Regression Q5

y_pred_full = model_full.predict(X_test_full)
plt.figure()
plt.scatter(y_pred_full, y_test_full)
limits = [
    min(y_pred_full.min(), y_test_full.min()),
    max(y_pred_full.max(), y_test_full.max())
]
plt.plot(limits, limits, "k--")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Temperature (°C)")
plt.ylabel("Actual Temperature (°C)")
plt.savefig("assignments_02/outputs/predicted_vs_actual_warmup.png")
plt.savefig("assignments_02/outputs/predicted_vs_actual_warmup.png")
plt.show()

# Points above the diagonal mean the model underestimated the actual temperature. Points below mean it overestimated.
