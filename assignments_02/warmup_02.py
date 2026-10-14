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
plt.show()
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
