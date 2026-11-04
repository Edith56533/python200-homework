# Task 1: Fetch the Data
# City: Eagle Pass, Texas

import requests
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import json
import sys
import sklearn
from pathlib import Path

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 28.71,
    "longitude": -100.50,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
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

# Task 2: Engineer the Label

df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)
print("\nClass Balance:")
print(df["good_for_running"].value_counts())
good_fraction = df["good_for_running"].mean()
print(f"\nGood running days: {good_fraction:.1%}")
# Only 21.1% of days were good for running, which makes sense because Eagle Pass is hot for most of the year. Since most days are labeled "skip", precision, recall,and F1 are important along with accuracy.


# Task 3: KNN Classifier

features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max"
]
X = df[features]
y = df["good_for_running"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
best_k = None
best_score = 0
for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    )
    mean_score = scores.mean()
    print(f"k={k}: {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_k = k
print(f"\nBest k: {best_k}")
knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
print("\nKNN Accuracy:", accuracy_score(y_test, knn_predictions))
print("\nKNN Classification Report:")
print(classification_report(y_test, knn_predictions))


# Task 4: Logistic Regression

logreg_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])
logreg_model.fit(X_train, y_train)
logreg_predictions = logreg_model.predict(X_test)
print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, logreg_predictions))
print("Precision:",
      precision_score(y_test, logreg_predictions))
print("Recall:",
      recall_score(y_test, logreg_predictions))
print("F1 Score:",
      f1_score(y_test, logreg_predictions))
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, logreg_predictions))
print("\nLogistic Regression Coefficients:")
coefficients = logreg_model.named_steps["logreg"].coef_[0]
for feature, coefficient in zip(features, coefficients):
    print(f"{feature}: {coefficient:.3f}")
# High temperatures and precipitation have the strongest negative effects on running conditions.This makes sense because Eagle Pass is hot more often than not.


# Task 5: Compare and Choose

# KNN had higher accuracy (95.89%) than Logistic Regression (93.15%).Precision matters because recommending a run on a bad day could be unsafe.I chose KNN because it performed better overall.Logistic Regression is simpler and gives more useful probability scores, but KNN was more accurate here.


# Task 6: Save the Model

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
joblib.dump(knn_model, models_dir / "weather_classifier.pkl")
metadata = {
    "city": "Eagle Pass, Texas",
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
    "features": features,
    "label_rule": (
        "High temperature 7-26 C, low temperature >= 0 C, "
        "precipitation < 3 mm, wind speed < 30 km/h"
    ),
    "model": "KNeighborsClassifier",
    "best_k": best_k,
    "test_accuracy": accuracy_score(y_test, knn_predictions),
    "test_f1": f1_score(y_test, knn_predictions)
}
with open(models_dir / "weather_classifier_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)
print("\nModel and metadata saved successfully.")