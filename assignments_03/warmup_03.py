import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#---Preprocessing---

# Q1:

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaled training means:")
print(X_train_scaled.mean(axis=0))
# Fit only on training data to prevent data leakage from the test set.


# Q2

encoder = OneHotEncoder(sparse_output=False)
seasons_encoded = encoder.fit_transform(df[["season"]])
print("Encoded array shape:", seasons_encoded.shape)
print("Season categories:", encoder.get_feature_names_out())
# One season column becomes four columns because
# there are four unique seasons in the dataset.

# Q3

features = numeric + ["season"]
X_full = df[features]
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["season"])
])
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train_full, y_train_full)
print("Pipeline test accuracy:", pipeline.score(X_test_full, y_test_full))
# The pipeline automatically scales numerical features, encodes categorical features, and passes the processed data to the classifier for training and prediction.


# --- KNN ---
# Q1

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
knn_pipeline.fit(X_train, y_train)
knn_predictions = knn_pipeline.predict(X_test)
print("\nKNN Test Accuracy:", accuracy_score(y_test, knn_predictions))
print("\nKNN Classification Report:")
print(classification_report(y_test, knn_predictions))


# Q2

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
for k in k_values:
    knn_cv = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(
        knn_cv, X_train, y_train, cv=5, scoring="accuracy"
    )
    print(f"k={k}: Mean CV accuracy = {scores.mean():.4f}")
# I would choose k=11 because it had the highest mean cross-validation accuracy (94.17%). Ironically, increasing k beyond 11 slightly reduced accuracy.


# Q3

knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
raw_accuracy = knn_raw.score(X_test, y_test)
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
knn_scaled.fit(X_train, y_train)
scaled_accuracy = knn_scaled.score(X_test, y_test)
print("\nKNN without scaling:", raw_accuracy)
print("KNN with scaling:", scaled_accuracy)
#Scaling improved accuracy from 92.5% to 95.83%. It helps KNN compare weather values more fairly because they use different units and ranges.


# --- Logistic Regression ---
# Q1

logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])
logreg_pipeline.fit(X_train, y_train)
logreg_predictions = logreg_pipeline.predict(X_test)
print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, logreg_predictions))
print("Precision:",
      precision_score(y_test, logreg_predictions))
print("Recall:",
      recall_score(y_test, logreg_predictions))
print("F1 Score:",
      f1_score(y_test, logreg_predictions))


# Q2

coefficients = logreg_pipeline.named_steps["logreg"].coef_[0]
print("\nLogistic Regression Coefficients:")
for feature, coefficient in zip(numeric, coefficients):
    print(f"{feature}: {coefficient:.3f}")
# Wind speed and precipitation have the strongest negative effects, making a day less suitable for running.This makes sense because running in wind and rain tends to be less enjoyable.

# Q3

probabilities = logreg_pipeline.predict_proba(X_test)
print("\nProbability of good running weather:")
for i, probability in enumerate(probabilities[:5]):
    print(f"Day {i+1}: {probability[1]:.2f}")
# A probability near 0.5 means the model is unsure whether the day is good for running.



# --- Evaluation ---
# Q1

import os
os.makedirs("outputs", exist_ok=True)
cm = confusion_matrix(y_test, logreg_predictions)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"]
)
display.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.show()
print("\nConfusion Matrix:")
print(cm)
# The model made 9 false positives and 7 false negatives. False positives matter more because the app could recommend running when the weather is bad.


# --- joblib ---
# Q1

os.makedirs("models", exist_ok=True)
joblib.dump(logreg_pipeline, "models/warmup_model.pkl")
loaded = joblib.load("models/warmup_model.pkl")
assert (
    loaded.predict(X_test) == logreg_pipeline.predict(X_test)
).all()
print("\nPredictions match.")
# Saving only Logistic Regression would leave out the scaler, so raw data may produce wrong predictions.

