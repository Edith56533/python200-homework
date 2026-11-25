# Task 7: Load and Predict

import joblib
import json
import pandas as pd

# Load the saved model and metadata
model = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json") as file:
    metadata = json.load(file)
print("City:", metadata["city"])
print("Features:", metadata["features"])
print("Test Accuracy:", metadata["test_accuracy"])

# Create four hypothetical weather days
days = pd.DataFrame([
    [20, 10, 0, 10],     # Very good
    [40, 28, 10, 45],    # Very bad
    [26, 15, 2, 28],     # Borderline
    [15, 5, 0, 12]       # Cool, calm day
], columns=metadata["features"])

# Make predictions
predictions = model.predict(days)
probabilities = model.predict_proba(days)
print("\nWeather Predictions:")
for i in range(len(days)):
    label = "good" if predictions[i] == 1 else "skip"
    confidence = probabilities[i][1]
    print(f"\nDay {i+1}:")
    print(days.iloc[i].to_dict())
    print(f"Prediction: {label}")
    print(f"Probability of good: {confidence:.2f}")


# Task 8: Reflection

# 1. My borderline day received a probability of 0.00. KNN appears confident, but with k=1, its probabilities are always 0 or 1. If it predicted 0.52, I would treat the day as uncertain and check the weather.
# 2. If train_classifier.py hasn't been run, the saved model file won't exist and predict.py will fail. A helpful error could tell the user to train the model first.
# 3. For a cloud pipeline, predict.py would need to use current weather data instead of hypothetical days.