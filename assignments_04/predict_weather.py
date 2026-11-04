from pathlib import Path
from weather_model import WeatherClassifier

def main():
    model_path = Path(__file__).parent / "models" / "weather_classifier.pkl"
    classifier = WeatherClassifier(model_path)
    days = [
        {
            "temperature_2m_max": 24,
            "temperature_2m_min": 15,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 10,
        },
        {
            "temperature_2m_max": 5,
            "temperature_2m_min": -2,
            "precipitation_sum": 15,
            "wind_speed_10m_max": 50,
        },
        {
            "temperature_2m_max": 30,
            "temperature_2m_min": 20,
            "precipitation_sum": 2,
            "wind_speed_10m_max": 20,
        },
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 5,
            "wind_speed_10m_max": 25,
        },
    ]
    predictions = classifier.predict(days)

    for number, (day, prediction) in enumerate(
        zip(days, predictions), start=1
    ):
        print(f"\nDay {number}:")
        print(f"Weather: {day}")
        print(f"Prediction: {prediction.label}")
        print(f"Probability of good: {prediction.probability:.2f}")
if __name__ == "__main__":
    main()
# The main guard prevents predictions from running automatically when another script imports this file.


# Task 6: Reflection

# 1. Keeping the features in the same order prevents the model from mixing up values, like reading wind speed as temperature.

# 2. Using a small test model lets us test the code without relying on the original saved model.

# 3. The classifier can handle multiple days at once and only loads the model once. I would add more input checks before using it in a real application.