import joblib
import pytest
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from weather_model import WeatherClassifier, Prediction

@pytest.fixture
def classifier(tmp_path):
    # Sample weather data:
    # [max temp, min temp, precipitation, wind speed]
    X = [
        [24, 15, 0, 10],    # Good
        [27, 18, 0, 12],    # Good
        [22, 12, 0, 8],     # Good
        [5, -2, 15, 50],    # Bad
        [8, 1, 10, 40],     # Bad
        [3, -5, 20, 55],    # Bad
    ]
    y = [1, 1, 1, 0, 0, 0]
    model = DecisionTreeClassifier(random_state=42)
    feature_names = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
    X = pd.DataFrame(X, columns=feature_names)
    model.fit(X, y)
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(model, model_path)
    return WeatherClassifier(model_path)
@pytest.mark.parametrize(
    "day, expected_label",
    [
        (
            {
                "temperature_2m_max": 24,
                "temperature_2m_min": 15,
                "precipitation_sum": 0,
                "wind_speed_10m_max": 10,
            },
            "good",
        ),
        (
            {
                "temperature_2m_max": 5,
                "temperature_2m_min": -2,
                "precipitation_sum": 15,
                "wind_speed_10m_max": 50,
            },
            "skip",
        ),
    ],
)
def test_weather_labels(classifier, day, expected_label):
    result = classifier.predict([day])
    assert result[0].label == expected_label
def test_probability_range(classifier):
    day = {
        "temperature_2m_max": 24,
        "temperature_2m_min": 15,
        "precipitation_sum": 0,
        "wind_speed_10m_max": 10,
    }
    result = classifier.predict([day])
    assert 0 <= result[0].probability <= 1
def test_multiple_days(classifier):
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
            "temperature_2m_max": 27,
            "temperature_2m_min": 18,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 12,
        },
    ]
    results = classifier.predict(days)
    assert len(results) == 3
    assert all(isinstance(result, Prediction) for result in results)
    assert [result.label for result in results] == ["good", "skip", "good"]
def test_empty_list(classifier):
    results = classifier.predict([])
    assert results == []
def test_missing_feature(classifier):
    day = {
        "temperature_2m_max": 24,
        "temperature_2m_min": 15,
        "precipitation_sum": 0,
        # wind_speed_10m_max is missing
    }
    with pytest.raises(ValueError, match="wind_speed_10m_max"):
        classifier.predict([day])
def test_missing_model_file(tmp_path):
    missing_path = tmp_path / "missing_model.pkl"
    with pytest.raises(FileNotFoundError, match="Weather model not found"):
        WeatherClassifier(missing_path) 