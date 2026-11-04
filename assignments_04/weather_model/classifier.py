from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd

@dataclass
class Prediction:
    """Stores a weather prediction and its probability."""
    label: str
    probability: float
class WeatherClassifier:
    """Loads a trained weather model and makes predictions."""
    FEATURES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
    def __init__(self, model_path: str | Path):
        """Load the saved model.
        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Weather model not found: {model_path}"
            )
        self.model = joblib.load(model_path)
    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """Predict running conditions for a list of days.
        Raises:
            ValueError: If a day is missing required features.
        """
        if not days:
            return []
        for day in days:
            missing = [feature for feature in self.FEATURES if feature not in day]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
        inputs = pd.DataFrame(
            days, 
            columns=self.FEATURES
        )
        labels = self.model.predict(inputs)
        probabilities = self.model.predict_proba(inputs)[:, 1]
        return [
            Prediction(
# Mutation test: test weather labels caught an incorrect "skip" label.
                label="good" if label == 1 else "skip",
                probability=float(probability),
            )
            for label, probability in zip(labels, probabilities)
        ]       