import json
from dataclasses import dataclass
import pytest

# Question 1
class ModelInfo:
    def __init__(self, path):
        with open(path, "r") as file:
            self.data = json.load(file)
    def feature_count(self) -> int:
        return len(self.data["features"])

#sample JSON data
sample_data = {
    "features": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ]
}

with open("sample_model.json", "w") as file:
    json.dump(sample_data, file)

model = ModelInfo("sample_model.json")
print("Feature count:", model.feature_count())

# Loading the file in __init__ is more efficient becauseit only reads the file once, even if feature_count() is called multiple times.

# Question 2

@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool
    def summary(self) -> str:
        rain_status = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c}C, {rain_status}"
day1 = DayForecast("2026-11-03", 24.1, False)
day2 = DayForecast("2026-11-04", 18.5, True)
print(day1.summary())
print(day2.summary())
# @dataclass automatically generates __init__, so we don't have to write it again.

# Question 3

def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError("Invalid temperature input")
def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)
def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")

# Using match="temperature" checks the error message. Without it, pytest would only check the error type,potentially missing an incorrect message.