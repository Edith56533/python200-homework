import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit.schemas import WeatherResponse


def test_valid_response():
    json_path = Path(__file__).parent.parent / "weather_raw.json"

# A plain relative path is unreliable because pytest may be run from a different working directory.
    with open(json_path, "r") as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)

    assert len(response.hourly.time) == 168


def test_invalid_latitude_raises():
    data = {
        "latitude": 200.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [10.0],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_mismatched_lengths_raise():
    data = {
        "latitude": 35.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 100.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
                "2026-04-08T02:00",
            ],
            "temperature_2m": [10.0, 11.0],
            "precipitation": [0.0, 0.0, 0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_null_temperature_raises():
    data = {
        "latitude": 35.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [None],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)