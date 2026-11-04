import json
from pathlib import Path
from weatherkit.schemas import WeatherResponse
from weatherkit.records import HourlyReading, to_readings

def load_response():
    json_path = Path(__file__).parent.parent / "weather_raw.json"

    with open(json_path, "r") as file:
        data = json.load(file)

    return WeatherResponse.model_validate(data)


def test_to_readings_returns_all_hours_in_order():
    response = load_response()
    readings = to_readings(response)

    assert len(readings) == 168
    assert readings[0].timestamp == response.hourly.time[0]
    assert readings[-1].timestamp == response.hourly.time[-1]


def test_reading_values_match_input_index():
    response = load_response()
    readings = to_readings(response)

    i = 10

    assert readings[i].timestamp == response.hourly.time[i]
    assert readings[i].temperature_c == response.hourly.temperature_2m[i]
    assert readings[i].precipitation_mm == response.hourly.precipitation[i]


def test_identical_hourly_readings_are_equal():
    reading_a = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )

    reading_b = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )

    assert reading_a == reading_b