from dataclasses import dataclass

from weatherkit.schemas import WeatherResponse


@dataclass
class HourlyReading:
    """Represents one hourly weather reading.

    Temperature is in degrees Celsius.
    Precipitation is in millimeters.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a validated weather response into hourly reading objects.

    Args:
        response: A validated WeatherResponse object.

    Returns:
        A list of HourlyReading objects in the same order as the API data.
    """
    readings = []

    for timestamp, temperature, precipitation in zip(
        response.hourly.time,
        response.hourly.temperature_2m,
        response.hourly.precipitation,
    ):
        readings.append(
            HourlyReading(
                timestamp=timestamp,
                temperature_c=temperature,
                precipitation_mm=precipitation,
            )
        )

    return readings


# HourlyReading is a dataclass because the data has already been validated by WeatherResponse at the program boundary. Pydantic is used for incoming external data, while dataclasses are used for trusted internal data.