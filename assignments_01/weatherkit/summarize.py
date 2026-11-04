from dataclasses import dataclass
from weatherkit.records import HourlyReading

@dataclass
class DailySummary:
    """Represents a summary of one day of weather observations."""
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between the daily high and low temperature."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Groups hourly weather readings into daily summaries."""

    def __init__(self, min_hours: int = 24):
        self.min_hours = min_hours

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Create daily summaries from hourly readings.

        Args:
            readings: Hourly weather readings.

        Returns:
            Daily summaries for dates with enough hourly observations.
        """
        grouped = {}

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in grouped:
                grouped[date] = []

            grouped[date].append(reading)

        summaries = []

        for date, daily_readings in grouped.items():
            if len(daily_readings) < self.min_hours:
                continue

            temperatures = [
                reading.temperature_c
                for reading in daily_readings
            ]

            precipitation = [
                reading.precipitation_mm
                for reading in daily_readings
            ]

            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(temperatures),
                    temp_min=min(temperatures),
                    precipitation_sum=sum(precipitation),
                    hours_observed=len(daily_readings),
                )
            )

        return sorted(summaries, key=lambda summary: summary.date)

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return dates that have fewer than the required number of readings."""
        grouped = {}

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in grouped:
                grouped[date] = 0

            grouped[date] += 1

        return sorted(
            date
            for date, count in grouped.items()
            if count < self.min_hours
        )