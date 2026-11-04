import json
from weatherkit.schemas import WeatherResponse
from weatherkit.records import to_readings
from weatherkit.summarize import DailyAggregator

def main():
    """Load, validate, summarize, and print the weather data."""
    with open("weather_raw.json", "r") as file:
        data = json.load(file)

    weather = WeatherResponse.model_validate(data)
    readings = to_readings(weather)

    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    print(
        f"{'Date':<12}"
        f"{'High':>8}"
        f"{'Low':>8}"
        f"{'Precip':>10}"
        f"{'Range':>10}"
    )

    for summary in summaries:
        print(
            f"{summary.date:<12}"
            f"{summary.temp_max:>8.1f}"
            f"{summary.temp_min:>8.1f}"
            f"{summary.precipitation_sum:>10.1f}"
            f"{summary.temp_range():>10.1f}"
        )

    if incomplete:
        print("Warning: incomplete days dropped:", ", ".join(incomplete))


if __name__ == "__main__":
    main()


# The __name__ == "__main__" guard prevents main() from running automatically if report.py is imported into another file. Without the guard, importing report.py to reuse one of its functions would also execute the whole report.


# --- Reflection ---

# 1. Rejecting the entire file for one null temperature can be useful when complete and trustworthy data is required, such as when generating reports that depend on every hourly value being present. In a live weather pipeline,though, I might prefer to tolerate an occasional missing reading so the rest of the day's data is still usable. To allow that, I would change temperature_2m from list[float] to list[float | None] and then handle missing values when converting or summarizing the data.

# 2. If the pipeline runs at noon, the current day only has about half of its hourly observations, so min_hours=24 would cause that day to be dropped even though the data is not actually bad. incomplete_days() helps because it makes those dropped dates visible instead of silently removing them.

# 3. Keeping weatherkit as a package makes it easier for a later pipeline to import only the pieces it needs, such as WeatherResponse, to_readings, or DailyAggregator, instead of copying code from one large script.