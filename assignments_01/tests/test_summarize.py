import pytest
from weatherkit.records import HourlyReading
from weatherkit.summarize import DailyAggregator


@pytest.fixture
def sample_readings():
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.0),
        HourlyReading("2026-04-08T01:00", 15.0, 0.5),
        HourlyReading("2026-04-08T02:00", 12.0, 0.2),
        HourlyReading("2026-04-09T00:00", 8.0, 0.0),
        HourlyReading("2026-04-09T01:00", 14.0, 0.3),
        HourlyReading("2026-04-09T02:00", 11.0, 0.0),
    ]


def test_grouping_produces_two_summaries(sample_readings):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2


@pytest.mark.parametrize(
    "summary_index, expected_max, expected_min",
    [
        (0, 15.0, 10.0),
        (1, 14.0, 8.0),
    ],
)
def test_temperature_max_and_min(
    sample_readings,
    summary_index,
    expected_max,
    expected_min,
):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[summary_index].temp_max == expected_max
    assert summaries[summary_index].temp_min == expected_min


def test_precipitation_sum(sample_readings):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(0.7)


def test_incomplete_day_is_dropped(sample_readings):
    aggregator = DailyAggregator(min_hours=4)

    summaries = aggregator.summarize(sample_readings)
    incomplete = aggregator.incomplete_days(sample_readings)

    assert summaries == []
    assert "2026-04-08" in incomplete
    assert "2026-04-09" in incomplete


def test_lower_min_hours_keeps_day(sample_readings):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    
# Deliberately changing temp_max to min(temperatures) caused test_temperature_max_and_min to fail, confirming the test catches the bug.