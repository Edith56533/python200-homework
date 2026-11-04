from dataclasses import dataclass, FrozenInstanceError,field
from pydantic import BaseModel, Field, ValidationError, model_validator
import pytest

# --- Classes ---
# Question 1

class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location

        if readings is None:
            self.readings = []
        else:
            self.readings = readings

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

    def __repr__(self):
        return (
            f"Thermometer(location='{self.location}', "
            f"n_readings={len(self.readings)}, "
            f"average={self.average()})"
        )


my_thermometer = Thermometer("Eagle Pass")
my_thermometer.add(24.0)
my_thermometer.add(28.5)
my_thermometer.add(31.0)
my_thermometer.add(27.5)

print("Average:", my_thermometer.average())
print("Hottest:", my_thermometer.hottest())

# average() needs to handle an empty list because dividing by zero
# would raise an error if there were no readings.

second_thermometer = Thermometer(
    "San Antonio",
    [22.0, 23.5, 25.0, 24.0]
)

print(my_thermometer)
print([my_thermometer, second_thermometer])
#(Comment for Question two since it was added in the middle of question 1)
# Without __repr__, Python would display something like
# <__main__.Thermometer object at 0x...>, which is not useful for debugging because it does not show the object's actual data.

# Question 3

class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [
            reading
            for reading in thermometer.readings
            if reading > self.threshold
        ]


alert_30 = TemperatureAlert(30.0)
alert_25 = TemperatureAlert(25.0)

print(alert_30.breaches(my_thermometer))
print(alert_25.breaches(my_thermometer))

# The threshold is stored on TemperatureAlert so the same alert object can be reused for many thermometers without passing the threshold every time.This is useful if you need to check many thermometers using the same limit.


# --- Dataclasses, Type Hints, and Docstrings ---

# Question 1
@dataclass(frozen=True)
class Station:
    """Represents a weather station and its geographic information."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float

station_a = Station(
    "TX001",
    "Eagle Pass Station",
    28.7091,
    -100.4995,
    223.0
)

station_b = Station(
    "TX001",
    "Eagle Pass Station",
    28.7091,
    -100.4995,
    223.0
)
from dataclasses import FrozenInstanceError

try:
    station_a.name = "New Station Name"
except FrozenInstanceError as e:
    print(e)

station_c = Station(
    "TX002",
    "San Antonio Station",
    29.4241,
    -98.4936,
    198.0
)

station_set = {station_a, station_b, station_c}

print(len(station_set))

# frozen=True makes the dataclass immutable and also allows it to be hashable.That means Station objects can be used in sets, where duplicate objects with the same field values are automatically treated as the same item.

print(station_a == station_b)

# Dataclasses compare their field values automatically, so these two objects are equal because all of their fields match. With the original hand-written class, they would compare as different objects unless __eq__ was written manually.

# Question 3

@dataclass
class StationBatch:
    """Represents a group of weather stations in the same region."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add a station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the greatest elevation, or None if empty."""
        if not self.stations:
            return None

        return max(self.stations, key=lambda station: station.elevation)

#Error attained:
# Average: 27.75
# Hottest: 31.0
# Thermometer(location='Eagle Pass', n_readings=4, average=27.75)
# [Thermometer(location='Eagle Pass', n_readings=4, average=27.75), Thermometer(location='San Antonio', n_readings=4, average=23.625)]
# [31.0]
# [28.5, 31.0, 27.5]
# cannot assign to field 'name'
# 2
# True
# Traceback (most recent call last):
#   File "C:\Users\edith\CTD-repos\python200-homework\assignments_01\warmup_01.py", line 140, in <module>
#     @dataclass
#      ^^^^^^^^^
#   File "C:\Users\edith\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 1450, in dataclass
#     return wrap(cls)
#   File "C:\Users\edith\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 1440, in wrap
#     return _process_class(cls, init, repr, eq, order, unsafe_hash,
#                           frozen, match_args, kw_only, slots,
#                           weakref_slot)
#   File "C:\Users\edith\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 1066, in _process_class
#     cls_fields.append(_get_field(cls, name, type, kw_only))
#                       ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\edith\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 917, in _get_field
#     raise ValueError(f'mutable default {type(f.default)} for field '
#                      f'{f.name} is not allowed: use default_factory')
# ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory

# Error from using stations: list[Station] = []:
# ValueError: mutable default <class 'list'> for field stations is not allowed:
# use default_factory


# Python refuses a shared mutable list as a default because multiple StationBatch objects could accidentally share the same list.field(default_factory=list) creates a new empty list for each object.

batch = StationBatch("South Texas")

batch.add(station_a)
batch.add(station_c)

print(batch.highest())


# --- Pydantic ---

# Question 1
class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)
    @model_validator(mode="after")
    def validate_sensor_reading(self):
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("Invalid sensor reading")
        return self

valid_reading = Reading(
    station_id="TX001",
    timestamp="2026-10-12T20:00:00",
    temperature_c=27.5,
    humidity=45.0
)

print(valid_reading)

# Question 2

try:
    Reading(
        station_id="TX001",
        temperature_c=25.0,
        humidity=40.0
    )
except ValidationError as e:
    print(e)


try:
    Reading(
        station_id="TX001",
        timestamp="2026-10-12T21:00:00",
        temperature_c=150.0,
        humidity=40.0
    )
except ValidationError as e:
    print(e)

try:
    Reading(
        station_id="TX001",
        timestamp="2026-10-12T21:00:00",
        temperature_c=25.0,
        humidity="very humid"
    )
except ValidationError as e:
    print(e)

converted_reading = Reading(
    station_id="TX001",
    timestamp="2026-10-12T21:00:00",
    temperature_c="21.5",
    humidity=40
)

print(converted_reading)
print(type(converted_reading.temperature_c))
print(type(converted_reading.humidity))

# Pydantic accepts "21.5" because it can convert that string into a float. It rejects "very humid" because that string cannot be converted into a number.

# Question 3
try:
    Reading(
        station_id="TX",
        temperature_c="cold",
        humidity=50.0
    )
except ValidationError as e:
    for error in e.errors():
        print(error["loc"], error["msg"])

# Three errors are reported because station_id is too short,timestamp is missing, and temperature_c is not numeric.Reporting all errors at once is more useful because the user can fix everything in one pass instead of rerunning the program repeatedly.

# Question 4

valid_sensor_reading = Reading(
    station_id="TX001",
    timestamp="2026-10-12T22:00:00",
    temperature_c=-20.0,
    humidity=0.0
)

print(valid_sensor_reading)

try:
    Reading(
        station_id="TX001",
        timestamp="2026-10-12T22:00:00",
        temperature_c=-50.0,
        humidity=0.0
    )
except ValidationError as e:
    print(e)

# This rule cannot be expressed with Field constraints alone because it depends on the combination of two fields: humidity and temperature_c.

# --- pytest ---

# Question 1

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature to Fahrenheit."""
    return (celsius * 9 / 5) + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

# pytest.approx is needed because decimal calculations can have tiny floating-point differences, so exact equality is not always reliable.

# Question 2

def mean(values: list[float]) -> float:
    """Return the mean of a list of numbers."""
    if not values:
        raise ValueError("values cannot be empty")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

# pytest.raises(ValueError) would only check that a ValueError happened. match="empty" also checks that the error message contains the expected word.

# Question 3

@pytest.mark.parametrize(
    "values, expected",
    [
        ([5.0], 5.0),
        ([1.0, 2.0, 3.0], 2.0),
        ([-2.0, -4.0, -6.0], -4.0),
        ([10.0, 20.0, 30.0, 40.0], 25.0),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == expected


# A parametrized test is better than four nearly identical test functions because the testing logic is written once while still checking multiple cases.

# Summary: 6 passed in 0.21s

# Failure output:
# assert 257.0 == 212
# where 257.0 = celsius_to_fahrenheit(100)
# pytest showed both the actual value (257.0) and the expected value (212).That is more useful than an "assertion failed" because it shows exactly what result was wrong and by how much.