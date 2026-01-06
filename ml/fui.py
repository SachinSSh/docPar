## u have a 2d array of daily temperatures for multiple cities (rows: cities, columns: days) and a 1d array with a daily offset for each day of the week. 
## Use broadcasting to adjust each city's temperatures by the daily offset

import numpy as np

# Example data: 2D array of daily temperatures for multiple cities
# Shape: (cities, days) - let's say 3 cities, 14 days (2 weeks)
city_temperatures = np.random.randint(60, 90, size=(3, 14))

# 1D array with daily offsets for each day of the week (7 values)
# These represent adjustments for Monday, Tuesday, ..., Sunday
daily_offsets = np.array([-2, 0, 1, 2, 3, 1, -1])

# Map each day to its day-of-week index (0=Monday, 1=Tuesday, etc.)
# Assuming the first day in our data is a Monday
day_indices = np.array([i % 7 for i in range(city_temperatures.shape[1])])

# Use the day indices to select the appropriate offset for each day
# Shape becomes (14,) - one offset value for each day in our data
selected_offsets = daily_offsets[day_indices]

# Reshape to allow broadcasting with city_temperatures
# New shape: (1, 14) - one row (to broadcast across all cities), 14 columns (days)
selected_offsets_reshaped = selected_offsets.reshape(1, -1)

# Apply the offsets using broadcasting
# Broadcasting will replicate the 1×14 array across all cities
adjusted_temperatures = city_temperatures + selected_offsets_reshaped

# Print example results
print("Original city temperatures:")
print(city_temperatures)
print("\nDaily offsets by day of week:")
print(daily_offsets)
print("\nSelected offsets for each day in data:")
print(selected_offsets)
print("\nAdjusted temperatures:")
print(adjusted_temperatures)

# Verify the adjustment worked correctly for the first city and first few days
print("\nVerification for first city:")
for i in range(5):
    day_of_week = i % 7
    print(f"Day {i} (day of week {day_of_week}): {city_temperatures[0,i]} + {daily_offsets[day_of_week]} = {adjusted_temperatures[0,i]}")
