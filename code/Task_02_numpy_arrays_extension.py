"""Task 2 extension - soil-moisture sensors in a greenhouse (IoT).

New case: 6 sensors report volumetric soil moisture (%) every 4 hours for one day.
We store the readings in a 2-D array (sensors x time) and use axis-wise statistics
to flag sensors that look faulty or beds that need watering.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

# Assumptions: readings are % volumetric water content, crops need >= 30 %.
# Sensor 5 is deliberately given a drifting, suspicious signal.
times = np.array([0, 4, 8, 12, 16, 20])
moisture = np.array([
    [38, 36, 33, 29, 34, 37],   # bed 1
    [41, 40, 37, 33, 36, 40],   # bed 2
    [35, 33, 30, 26, 29, 33],   # bed 3
    [44, 43, 40, 37, 39, 43],   # bed 4
    [40, 31, 22, 14,  9,  4],   # bed 5 (drying fast / sensor drifting?)
    [37, 35, 32, 28, 31, 35],   # bed 6
], dtype=float)

print("Mean per bed (%):", moisture.mean(axis=1).round(2))
print("Lowest reading per bed (%):", moisture.min(axis=1))
print("Mean across beds at each time (%):", moisture.mean(axis=0).round(2))

needs_water = moisture.min(axis=1) < 30
print("Beds that dipped below 30 %:", np.where(needs_water)[0] + 1)

# Spread over the day - a very large spread points to a faulty sensor or a leak
spread = moisture.max(axis=1) - moisture.min(axis=1)
print("Daily spread per bed (%):", spread)
print("Possible faulty sensor (spread > 20):", np.where(spread > 20)[0] + 1)

for i, row in enumerate(moisture, start=1):
    plt.plot(times, row, marker="o", label=f"Bed {i}")
plt.axhline(30, color="red", linestyle="--", label="30 % threshold")
plt.xlabel("Hour of day")
plt.ylabel("Soil moisture (%)")
plt.title("Task 2 extension: greenhouse soil moisture")
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.savefig(os.path.join(FIG, "task02_ext_soil_moisture.png"), dpi=150, bbox_inches="tight")
