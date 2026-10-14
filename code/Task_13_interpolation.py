"""Task 13 - linear vs cubic-spline interpolation of hourly temperatures."""
import os
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---------- Part 1: original program ----------
hours = np.array([8, 9, 10, 11, 12, 13, 14], dtype=float)
temperature = np.array([18.0, 20.0, 22.5, 24.0, 25.0, 25.5, 25.0])

linear = interp1d(hours, temperature, kind="linear")
cubic = CubicSpline(hours, temperature)

ten_minute_times = np.arange(8, 14 + 1/6, 1/6)
linear_estimates = linear(ten_minute_times)
cubic_estimates = cubic(ten_minute_times)

target_time = 10.5
print("=== Part 1 ===")
print(f"Linear estimate at 10:30: {float(linear(target_time)):.2f} C")
print(f"Cubic estimate at 10:30: {float(cubic(target_time)):.2f} C")
print(f"Largest gap between the two methods: {np.max(np.abs(linear_estimates - cubic_estimates)):.3f} C")

# ---------- Part 2: changed input ----------
# (a) hide one real reading and see which method recovers it better
print("\n=== Part 2a: leave out the 12:00 reading and estimate it ===")
keep = hours != 12
lin_miss = interp1d(hours[keep], temperature[keep])(12.0)
cub_miss = CubicSpline(hours[keep], temperature[keep])(12.0)
print(f"True value 25.0 C | linear {float(lin_miss):.2f} C (err {abs(lin_miss-25):.2f}) | cubic {float(cub_miss):.2f} C (err {abs(cub_miss-25):.2f})")

# (b) a sudden spike in the data
temp_spike = temperature.copy()
temp_spike[3] = 30.0   # sensor glitch at 11:00
cs2 = CubicSpline(hours, temp_spike)
print("\n=== Part 2b: spike of 30 C at 11:00 ===")
print(f"Cubic spline at 10:30: {float(cs2(10.5)):.2f} C, at 12:30: {float(cs2(12.5)):.2f} C")
print(f"Linear at 12:30: {float(interp1d(hours, temp_spike)(12.5)):.2f} C")
print(f"Spline minimum between 12 and 13: {cs2(np.linspace(12,13,200)).min():.2f} C (undershoot below both neighbours 25.0 and 25.5?)")

# (c) extrapolation warning
print("\nExtrapolating to 15:00 with the cubic spline:", round(float(cubic(15.0, extrapolate=True)), 2), "C")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(hours, temperature, color="k", zorder=3, label="Hourly measurements")
ax[0].plot(ten_minute_times, linear_estimates, label="Linear")
ax[0].plot(ten_minute_times, cubic_estimates, label="Cubic spline")
ax[0].set_title("Original data")
ax[1].scatter(hours, temp_spike, color="k", zorder=3, label="Measurements (with spike)")
tt = np.linspace(8, 14, 300)
ax[1].plot(tt, interp1d(hours, temp_spike)(tt), label="Linear")
ax[1].plot(tt, cs2(tt), label="Cubic spline")
ax[1].set_title("Spike at 11:00")
for a in ax:
    a.set_xlabel("Hour of day")
    a.set_ylabel("Temperature (C)")
    a.legend(fontsize=8)
    a.grid(True)
fig.suptitle("Task 13: interpolation")
fig.savefig(os.path.join(FIG, "task13_interpolation.png"), dpi=150, bbox_inches="tight")
