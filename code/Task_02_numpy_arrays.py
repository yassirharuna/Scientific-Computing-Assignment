"""Task 2 - NumPy arrays: daily temperature statistics + Celsius -> Fahrenheit."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---------- Part 1: original program ----------
# 24 hourly temperature readings in degrees Celsius
temperatures_c = np.array([
    18.5, 18.0, 17.8, 17.5, 17.9, 19.0,
    21.5, 23.8, 25.6, 27.1, 28.3, 29.0,
    29.4, 29.1, 28.5, 27.6, 26.2, 24.8,
    23.4, 22.1, 21.0, 20.2, 19.6, 19.0
])

mean_temp = np.mean(temperatures_c)
min_temp = np.min(temperatures_c)
max_temp = np.max(temperatures_c)

# Vectorized Celsius-to-Fahrenheit conversion
temperatures_f = 1.8 * temperatures_c + 32

print("=== Part 1: original data ===")
print(f"Daily mean temperature: {mean_temp:.2f} C")
print(f"Minimum temperature: {min_temp:.2f} C")
print(f"Maximum temperature: {max_temp:.2f} C")
print("First six Fahrenheit readings:", temperatures_f[:6])

# ---------- Part 2: changed input ----------
# What if the whole day was 5 C cooler (e.g. a cloudy day)?
cooler = temperatures_c - 5
print("\n=== Part 2: every reading lowered by 5 C ===")
print(f"Mean: {cooler.mean():.2f} C  (change {cooler.mean() - mean_temp:+.2f})")
print(f"Min: {cooler.min():.2f} C   Max: {cooler.max():.2f} C")
print(f"Range (max-min): original {max_temp-min_temp:.2f} C, cooler {cooler.max()-cooler.min():.2f} C")
print(f"Mean in Fahrenheit: original {temperatures_f.mean():.2f} F, cooler {(1.8*cooler+32).mean():.2f} F")

hours = np.arange(24)
plt.plot(hours, temperatures_c, marker="o", label="Original day")
plt.plot(hours, cooler, marker="s", linestyle="--", label="Day shifted by -5 C")
plt.axhline(mean_temp, color="grey", linewidth=0.8, label=f"Original mean ({mean_temp:.1f} C)")
plt.xlabel("Hour of day")
plt.ylabel("Temperature (C)")
plt.title("Task 2: hourly temperatures")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(FIG, "task02_temperatures.png"), dpi=150, bbox_inches="tight")
