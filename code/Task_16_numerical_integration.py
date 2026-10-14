"""Task 16 - numerical integration: daily energy from smart-meter readings."""
import os
import numpy as np
from scipy.integrate import simpson, quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---------- Part 1: original program ----------
time_hours = np.array([0, 2, 4, 6, 8, 10, 12], dtype=float)
power_kw = np.array([0.8, 1.0, 1.4, 1.8, 1.6, 1.2, 0.9])

energy_trapezoid = np.trapezoid(power_kw, time_hours)
energy_simpson = simpson(power_kw, x=time_hours)
print("=== Part 1 ===")
print(f"Energy using trapezoidal rule: {energy_trapezoid:.3f} kWh")
print(f"Energy using Simpson's rule: {energy_simpson:.3f} kWh")
print(f"Difference between estimates: {abs(energy_simpson - energy_trapezoid):.3f} kWh")

# ---------- Part 2: changed parameters ----------
# (a) take readings every hour instead of every two hours (using a smooth curve through the same data)
from scipy.interpolate import CubicSpline
cs = CubicSpline(time_hours, power_kw)
t_fine = np.arange(0, 12.5, 1.0)
p_fine = cs(t_fine)
print("\n=== Part 2a: hourly readings (spline-interpolated) ===")
print(f"Trapezoid: {np.trapezoid(p_fine, t_fine):.3f} kWh, Simpson: {simpson(p_fine, x=t_fine):.3f} kWh")
exact_spline, _ = quad(cs, 0, 12)
print(f"Reference (adaptive quad of the spline): {exact_spline:.3f} kWh")

# (b) accuracy test on a function with a known integral: integral of sin(x) from 0 to pi = 2
print("\n=== Part 2b: accuracy test, integral of sin(x) on [0, pi] = 2 ===")
for n in [3, 5, 9, 17, 33]:
    x = np.linspace(0, np.pi, n)
    y = np.sin(x)
    print(f"n={n:>2}: trapezoid err={abs(np.trapezoid(y, x) - 2):.2e}   Simpson err={abs(simpson(y, x=x) - 2):.2e}")

# (c) a spike in demand
p_spike = power_kw.copy()
p_spike[3] = 3.5
print(f"\n=== Part 2c: 6:00 reading changed to 3.5 kW ===")
print(f"Trapezoid {np.trapezoid(p_spike, time_hours):.3f} kWh, Simpson {simpson(p_spike, x=time_hours):.3f} kWh")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(time_hours, power_kw, marker="o")
ax[0].fill_between(time_hours, power_kw, alpha=0.3)
ax[0].set_xlabel("Time (h)")
ax[0].set_ylabel("Power (kW)")
ax[0].set_title(f"Area = energy ({energy_trapezoid:.2f} kWh)")
ax[0].grid(True)
ns = np.array([3, 5, 9, 17, 33, 65])
et, es = [], []
for n in ns:
    x = np.linspace(0, np.pi, n)
    et.append(abs(np.trapezoid(np.sin(x), x) - 2))
    es.append(abs(simpson(np.sin(x), x=x) - 2) + 1e-17)
ax[1].loglog(ns, et, marker="o", label="Trapezoid")
ax[1].loglog(ns, es, marker="s", label="Simpson")
ax[1].set_xlabel("Number of points")
ax[1].set_ylabel("Absolute error")
ax[1].set_title("Accuracy on sin(x)")
ax[1].legend()
ax[1].grid(True, which="both")
fig.suptitle("Task 16: numerical integration")
fig.savefig(os.path.join(FIG, "task16_integration.png"), dpi=150, bbox_inches="tight")
