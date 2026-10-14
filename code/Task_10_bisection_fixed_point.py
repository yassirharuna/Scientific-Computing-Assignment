"""Task 10 - bisection and fixed-point iteration for a heat-balance equation."""
import os
import numpy as np
from scipy.optimize import root_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def heat_balance(T, ambient=25.0, gain=10.0, rate=20.0):
    return T - ambient - gain * np.exp(-T / rate)

def fixed_point(ambient=25.0, gain=10.0, rate=20.0, T0=30.0, tol=1e-10):
    g = lambda T: ambient + gain * np.exp(-T / rate)
    T = T0
    hist = [T]
    for _ in range(200):
        T_new = g(T)
        hist.append(T_new)
        if abs(T_new - T) < tol:
            break
        T = T_new
    return hist

# ---------- Part 1: original program ----------
sol = root_scalar(heat_balance, bracket=[20, 40], method="bisect", xtol=1e-10)
hist = fixed_point()
print("=== Part 1 ===")
print(f"Bisection equilibrium temperature: {sol.root:.6f} C  ({sol.iterations} iterations)")
print(f"Fixed-point equilibrium temperature: {hist[-1]:.6f} C  ({len(hist)-1} iterations)")

# ---------- Part 2: changed parameters ----------
# Increase the heat gain term from 10 to 30 and change the ambient temperature
print("\n=== Part 2: gain = 30 ===")
sol2 = root_scalar(lambda T: heat_balance(T, gain=30.0), bracket=[20, 60], method="bisect", xtol=1e-10)
hist2 = fixed_point(gain=30.0)
print(f"Bisection: {sol2.root:.6f} C   Fixed point: {hist2[-1]:.6f} C  ({len(hist2)-1} iterations)")

print("\nBisection iterations needed for different tolerances (bracket [20,40]):")
for tol in [1e-2, 1e-4, 1e-8, 1e-12]:
    s = root_scalar(heat_balance, bracket=[20, 40], method="bisect", xtol=tol)
    print(f"  xtol={tol:.0e}: {s.iterations} iterations, root={s.root:.10f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
temps = np.linspace(20, 40, 300)
ax[0].plot(temps, heat_balance(temps), label="Original (gain=10)")
ax[0].plot(temps, heat_balance(temps, gain=30.0), label="gain=30")
ax[0].axhline(0, color="k", linewidth=0.8)
ax[0].axvline(sol.root, linestyle="--", color="C0")
ax[0].axvline(sol2.root, linestyle="--", color="C1")
ax[0].set_xlabel("Temperature (C)")
ax[0].set_ylabel("Heat-balance function")
ax[0].set_title("Equilibrium temperature")
ax[0].legend()
ax[0].grid(True)
ax[1].plot(hist, marker="o", label="gain=10")
ax[1].plot(hist2, marker="s", label="gain=30")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("T estimate (C)")
ax[1].set_title("Fixed-point iterates")
ax[1].legend()
ax[1].grid(True)
fig.savefig(os.path.join(FIG, "task10_root_finding.png"), dpi=150, bbox_inches="tight")
