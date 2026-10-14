"""Task 15 extension - acceleration of a lift (elevator) from velocity readings.

New case: an elevator logs its velocity (m/s) every 0.5 s while starting from rest,
cruising and stopping. Acceleration is the derivative of velocity. We use
central differences and flag moments that exceed a comfort limit of 1.0 m/s^2.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

t = np.arange(0, 10.5, 0.5)
v = np.array([0, 0.3, 0.9, 1.6, 2.3, 2.8, 3.0, 3.0, 3.0, 3.0, 3.0,
              3.0, 3.0, 3.0, 2.9, 2.5, 1.9, 1.2, 0.5, 0.1, 0.0])
a = np.gradient(v, t)       # central differences inside, one-sided at ends
for ti, vi, ai in zip(t, v, a):
    flag = "  <-- above comfort limit" if abs(ai) > 1.0 else ""
    print(f"t={ti:4.1f}s v={vi:.1f} m/s a={ai:+.2f} m/s^2{flag}")

print("Peak acceleration:", a.max().round(2), "m/s^2; peak deceleration:", a.min().round(2), "m/s^2")
dist = np.trapezoid(v, t)
print(f"Distance travelled (trapezoid): {dist:.2f} m")

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, v, marker="o")
ax[0].set_ylabel("Velocity (m/s)")
ax[0].grid(True)
ax[1].plot(t, a, marker="o", color="C1")
ax[1].axhline(1.0, color="r", linestyle="--")
ax[1].axhline(-1.0, color="r", linestyle="--")
ax[1].set_ylabel("Acceleration (m/s^2)")
ax[1].set_xlabel("Time (s)")
ax[1].grid(True)
fig.suptitle("Task 15 extension: elevator ride")
fig.savefig(os.path.join(FIG, "task15_ext_elevator.png"), dpi=150, bbox_inches="tight")
