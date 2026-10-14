"""Task 13 extension - filling a gap in river-level readings.

New case: a river gauge logs water level (m) every 6 hours over 3 days. Two readings
are missing because of a power cut. We estimate them with a cubic spline built from the
remaining points and compare with the true values (known here only because we
simulated them). Assumption: the level changes smoothly (no flash flood in the gap).
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

t = np.arange(0, 78, 6, dtype=float)                     # hours
level = 2.0 + 0.8 * np.sin(2 * np.pi * t / 48) + 0.002 * t   # smooth simulated level
missing = [4, 9]                                         # indices lost in the power cut
keep = np.setdiff1d(np.arange(t.size), missing)

spline = CubicSpline(t[keep], level[keep])
linear = interp1d(t[keep], level[keep])

for i in missing:
    print(f"t={t[i]:>4.0f} h: true={level[i]:.3f} m | spline={float(spline(t[i])):.3f} m | linear={float(linear(t[i])):.3f} m")

tt = np.linspace(0, 72, 400)
plt.plot(tt, spline(tt), label="Cubic spline (missing points removed)")
plt.scatter(t[keep], level[keep], color="k", label="Recorded readings")
plt.scatter(t[missing], level[missing], color="red", marker="x", s=80, label="Lost readings (true)")
plt.xlabel("Time (hours)")
plt.ylabel("Water level (m)")
plt.title("Task 13 extension: river gauge gap filling")
plt.legend(fontsize=8)
plt.grid(True)
plt.savefig(os.path.join(FIG, "task13_ext_river_gap.png"), dpi=150, bbox_inches="tight")
