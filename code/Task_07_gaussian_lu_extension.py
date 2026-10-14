"""Task 7 extension - steady-state heat in a metal rod with changing heat sources.

New case: a rod split into 8 interior segments. The coefficient matrix (tridiagonal,
from the finite-difference version of the heat equation) never changes, but the heat
source vector changes for every scenario. We factor once and solve for 4 scenarios.
Assumptions: both ends held at 0 C, unit spacing, source values in arbitrary units.
"""
import os
import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

n = 8
A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
lu, piv = lu_factor(A)

scenarios = {
    "Heater in the middle": np.array([0, 0, 0, 10, 10, 0, 0, 0], dtype=float),
    "Heater at left end":   np.array([10, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    "Even heating":         np.full(n, 3.0),
    "Two heaters":          np.array([0, 8, 0, 0, 0, 0, 8, 0], dtype=float),
}

x_axis = np.arange(1, n + 1)
for name, b in scenarios.items():
    T = lu_solve((lu, piv), b)
    print(f"{name}: peak temp {T.max():.2f} at segment {T.argmax()+1}; residual {np.linalg.norm(b - A @ T):.1e}")
    plt.plot(x_axis, T, marker="o", label=name)

plt.xlabel("Segment along the rod")
plt.ylabel("Temperature rise (arbitrary units)")
plt.title("Task 7 extension: rod temperature for 4 heat-source scenarios")
plt.legend(fontsize=8)
plt.grid(True)
plt.savefig(os.path.join(FIG, "task07_ext_rod_heating.png"), dpi=150, bbox_inches="tight")
