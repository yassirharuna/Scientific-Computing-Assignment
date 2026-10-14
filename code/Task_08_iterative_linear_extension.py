"""Task 8 extension - 2-D heat plate (Laplace equation) solved with Gauss-Seidel.

New case: a 20 x 20 grid representing a square metal plate. Top edge = 100 C,
the other three edges = 0 C. Each interior cell becomes the average of its four
neighbours. This gives 324 unknowns - too many to want to factor by hand, and
a classic example where iteration is natural.
Assumptions: steady state, no internal heat source, uniform material.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

N = 20
T = np.zeros((N, N))
T[0, :] = 100.0   # hot top edge

errors = []
for it in range(5000):
    max_change = 0.0
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            new = 0.25 * (T[i-1, j] + T[i+1, j] + T[i, j-1] + T[i, j+1])
            max_change = max(max_change, abs(new - T[i, j]))
            T[i, j] = new
    errors.append(max_change)
    if max_change < 1e-6:
        break

print(f"Converged after {it+1} sweeps, final change {max_change:.2e}")
print(f"Temperature at plate centre: {T[N//2, N//2]:.2f} C")
print(f"Temperature one row below the hot edge, middle: {T[1, N//2]:.2f} C")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im = axes[0].imshow(T, cmap="inferno")
fig.colorbar(im, ax=axes[0], label="Temperature (C)")
axes[0].set_title("Steady-state plate temperature")
axes[1].semilogy(errors)
axes[1].set_xlabel("Sweep")
axes[1].set_ylabel("Maximum change")
axes[1].set_title("Gauss-Seidel convergence")
axes[1].grid(True)
fig.suptitle("Task 8 extension: 2-D heated plate")
fig.savefig(os.path.join(FIG, "task08_ext_heated_plate.png"), dpi=150, bbox_inches="tight")
