"""Task 4 - floating-point arithmetic and numerical errors."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---------- Part 1: original program ----------
calculated = 0.1 + 0.2
expected = 0.3
print("0.1 + 0.2 =", calculated)
print("Exact comparison:", calculated == expected)
print("Tolerance-based comparison:", np.isclose(calculated, expected))

true_value = np.sqrt(2)
approx_value = 1.414
absolute_error = abs(true_value - approx_value)
relative_error = absolute_error / abs(true_value)
print(f"Absolute error: {absolute_error:.8f}")
print(f"Relative error: {relative_error:.8%}")

print("\nUnstable vs stable formulation of (sqrt(1+x)-1)/x")
for x in [1e-4, 1e-8, 1e-12]:
    direct = (np.sqrt(1 + x) - 1) / x
    stable = 1 / (np.sqrt(1 + x) + 1)
    print(f"x={x:.0e}: direct={direct:.12f}, stable={stable:.12f}")

# ---------- Part 2: changed input ----------
# Use a closer approximation of sqrt(2) and push x much smaller.
print("\n=== Part 2: changed parameters ===")
for approx in [1.4, 1.41, 1.414, 1.41421, 1.4142136]:
    ae = abs(np.sqrt(2) - approx)
    print(f"approx={approx:<10} abs error={ae:.2e}  rel error={ae/np.sqrt(2):.2e}")

xs = np.logspace(-1, -16, 60)
direct = (np.sqrt(1 + xs) - 1) / xs
stable = 1 / (np.sqrt(1 + xs) + 1)
print("x=1e-16: direct =", (np.sqrt(1 + 1e-16) - 1) / 1e-16, " stable =", 1 / (np.sqrt(1 + 1e-16) + 1))

plt.semilogx(xs, np.abs(direct - stable) / stable, marker=".", label="Relative error of direct formula")
plt.gca().invert_xaxis()
plt.yscale("log")
plt.xlabel("x (decreasing to the right)")
plt.ylabel("Relative error")
plt.title("Task 4: catastrophic cancellation")
plt.legend()
plt.grid(True, which="both")
plt.savefig(os.path.join(FIG, "task04_cancellation.png"), dpi=150, bbox_inches="tight")
