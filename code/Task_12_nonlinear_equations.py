"""Task 12 - nonlinear systems: circle and parabola intersections."""
import os
import numpy as np
from scipy.optimize import root
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def make_eq(radius=2.0, shift=1.0):
    def equations(z):
        x, y = z
        return [x**2 + y**2 - radius**2, y - (x**2 - shift)]
    return equations

# ---------- Part 1: original program ----------
eq = make_eq()
print("=== Part 1: circle radius 2, parabola y = x^2 - 1 ===")
found = []
for guess in ([1.0, 1.0], [-1.0, 1.0]):
    s = root(eq, guess)
    found.append(s.x)
    print("Initial guess:", guess)
    print("Intersection:", s.x)
    print("Residuals:", eq(s.x))
    print("Converged:", s.success)
    print()

# polynomial in y: substituting x^2 = y + 1 gives y^2 + y - 3 = 0
print("Roots of y^2 + y - 3 = 0 (from substitution):", np.roots([1, 1, -3]))

# ---------- Part 2: changed parameters ----------
print("\n=== Part 2: circle radius 1, parabola y = x^2 - 1 ===")
eq2 = make_eq(radius=1.0, shift=1.0)
found2 = []
for guess in ([0.5, 0.5], [-0.5, 0.5], [0.0, -1.0], [3.0, 3.0]):
    s = root(eq2, guess)
    print(f"guess {guess}: x={s.x.round(4)}, converged={s.success}, residual norm={np.linalg.norm(eq2(s.x)):.1e}")
    if s.success:
        found2.append(s.x)

print("\n=== Part 2b: circle radius 2, parabola y = x^2 - 5 (no real intersection?) ===")
eq3 = make_eq(radius=2.0, shift=5.0)
s = root(eq3, [1.0, 1.0])
print("Converged:", s.success, "| message:", s.message[:60])

theta = np.linspace(0, 2 * np.pi, 400)
x = np.linspace(-2.5, 2.5, 400)
plt.plot(2 * np.cos(theta), 2 * np.sin(theta), label="Circle r=2")
plt.plot(x, x**2 - 1, label="y = x^2 - 1")
plt.plot(np.cos(theta), np.sin(theta), linestyle=":", label="Circle r=1")
for p in found:
    plt.plot(*p, "ko")
plt.gca().set_aspect("equal")
plt.ylim(-2.5, 3)
plt.legend(fontsize=8)
plt.grid(True)
plt.title("Task 12: circle-parabola intersections")
plt.savefig(os.path.join(FIG, "task12_intersections.png"), dpi=150, bbox_inches="tight")
