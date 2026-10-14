"""Task 19 - constrained optimization of server allocation cost."""
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def solve(cost_a=12.0, cost_b=18.0, cap_a=100.0, cap_b=170.0, required=1000.0):
    cost = lambda z: cost_a * z[0] + cost_b * z[1]
    cons = {"type": "ineq", "fun": lambda z: cap_a * z[0] + cap_b * z[1] - required}
    res = minimize(cost, x0=[5, 3], method="SLSQP", bounds=[(0, None), (0, None)], constraints=cons)
    best = None
    for a in range(0, 40):
        for b in range(0, 40):
            if cap_a * a + cap_b * b >= required:
                tc = cost_a * a + cost_b * b
                if best is None or tc < best[0]:
                    best = (tc, a, b, cap_a * a + cap_b * b)
    return res, best

# ---------- Part 1: original program ----------
res, best = solve()
x, y = res.x
print("=== Part 1 ===")
print("Optimization success:", res.success)
print(f"Type A allocation: {x:.3f}")
print(f"Type B allocation: {y:.3f}")
print(f"Total capacity: {100*x + 170*y:.2f}")
print(f"Minimum planning cost: {res.fun:.2f}")
print("Best nearby integer solution (cost, A, B, capacity):", best)
print(f"Cost per unit capacity: A = {12/100:.4f}, B = {18/170:.4f}")

# ---------- Part 2: changed parameters ----------
print("\n=== Part 2a: required capacity raised to 1500 ===")
res2, best2 = solve(required=1500.0)
print(f"Continuous: A={res2.x[0]:.3f}, B={res2.x[1]:.3f}, cost={res2.fun:.2f}")
print("Integer (cost, A, B, capacity):", best2)

print("\n=== Part 2b: Type B becomes more expensive (cost 30) ===")
res3, best3 = solve(cost_b=30.0)
print(f"Continuous: A={res3.x[0]:.3f}, B={res3.x[1]:.3f}, cost={res3.fun:.2f}")
print("Integer (cost, A, B, capacity):", best3)

print("\n=== Part 2c: different starting guesses ===")
for x0 in ([0, 0], [20, 20], [1, 10]):
    cost = lambda z: 12 * z[0] + 18 * z[1]
    cons = {"type": "ineq", "fun": lambda z: 100 * z[0] + 170 * z[1] - 1000}
    r = minimize(cost, x0=x0, method="SLSQP", bounds=[(0, None), (0, None)], constraints=cons)
    print(f"x0={x0}: A={r.x[0]:.3f}, B={r.x[1]:.3f}, cost={r.fun:.2f}, success={r.success}")

xs = np.linspace(0, 12, 200)
plt.fill_between(xs, np.maximum((1000 - 100 * xs) / 170, 0), 12, alpha=0.15, label="Feasible region")
plt.plot(xs, (1000 - 100 * xs) / 170, label="Capacity = 1000")
for c, ls in [(60, "--"), (100, "--"), (140, "--")]:
    plt.plot(xs, (c - 12 * xs) / 18, color="grey", linestyle=ls, linewidth=0.8)
plt.plot(*res.x, "ro", label=f"Optimum (cost {res.fun:.1f})")
plt.xlim(0, 12); plt.ylim(0, 10)
plt.xlabel("Type A units")
plt.ylabel("Type B units")
plt.title("Task 19: cost minimisation (grey = cost contours)")
plt.legend(fontsize=8)
plt.grid(True)
plt.savefig(os.path.join(FIG, "task19_optimization.png"), dpi=150, bbox_inches="tight")
