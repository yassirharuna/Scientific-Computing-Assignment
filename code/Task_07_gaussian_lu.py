"""Task 7 - Gaussian elimination and LU decomposition."""
import os
import time
import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---------- Part 1: original program ----------
A = np.array([[4.0, -1.0, 0.0],
              [-1.0, 4.0, -1.0],
              [0.0, -1.0, 3.0]])
measurement_sets = [np.array([15.0, 10.0, 10.0]),
                    np.array([16.0, 11.0, 9.0]),
                    np.array([14.0, 12.0, 11.0])]

lu, piv = lu_factor(A)
for i, b in enumerate(measurement_sets, start=1):
    x = lu_solve((lu, piv), b)
    residual = b - A @ x
    print(f"Measurement set {i}")
    print("Solution:", x)
    print("Residual norm:", np.linalg.norm(residual))
    print()
print("Direct solution for first set:", np.linalg.solve(A, measurement_sets[0]))

# ---------- Part 2: changed input ----------
# (a) show the L and U factors, (b) change A's diagonal, (c) time LU reuse vs re-solving
from scipy.linalg import lu
P, L, U = lu(A)
print("\nL =\n", np.round(L, 4))
print("U =\n", np.round(U, 4))
print("L @ U equals P^T A ?", np.allclose(P @ L @ U, A))

A_b = A.copy()
A_b[2, 2] = 1.0   # weaker diagonal in the last row
print("\nChanged A[2,2] from 3 to 1; solution for set 1:", np.linalg.solve(A_b, measurement_sets[0]))

n = 600
rng = np.random.default_rng(0)
big = rng.random((n, n)) + n * np.eye(n)
rhs_list = [rng.random(n) for _ in range(200)]

t0 = time.perf_counter()
for r in rhs_list:
    np.linalg.solve(big, r)
t_solve = time.perf_counter() - t0

t0 = time.perf_counter()
f = lu_factor(big)
for r in rhs_list:
    lu_solve(f, r)
t_lu = time.perf_counter() - t0
print(f"\n200 right-hand sides, n={n}: solve each time={t_solve:.3f}s, factor once={t_lu:.3f}s, speed-up={t_solve/t_lu:.1f}x")

plt.bar(["np.linalg.solve\n(every time)", "LU factor once\n+ lu_solve"], [t_solve, t_lu], color=["tab:red", "tab:green"])
plt.ylabel("Time for 200 solves (s)")
plt.title("Task 7: cost of repeated solves (600 x 600)")
plt.savefig(os.path.join(FIG, "task07_lu_timing.png"), dpi=150, bbox_inches="tight")
