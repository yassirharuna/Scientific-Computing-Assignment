"""Task 8 - Jacobi and Gauss-Seidel iterative solvers."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def jacobi(A, b, tol=1e-8, max_iter=500):
    D = np.diag(A)
    R = A - np.diagflat(D)
    x = np.zeros_like(b)
    errors = []
    for _ in range(max_iter):
        x_new = (b - R @ x) / D
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x_new, errors
        x = x_new
    return x, errors

def gauss_seidel(A, b, tol=1e-8, max_iter=500):
    x = np.zeros_like(b)
    errors = []
    for _ in range(max_iter):
        old = x.copy()
        for i in range(len(b)):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ old[i+1:]) / A[i, i]
        error = np.linalg.norm(x - old, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x, errors
    return x, errors

# ---------- Part 1: original program ----------
A = np.array([[ 2., -1.,  0.,  0.,  0.],
              [-1.,  2., -1.,  0.,  0.],
              [ 0., -1.,  2., -1.,  0.],
              [ 0.,  0., -1.,  2., -1.],
              [ 0.,  0.,  0., -1.,  2.]])
b = np.array([100., 0., 0., 0., 20.])

x_j, err_j = jacobi(A, b)
x_gs, err_gs = gauss_seidel(A, b)
x_exact = np.linalg.solve(A, b)
print("=== Part 1 ===")
print("Jacobi temperatures:", x_j, f"({len(err_j)} iterations)")
print("Gauss-Seidel temperatures:", x_gs, f"({len(err_gs)} iterations)")
print("Direct solution:", x_exact)

# ---------- Part 2: changed parameters ----------
# tighter tolerance and a hotter left boundary
for tol in [1e-4, 1e-8, 1e-12]:
    _, ej = jacobi(A, b, tol=tol)
    _, eg = gauss_seidel(A, b, tol=tol)
    print(f"tol={tol:.0e}: Jacobi {len(ej)} its, Gauss-Seidel {len(eg)} its")

b2 = np.array([200., 0., 0., 0., 20.])
x2, e2 = gauss_seidel(A, b2)
print("Left boundary 200 C ->", np.round(x2, 3), f"({len(e2)} its)")

# A matrix that is NOT diagonally dominant: Jacobi should struggle
A_bad = np.array([[1., 3.], [4., 1.]])
b_bad = np.array([5., 6.])
xb, eb = jacobi(A_bad, b_bad, max_iter=30)
print("Non-diagonally-dominant system, Jacobi after 30 its:", xb, " last change:", eb[-1])
print("True solution:", np.linalg.solve(A_bad, b_bad))

plt.semilogy(err_j, label="Jacobi")
plt.semilogy(err_gs, label="Gauss-Seidel")
plt.xlabel("Iteration")
plt.ylabel("Maximum change")
plt.title("Task 8: convergence of iterative temperature solution")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(FIG, "task08_convergence.png"), dpi=150, bbox_inches="tight")
