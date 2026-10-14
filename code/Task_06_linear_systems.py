"""Task 6 - solving Ax = b for two server types."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---------- Part 1: original program ----------
# x = Type A servers, y = Type B servers
# A uses 4 CPU, 8 GB;  B uses 2 CPU, 12 GB;  totals: 20 CPU, 72 GB
A = np.array([[4.0, 2.0], [8.0, 12.0]])
b = np.array([20.0, 72.0])

solution = np.linalg.solve(A, b)
residual = b - A @ solution
print("=== Part 1 ===")
print(f"Type A servers: {solution[0]:.0f}")
print(f"Type B servers: {solution[1]:.0f}")
print("Verification A @ solution:", A @ solution)
print("Residual:", residual)
print("Residual norm:", np.linalg.norm(residual))

# ---------- Part 2: changed input ----------
# Memory usage is now 80 GB instead of 72 GB
b2 = np.array([20.0, 80.0])
sol2 = np.linalg.solve(A, b2)
print("\n=== Part 2: memory total changed to 80 GB ===")
print("Solution:", sol2)

# A nearly singular case: Type B now looks almost like Type A
A3 = np.array([[4.0, 2.0], [8.0, 4.0001]])
print("Condition number original A:", np.linalg.cond(A))
print("Condition number near-singular A:", np.linalg.cond(A3))
print("Solution near-singular, b=[20,40]:", np.linalg.solve(A3, [20.0, 40.0]))
print("Solution near-singular, b=[20,40.01]:", np.linalg.solve(A3, [20.0, 40.01]))

# Graph: each equation is a line, the solution is where they meet
x = np.linspace(0, 8, 100)
plt.plot(x, (20 - 4 * x) / 2, label="CPU: 4x + 2y = 20")
plt.plot(x, (72 - 8 * x) / 12, label="Memory: 8x + 12y = 72")
plt.plot(solution[0], solution[1], "ko", label="Solution (3, 4)")
plt.xlim(0, 8); plt.ylim(0, 12)
plt.xlabel("Type A servers (x)")
plt.ylabel("Type B servers (y)")
plt.title("Task 6: two linear equations meeting at one point")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(FIG, "task06_linear_system.png"), dpi=150, bbox_inches="tight")
