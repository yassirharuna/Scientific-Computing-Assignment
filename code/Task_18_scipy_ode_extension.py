"""Task 18 extension - predator-prey (Lotka-Volterra) wildlife model with solve_ivp.

New case: a park has gazelles (prey, x) and cheetahs (predators, y), in hundreds.
    dx/dt = a x - b x y
    dy/dt = -c y + d x y
a=0.8, b=0.4, c=0.6, d=0.3. Start with x=3, y=1.
Assumptions: unlimited food for prey, no other predators, no migration.
"""
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

a, b, c, d = 0.8, 0.4, 0.6, 0.3

def lv(t, z):
    x, y = z
    return [a * x - b * x * y, -c * y + d * x * y]

sol = solve_ivp(lv, (0, 60), [3.0, 1.0], t_eval=np.linspace(0, 60, 1500), rtol=1e-9, atol=1e-11)
x, y = sol.y
print("Success:", sol.success)
print(f"Prey range: {x.min():.2f} - {x.max():.2f} (hundreds)")
print(f"Predator range: {y.min():.2f} - {y.max():.2f} (hundreds)")
print(f"Equilibrium point: prey = {c/d:.2f}, predators = {a/b:.2f}")

# conserved quantity check (should stay constant for exact solution)
H = d * x - c * np.log(x) + b * y - a * np.log(y)
print(f"Drift in conserved quantity: {H.max() - H.min():.2e}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(sol.t, x, label="Prey")
ax[0].plot(sol.t, y, label="Predators")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Population (hundreds)")
ax[0].legend()
ax[0].grid(True)
ax[1].plot(x, y)
ax[1].plot(c / d, a / b, "ro", label="Equilibrium")
ax[1].set_xlabel("Prey")
ax[1].set_ylabel("Predators")
ax[1].set_title("Phase portrait")
ax[1].legend()
ax[1].grid(True)
fig.suptitle("Task 18 extension: predator-prey model")
fig.savefig(os.path.join(FIG, "task18_ext_predator_prey.png"), dpi=150, bbox_inches="tight")
