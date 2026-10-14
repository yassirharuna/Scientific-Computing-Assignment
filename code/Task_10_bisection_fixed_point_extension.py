"""Task 10 extension - break-even point for a small business.

New case: a bakery sells loaves at 60 KES each. Fixed monthly costs are 45,000 KES.
Variable cost per loaf falls with volume (bulk flour discount):
    cost per loaf = 35 - 0.002 * q   (q = loaves per month)
Profit(q) = 60 q - (45000 + (35 - 0.002 q) q). Find the break-even quantity
with bisection. Assumption: prices are constant and everything produced is sold.
"""
import os
import numpy as np
from scipy.optimize import root_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

price, fixed = 60.0, 45_000.0

def profit(q, price=price):
    cost = fixed + (35.0 - 0.002 * q) * q
    return price * q - cost

sol = root_scalar(profit, bracket=[100, 5000], method="bisect", xtol=1e-6)
print(f"Break-even quantity: {sol.root:.2f} loaves/month ({sol.iterations} iterations)")

# What if the price drops to 55?
sol55 = root_scalar(lambda q: profit(q, 55.0), bracket=[100, 20000], method="bisect", xtol=1e-6)
print(f"Break-even at price 55: {sol55.root:.2f} loaves/month")

q = np.linspace(0, 4000, 400)
plt.plot(q, profit(q), label="Price 60 KES")
plt.plot(q, profit(q, 55.0), label="Price 55 KES")
plt.axhline(0, color="k", linewidth=0.8)
plt.axvline(sol.root, linestyle="--", color="C0")
plt.xlabel("Loaves sold per month")
plt.ylabel("Profit (KES)")
plt.title("Task 10 extension: bakery break-even")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(FIG, "task10_ext_break_even.png"), dpi=150, bbox_inches="tight")
