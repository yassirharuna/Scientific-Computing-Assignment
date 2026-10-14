"""Task 11 - Newton-Raphson and secant methods for an internal rate of return."""
import os
import numpy as np
from scipy.optimize import root_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def make_functions(cash_flows):
    def npv(rate):
        return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))
    def d_npv(rate):
        return sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows) if t > 0)
    return npv, d_npv

# ---------- Part 1: original program ----------
cash_flows = [-10000, 3000, 3500, 4000, 2500]
npv, d_npv = make_functions(cash_flows)
newton = root_scalar(npv, fprime=d_npv, x0=0.10, method="newton", xtol=1e-12)
secant = root_scalar(npv, x0=0.05, x1=0.20, method="secant", xtol=1e-12)
print("=== Part 1 ===")
print(f"IRR using Newton-Raphson: {newton.root:.6%}  ({newton.iterations} iterations)")
print(f"IRR using Secant method: {secant.root:.6%}  ({secant.iterations} iterations)")
print(f"NPV at Newton solution: {npv(newton.root):.6f}")

# ---------- Part 2: changed parameters ----------
# (a) different starting guesses
print("\n=== Part 2a: different starting guesses (Newton) ===")
for x0 in [0.01, 0.10, 0.50, 1.0]:
    try:
        r = root_scalar(npv, fprime=d_npv, x0=x0, method="newton", xtol=1e-12)
        print(f"x0={x0}: IRR={r.root:.6%}, iterations={r.iterations}")
    except Exception as e:
        print(f"x0={x0}: failed ({type(e).__name__})")

# (b) weaker cash flows
cf2 = [-10000, 2000, 2500, 3000, 2000]
npv2, d2 = make_functions(cf2)
r2 = root_scalar(npv2, fprime=d2, x0=0.10, method="newton", xtol=1e-12)
print(f"\n=== Part 2b: smaller inflows {cf2} ===")
print(f"IRR = {r2.root:.4%}")

# (c) iteration-by-iteration error comparison against the bisection answer
exact = root_scalar(npv, bracket=[0, 1], method="brentq", xtol=1e-14).root

def newton_hist(x, n=8):
    h = [abs(x - exact)]
    for _ in range(n):
        x = x - npv(x) / d_npv(x)
        h.append(abs(x - exact))
    return h

def secant_hist(x0, x1, n=8):
    h = [abs(x0 - exact), abs(x1 - exact)]
    for _ in range(n):
        x2 = x1 - npv(x1) * (x1 - x0) / (npv(x1) - npv(x0))
        x0, x1 = x1, x2
        h.append(abs(x1 - exact))
    return h

hn = newton_hist(0.10)
hs = secant_hist(0.05, 0.20)
bis = [abs((0 + 1) / 2 - exact)]
a, b = 0.0, 1.0
for _ in range(8):
    m = (a + b) / 2
    if npv(a) * npv(m) <= 0:
        b = m
    else:
        a = m
    bis.append(abs((a + b) / 2 - exact))
print("\nError after each step:")
print("Newton :", [f"{e:.1e}" for e in hn])
print("Secant :", [f"{e:.1e}" for e in hs])
print("Bisect :", [f"{e:.1e}" for e in bis])

plt.semilogy(hn, marker="o", label="Newton-Raphson")
plt.semilogy(hs, marker="s", label="Secant")
plt.semilogy(bis, marker="^", label="Bisection")
plt.xlabel("Iteration")
plt.ylabel("|error|")
plt.title("Task 11: speed of convergence (IRR problem)")
plt.legend()
plt.grid(True, which="both")
plt.savefig(os.path.join(FIG, "task11_convergence.png"), dpi=150, bbox_inches="tight")
