"""Task 20 - Monte Carlo estimate of server-overload probability."""
import os
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def simulate(capacity=1000.0, mean=850.0, std=120.0, N=100_000, seed=42):
    rng = np.random.default_rng(seed)
    demand = rng.normal(mean, std, N)
    over = demand > capacity
    return demand, over

# ---------- Part 1: original program ----------
demand, over = simulate()
exact = 1 - norm.cdf(1000, 850, 120)
print("=== Part 1 ===")
print(f"Estimated overload probability: {over.mean():.4%}")
print(f"Number of overload cases: {over.sum()} out of {len(over)}")
print(f"Exact probability from the normal distribution: {exact:.4%}")

# ---------- Part 2: changed parameters ----------
print("\n=== Part 2a: different numbers of simulations ===")
for N in [100, 1_000, 10_000, 100_000, 1_000_000]:
    _, o = simulate(N=N)
    print(f"N={N:>9,}: estimate={o.mean():.4%}  error={abs(o.mean()-exact):.4%}")

print("\n=== Part 2b: capacity 1100 and mean demand 900 ===")
for cap, mu in [(1100, 850), (1000, 900), (1000, 850)]:
    d, o = simulate(capacity=cap, mean=mu)
    ex = 1 - norm.cdf(cap, mu, 120)
    print(f"capacity={cap}, mean={mu}: simulated {o.mean():.3%}, exact {ex:.3%}")

print("\n=== Part 2c: spread of the estimate over 200 different seeds (N=10,000) ===")
ests = np.array([simulate(N=10_000, seed=s)[1].mean() for s in range(200)])
print(f"mean={ests.mean():.4%}, std={ests.std():.4%}, theoretical std={np.sqrt(exact*(1-exact)/10_000):.4%}")

N = len(over)
running = np.cumsum(over) / np.arange(1, N + 1)
idx = np.unique(np.logspace(1, np.log10(N), 200).astype(int)) - 1
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogx(idx + 1, running[idx], label="Running estimate")
ax[0].axhline(exact, color="r", linestyle="--", label="Exact")
ax[0].set_xlabel("Number of simulations")
ax[0].set_ylabel("Estimated overload probability")
ax[0].legend()
ax[0].grid(True)
ax[1].hist(demand, bins=60, alpha=0.8)
ax[1].axvline(1000, color="r", linestyle="--", label="Capacity")
ax[1].set_xlabel("Demand")
ax[1].set_ylabel("Count")
ax[1].legend()
fig.suptitle("Task 20: Monte Carlo estimate of server-overload risk")
fig.savefig(os.path.join(FIG, "task20_monte_carlo.png"), dpi=150, bbox_inches="tight")
