"""Task 17 - Euler's method for Newton's law of cooling."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def euler_cooling(T0=90.0, T_amb=25.0, k=0.12, h=1.0, end=30.0):
    t = np.arange(0, end + h, h)
    T = np.zeros_like(t)
    T[0] = T0
    for i in range(len(t) - 1):
        T[i + 1] = T[i] + h * (-k * (T[i] - T_amb))
    exact = T_amb + (T0 - T_amb) * np.exp(-k * t)
    return t, T, exact

# ---------- Part 1: original program ----------
t, T, exact = euler_cooling()
print("=== Part 1: h = 1 min ===")
print(f"Euler temperature after 30 min: {T[-1]:.2f} C")
print(f"Exact temperature after 30 min: {exact[-1]:.2f} C")
print(f"Final absolute error: {abs(T[-1] - exact[-1]):.4f} C")

# ---------- Part 2: changed parameters ----------
print("\n=== Part 2a: different step sizes ===")
runs = {}
for h in [5.0, 2.0, 1.0, 0.5, 0.1]:
    th, Th, ex = euler_cooling(h=h)
    runs[h] = (th, Th, ex)
    print(f"h={h:<4}: Euler={Th[-1]:.4f} C  error={abs(Th[-1]-ex[-1]):.4f} C")

print("\n=== Part 2b: faster cooling constant k = 0.3, h=1 vs 5 ===")
for h in [1.0, 5.0, 7.0]:
    th, Th, ex = euler_cooling(k=0.3, h=h)
    print(f"h={h}: final Euler={Th[-1]:.3f}, exact={ex[-1]:.3f}, min value reached={Th.min():.2f}")
print("(with k*h > 2 Euler becomes unstable - k*h = 2.1 for h=7)")

th5, Th5, ex5 = euler_cooling(k=0.3, h=7.0)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for h, (th, Th, ex) in runs.items():
    ax[0].plot(th, Th, label=f"Euler h={h}")
ax[0].plot(np.linspace(0, 30, 300), 25 + 65 * np.exp(-0.12 * np.linspace(0, 30, 300)), "k--", label="Exact")
ax[0].set_xlabel("Time (min)")
ax[0].set_ylabel("Temperature (C)")
ax[0].set_title("Effect of step size (k=0.12)")
ax[0].legend(fontsize=8)
ax[0].grid(True)
ax[1].plot(th5, Th5, marker="o", label="Euler h=7, k=0.3")
ax[1].plot(np.linspace(0, 30, 300), 25 + 65 * np.exp(-0.3 * np.linspace(0, 30, 300)), "k--", label="Exact")
ax[1].set_xlabel("Time (min)")
ax[1].set_ylabel("Temperature (C)")
ax[1].set_title("Instability when k*h > 2")
ax[1].legend(fontsize=8)
ax[1].grid(True)
fig.suptitle("Task 17: cooling of a hot object")
fig.savefig(os.path.join(FIG, "task17_euler.png"), dpi=150, bbox_inches="tight")
