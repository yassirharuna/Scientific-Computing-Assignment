"""Task 15 - numerical differentiation of GPS position data."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

time = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
position = np.array([0.0, 4.8, 10.2, 16.1, 22.7, 30.0, 38.0])

def speed_estimate(t, x):
    v = np.zeros_like(x)
    v[0] = (x[1] - x[0]) / (t[1] - t[0])
    for i in range(1, len(t) - 1):
        v[i] = (x[i + 1] - x[i - 1]) / (t[i + 1] - t[i - 1])
    v[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
    return v

# ---------- Part 1: original program ----------
speed = speed_estimate(time, position)
print("=== Part 1 ===")
for t, x, v in zip(time, position, speed):
    print(f"t={t:.0f}s, position={x:.1f}m, estimated speed={v:.2f}m/s")
print("np.gradient check:", np.round(np.gradient(position, time), 2))

# ---------- Part 2: changed parameters ----------
# (a) step size study on a known function: x(t) = t^3, v(t) = 3t^2, at t = 2 -> 12
print("\n=== Part 2a: forward vs central difference on x(t)=t^3 at t=2 (true derivative 12) ===")
f = lambda t: t**3
for h in [1.0, 0.1, 0.01, 0.001, 1e-6, 1e-10, 1e-14]:
    fwd = (f(2 + h) - f(2)) / h
    cen = (f(2 + h) - f(2 - h)) / (2 * h)
    print(f"h={h:.0e}: forward err={abs(fwd-12):.2e}  central err={abs(cen-12):.2e}")

# (b) noisy GPS
rng = np.random.default_rng(3)
noisy = position + rng.normal(0, 0.3, position.size)
speed_noisy = speed_estimate(time, noisy)
print("\n=== Part 2b: 0.3 m of GPS noise ===")
print("Clean speeds:", np.round(speed, 2))
print("Noisy speeds:", np.round(speed_noisy, 2))
print("Max change in speed estimate:", np.round(np.max(np.abs(speed_noisy - speed)), 2), "m/s")

hs = np.logspace(-14, 0, 60)
fwd_err = np.abs((f(2 + hs) - f(2)) / hs - 12)
cen_err = np.abs((f(2 + hs) - f(2 - hs)) / (2 * hs) - 12)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(time, speed, marker="o", label="Clean")
ax[0].plot(time, speed_noisy, marker="s", linestyle="--", label="With noise")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Estimated speed (m/s)")
ax[0].set_title("Speed from GPS positions")
ax[0].legend()
ax[0].grid(True)
ax[1].loglog(hs, fwd_err, label="Forward")
ax[1].loglog(hs, cen_err, label="Central")
ax[1].set_xlabel("Step size h")
ax[1].set_ylabel("Absolute error")
ax[1].set_title("Error vs step size")
ax[1].legend()
ax[1].grid(True, which="both")
fig.savefig(os.path.join(FIG, "task15_differentiation.png"), dpi=150, bbox_inches="tight")
