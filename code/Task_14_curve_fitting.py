"""Task 14 - curve fitting of server response time versus load."""
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

users = np.array([50, 100, 150, 200, 250, 300, 350], dtype=float)
response_ms = np.array([82, 95, 115, 145, 185, 238, 305], dtype=float)

def quad(x, a, b, c):
    return a * x**2 + b * x + c

def exp_model(x, a, b, c):
    return a * np.exp(b * x) + c

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))

# ---------- Part 1: original program ----------
p, cov = curve_fit(quad, users, response_ms)
a, b, c = p
pred = quad(users, *p)
print("=== Part 1: quadratic ===")
print(f"Fitted model: y = {a:.6f}x^2 + {b:.4f}x + {c:.2f}")
print(f"RMSE: {rmse(response_ms, pred):.2f} ms")
print(f"Predicted response at 400 users: {quad(400, *p):.2f} ms")
print("Parameter standard errors:", np.sqrt(np.diag(cov)).round(6))

# ---------- Part 2: changed model / data ----------
p2, _ = curve_fit(exp_model, users, response_ms, p0=[50, 0.005, 40], maxfev=10000)
pred2 = exp_model(users, *p2)
print("\n=== Part 2a: exponential model a*exp(bx)+c ===")
print(f"a={p2[0]:.3f}, b={p2[1]:.5f}, c={p2[2]:.2f}")
print(f"RMSE: {rmse(response_ms, pred2):.2f} ms")
print(f"Predicted at 400 users: {exp_model(400, *p2):.2f} ms")

lin = np.polyfit(users, response_ms, 1)
print("\n=== Part 2b: straight line ===")
print(f"y = {lin[0]:.4f}x + {lin[1]:.2f}, RMSE {rmse(response_ms, np.polyval(lin, users)):.2f} ms, at 400: {np.polyval(lin, 400):.2f} ms")

# (c) add one outlier measurement
resp_out = response_ms.copy()
resp_out[3] = 220
p3, _ = curve_fit(quad, users, resp_out)
print("\n=== Part 2c: one outlier (200 users -> 220 ms) ===")
print(f"Prediction at 400 users changes from {quad(400, *p):.1f} to {quad(400, *p3):.1f} ms")

x_plot = np.linspace(50, 400, 300)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(users, response_ms, color="k", label="Measurements")
ax[0].plot(x_plot, quad(x_plot, *p), label="Quadratic")
ax[0].plot(x_plot, exp_model(x_plot, *p2), label="Exponential")
ax[0].plot(x_plot, np.polyval(lin, x_plot), linestyle="--", label="Linear")
ax[0].set_xlabel("Concurrent users")
ax[0].set_ylabel("Response time (ms)")
ax[0].legend()
ax[0].grid(True)
ax[0].set_title("Fits")
ax[1].bar(users - 8, response_ms - pred, width=8, label="Quadratic")
ax[1].bar(users, response_ms - pred2, width=8, label="Exponential")
ax[1].bar(users + 8, response_ms - np.polyval(lin, users), width=8, label="Linear")
ax[1].axhline(0, color="k", linewidth=0.8)
ax[1].set_xlabel("Concurrent users")
ax[1].set_ylabel("Residual (ms)")
ax[1].set_title("Residuals")
ax[1].legend(fontsize=8)
fig.suptitle("Task 14: server response-time curve fitting")
fig.savefig(os.path.join(FIG, "task14_curve_fitting.png"), dpi=150, bbox_inches="tight")
