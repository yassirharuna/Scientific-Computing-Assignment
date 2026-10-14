"""Task 14 extension - logistic growth of mobile-money subscribers.

New case: monthly number of subscribers (thousands) of a new mobile app. Growth
slows as the market saturates, so a logistic curve fits better than a straight line.
    N(t) = K / (1 + exp(-r (t - t0)))
Assumption: data are noise-free enough to fit; K is the carrying capacity.
"""
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

month = np.arange(1, 13, dtype=float)
subs = np.array([12, 20, 33, 52, 80, 118, 160, 200, 232, 255, 270, 281], dtype=float)

def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

p, cov = curve_fit(logistic, month, subs, p0=[300, 0.5, 6])
K, r, t0 = p
pred = logistic(month, *p)
rmse = np.sqrt(np.mean((subs - pred) ** 2))
ss_res = np.sum((subs - pred) ** 2)
ss_tot = np.sum((subs - subs.mean()) ** 2)
print(f"K = {K:.1f} thousand, r = {r:.3f} per month, midpoint t0 = {t0:.2f} months")
print(f"RMSE = {rmse:.2f} thousand, R^2 = {1 - ss_res/ss_tot:.4f}")
print(f"Forecast for month 18: {logistic(18, *p):.1f} thousand")

lin = np.polyfit(month, subs, 1)
print(f"A straight line would forecast month 18 as {np.polyval(lin, 18):.1f} thousand (above the market limit K)")

mm = np.linspace(1, 18, 200)
plt.scatter(month, subs, color="k", label="Observed")
plt.plot(mm, logistic(mm, *p), label="Logistic fit")
plt.plot(mm, np.polyval(lin, mm), linestyle="--", label="Linear fit")
plt.axhline(K, color="grey", linewidth=0.8)
plt.xlabel("Month")
plt.ylabel("Subscribers (thousands)")
plt.title("Task 14 extension: logistic growth")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(FIG, "task14_ext_logistic.png"), dpi=150, bbox_inches="tight")
