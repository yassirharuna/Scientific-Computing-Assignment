"""Task 17 extension - SIR epidemic model with Euler's method.

New case: a school of 1,000 pupils, 1 infected at the start.
    dS/dt = -beta S I / N
    dI/dt =  beta S I / N - gamma I
    dR/dt =  gamma I
beta = 0.4 per day (contact rate), gamma = 0.1 per day (recovery, 10-day illness).
R0 = beta/gamma = 4. We then test a control measure that cuts beta to 0.2.
Assumptions: closed population, no re-infection, h = 0.5 day.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def sir(beta, gamma=0.1, N=1000.0, days=120, h=0.5):
    n = int(days / h) + 1
    S = np.zeros(n); I = np.zeros(n); R = np.zeros(n)
    S[0], I[0] = N - 1, 1
    for k in range(n - 1):
        new_inf = beta * S[k] * I[k] / N
        rec = gamma * I[k]
        S[k + 1] = S[k] - h * new_inf
        I[k + 1] = I[k] + h * (new_inf - rec)
        R[k + 1] = R[k] + h * rec
    return np.arange(n) * h, S, I, R

for beta in [0.4, 0.2, 0.12]:
    t, S, I, R = sir(beta)
    print(f"beta={beta}: R0={beta/0.1:.1f}, peak infected={I.max():.0f} on day {t[I.argmax()]:.0f}, total ever infected={R[-1]:.0f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
t, S, I, R = sir(0.4)
ax[0].plot(t, S, label="Susceptible")
ax[0].plot(t, I, label="Infected")
ax[0].plot(t, R, label="Recovered")
ax[0].set_title("beta = 0.4")
for beta in [0.4, 0.2, 0.12]:
    t, S, I, R = sir(beta)
    ax[1].plot(t, I, label=f"beta={beta}")
ax[1].set_title("Infected pupils for different contact rates")
for a in ax:
    a.set_xlabel("Day")
    a.set_ylabel("Pupils")
    a.legend()
    a.grid(True)
fig.suptitle("Task 17 extension: SIR model")
fig.savefig(os.path.join(FIG, "task17_ext_sir.png"), dpi=150, bbox_inches="tight")
