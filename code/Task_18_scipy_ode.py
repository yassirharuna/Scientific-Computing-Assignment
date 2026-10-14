"""Task 18 - solve_ivp for an RC charging circuit."""
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def run(Vs=5.0, R=1000.0, C=0.001, V0=0.0, t_end=5.0, rtol=1e-9, atol=1e-11):
    def rc(t, V):
        return [(Vs - V[0]) / (R * C)]
    t_eval = np.linspace(0, t_end, 200)
    sol = solve_ivp(rc, (0, t_end), [V0], t_eval=t_eval, rtol=rtol, atol=atol)
    exact = Vs + (V0 - Vs) * np.exp(-sol.t / (R * C))
    return sol, exact

# ---------- Part 1: original program ----------
sol, exact = run()
print("=== Part 1: R=1000, C=0.001 (tau = 1 s) ===")
print("Solver success:", sol.success)
print(f"Capacitor voltage at 5 s: {sol.y[0, -1]:.4f} V")
print(f"Maximum error against exact solution: {np.max(np.abs(sol.y[0] - exact)):.3e} V")

# ---------- Part 2: changed parameters ----------
print("\n=== Part 2a: different capacitances (time constant tau = R*C) ===")
curves = {}
for C in [0.0005, 0.001, 0.002]:
    s, e = run(C=C)
    tau = 1000 * C
    idx63 = np.argmax(s.y[0] >= 0.632 * 5)
    curves[C] = s
    print(f"C={C} F: tau={tau:.4f} s, V(5s)={s.y[0,-1]:.4f} V, reaches 63.2 % of 5 V at t={s.t[idx63]:.3f} s")

print("\n=== Part 2b: discharging (V0 = 5, Vs = 0) ===")
s, e = run(Vs=0.0, V0=5.0)
i1 = np.argmin(np.abs(s.t - 1))
print(f"V at t={s.t[i1]:.3f} s = {s.y[0, i1]:.3f} V (exact value there: {5*np.exp(-s.t[i1]):.3f} V, about 5*exp(-1) = {5*np.exp(-1):.3f} V)")

print("\n=== Part 2c: looser tolerance ===")
s_loose, e_loose = run(rtol=1e-3, atol=1e-6)
print(f"rtol=1e-3: max error {np.max(np.abs(s_loose.y[0]-e_loose)):.2e} V, function evaluations {s_loose.nfev}")
print(f"rtol=1e-9: function evaluations {sol.nfev}")

for C, s in curves.items():
    plt.plot(s.t, s.y[0], label=f"C = {C*1000:g} mF")
plt.axhline(5 * 0.632, color="grey", linestyle=":", label="63.2 % of Vs")
plt.xlabel("Time (s)")
plt.ylabel("Capacitor voltage (V)")
plt.title("Task 18: RC circuit charging")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(FIG, "task18_rc_circuit.png"), dpi=150, bbox_inches="tight")
