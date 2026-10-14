"""Task 12 extension - projectile hitting a hillside (kinematics).

New case: a ball is thrown at 20 m/s at 50 degrees. The ground rises as a straight
slope y = 0.3 x beyond the launch point. Where does it land? Equations:
    x = v cos(a) t
    y = v sin(a) t - 0.5 g t^2
    y = 0.3 x
Unknowns: t, x, y. Assumptions: no air resistance, g = 9.81 m/s^2.
"""
import numpy as np
from scipy.optimize import root

v, a, g, slope = 20.0, np.deg2rad(50), 9.81, 0.3

def eqs(z):
    t, x, y = z
    return [x - v * np.cos(a) * t,
            y - (v * np.sin(a) * t - 0.5 * g * t**2),
            y - slope * x]

s = root(eqs, [2.0, 20.0, 5.0])
t, x, y = s.x
print(f"Converged: {s.success}")
print(f"Time of flight: {t:.3f} s, landing x = {x:.2f} m, y = {y:.2f} m")
print("Residuals:", np.round(eqs(s.x), 10))

# Cross-check: polynomial in t -> v sin(a) t - 0.5 g t^2 = slope * v cos(a) t  (t != 0)
t_check = (v * np.sin(a) - slope * v * np.cos(a)) / (0.5 * g)
print(f"Analytical flight time: {t_check:.3f} s")

# Change the slope to a steeper 0.6
slope = 0.6
s2 = root(eqs, [2.0, 20.0, 5.0])
print(f"Steeper hill (0.6): t = {s2.x[0]:.3f} s, x = {s2.x[1]:.2f} m, y = {s2.x[2]:.2f} m")
