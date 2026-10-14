"""Task 6 extension - balancing a chemical equation with linear algebra.

New case: balance  C3H8 + O2 -> CO2 + H2O  (propane combustion).
Unknown coefficients a, b, c, d for  a C3H8 + b O2 -> c CO2 + d H2O.
Atom balance:  C: 3a = c ;  H: 8a = 2d ;  O: 2b = 2c + d.
Fix a = 1 (balanced equations are only defined up to a scale factor) and solve
for b, c, d.
"""
import numpy as np

# unknown vector [b, c, d] with a = 1
A = np.array([
    [0.0, -1.0, 0.0],   # C: 3 = c      ->  -c = -3
    [0.0,  0.0, -2.0],  # H: 8 = 2d     ->  -2d = -8
    [2.0, -2.0, -1.0],  # O: 2b = 2c+d  ->  2b - 2c - d = 0
])
rhs = np.array([-3.0, -8.0, 0.0])

b, c, d = np.linalg.solve(A, rhs)
print(f"C3H8 + {b:g} O2 -> {c:g} CO2 + {d:g} H2O")
print("Residual:", rhs - A @ np.array([b, c, d]))

# Second case: iron + oxygen -> iron(III) oxide  (a Fe + b O2 -> c Fe2O3), a = 4
# Fe: a = 2c ; O: 2b = 3c  with a = 4
A2 = np.array([[0.0, -2.0], [2.0, -3.0]])
rhs2 = np.array([-4.0, 0.0])
b2, c2 = np.linalg.solve(A2, rhs2)
print(f"4 Fe + {b2:g} O2 -> {c2:g} Fe2O3")
