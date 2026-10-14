"""Task 19 extension - diet / feed mix at minimum cost (linear program with scipy).

New case: a farmer mixes maize bran and soya meal to feed pigs. Per kg:
    maize bran: costs 40 KES, protein 9 %, energy 3.0 Mcal
    soya meal : costs 95 KES, protein 44 %, energy 2.6 Mcal
The mix of 100 kg must have at least 18 kg protein and at least 285 Mcal energy.
Minimise cost. Assumptions: only two ingredients, prices fixed, nutrient values are averages.
"""
import numpy as np
from scipy.optimize import linprog

cost = [40.0, 95.0]               # KES per kg [maize, soya]
# constraints in the form A_ub x <= b_ub
A_ub = [[-0.09, -0.44],           # protein >= 18 kg
        [-3.0, -2.6]]             # energy  >= 290 Mcal
b_ub = [-18.0, -285.0]
A_eq = [[1, 1]]                   # total 100 kg
b_eq = [100.0]

res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * 2)
print("Success:", res.success)
print(f"Maize bran: {res.x[0]:.2f} kg, soya meal: {res.x[1]:.2f} kg")
print(f"Minimum cost: {res.fun:,.2f} KES per 100 kg")
print(f"Protein supplied: {0.09*res.x[0] + 0.44*res.x[1]:.2f} kg, energy: {3.0*res.x[0] + 2.6*res.x[1]:.1f} Mcal")

# first attempt used 290 Mcal, which turned out to be impossible: more soya
# lifts protein but lowers energy, so 18 kg protein and 290 Mcal can't both hold.
# change 1: soya price rises to 130
cost2 = [40.0, 130.0]
res2 = linprog(cost2, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * 2)
print(f"\nSoya at 130 KES: maize {res2.x[0]:.2f} kg, soya {res2.x[1]:.2f} kg, cost {res2.fun:,.2f} KES")

# change 2: higher protein target (20 kg) for young piglets
res3 = linprog(cost, A_ub=[[-0.09, -0.44], [-3.0, -2.6]], b_ub=[-20.0, -285.0],
               A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * 2)
print(f"Protein target 20 kg: maize {res3.x[0]:.2f} kg, soya {res3.x[1]:.2f} kg, cost {res3.fun:,.2f} KES")

# infeasible check
res4 = linprog(cost, A_ub=A_ub, b_ub=[-18.0, -290.0], A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * 2)
print("Energy target 290 Mcal feasible?", res4.success, "-", res4.message)
