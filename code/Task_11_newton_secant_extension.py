"""Task 11 extension - monthly payment of a loan: find the interest rate.

New case: a motorbike costs 150,000 KES. A dealer asks for 36 monthly payments of
5,200 KES. What monthly interest rate does that imply? We solve
    P = pmt * (1 - (1 + r)^-n) / r   for r  using the secant method (no derivative)
and then Newton. Assumption: payments at end of each month, no fees.
"""
from scipy.optimize import root_scalar

P, pmt, n = 150_000.0, 5_200.0, 36

def f(r):
    return pmt * (1 - (1 + r) ** (-n)) / r - P

def df(r, h=1e-8):
    return (f(r + h) - f(r - h)) / (2 * h)

sec = root_scalar(f, x0=0.005, x1=0.02, method="secant", xtol=1e-12)
new = root_scalar(f, fprime=df, x0=0.01, method="newton", xtol=1e-12)
print(f"Secant : monthly rate = {sec.root:.6%}  ({sec.iterations} iterations)")
print(f"Newton : monthly rate = {new.root:.6%}  ({new.iterations} iterations)")
print(f"Annual nominal rate (x12): {sec.root*12:.4%}")
print(f"Effective annual rate: {(1+sec.root)**12-1:.4%}")
print(f"Total repaid: {pmt*n:,.0f} KES, i.e. {pmt*n-P:,.0f} KES of interest")
