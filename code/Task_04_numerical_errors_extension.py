"""Task 4 extension - money and rounding in a banking calculation.

New case: a shop adds up 10,000 payments of 0.10 (Kenyan shillings). Using plain
floats the total drifts slightly from 1000.00. We compare naive summation,
math.fsum, np.sum and Python's Decimal type (the right tool for money).
"""
import math
from decimal import Decimal
import numpy as np

n = 10_000
payment = 0.10

naive = 0.0
for _ in range(n):
    naive += payment

print(f"Expected total      : {n * 0.1}")
print(f"Naive float loop    : {naive!r}")
print(f"Error of naive loop : {abs(naive - 1000):.3e}")
print(f"math.fsum           : {math.fsum([payment] * n)!r}")
print(f"np.sum              : {np.sum(np.full(n, payment))!r}")
print(f"Decimal             : {sum(Decimal('0.10') for _ in range(n))}")
print("Naive total == 1000 ?", naive == 1000)
print("np.isclose(naive, 1000) ?", np.isclose(naive, 1000))

# Interest compounding: 5,000,000 at 7 % for 30 years, float32 vs float64
p = 5_000_000
f32 = np.float32(p)
f64 = np.float64(p)
for _ in range(30):
    f32 = f32 * np.float32(1.07)
    f64 = f64 * 1.07
print(f"\nCompound interest after 30 years")
print(f"float64: {f64:,.2f}")
print(f"float32: {float(f32):,.2f}")
print(f"Difference: {abs(float(f32) - f64):,.2f}")
