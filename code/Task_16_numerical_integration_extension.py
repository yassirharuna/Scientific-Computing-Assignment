"""Task 16 extension - total water pumped from a borehole.

New case: a flow meter on a borehole reads flow rate (litres/minute) every 15 minutes
over 3 hours. Integrating gives the volume pumped. We also work out how much a
tank would hold if it overflows at 5,000 litres.
Assumption: flow varies smoothly between readings.
"""
import numpy as np
from scipy.integrate import simpson

minutes = np.arange(0, 181, 15, dtype=float)
flow_lpm = np.array([0, 22, 35, 40, 42, 41, 38, 36, 35, 33, 30, 24, 10], dtype=float)

trap = np.trapezoid(flow_lpm, minutes)
simp = simpson(flow_lpm, x=minutes)
print(f"Volume (trapezoid): {trap:.0f} litres")
print(f"Volume (Simpson):   {simp:.0f} litres")
print(f"Average flow: {trap/180:.1f} L/min")

cum = np.concatenate([[0], np.cumsum(0.5 * (flow_lpm[1:] + flow_lpm[:-1]) * np.diff(minutes))])
idx = np.argmax(cum >= 5000)
print(f"Tank (5,000 L) fills at about minute {minutes[idx]:.0f}" if cum.max() >= 5000 else "Tank never reaches 5,000 L")
print("Cumulative volume (L):", cum.round(0))
