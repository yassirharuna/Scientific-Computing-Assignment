"""Task 5 extension - robot arm: forward kinematics with rotation matrices.

New case: a planar robot arm with two links (L1 = 0.5 m, L2 = 0.3 m). Each joint
adds a rotation. The tip position is found by chaining rotation matrices.
Assumption: links are rigid and the base is fixed at the origin.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

L1, L2 = 0.5, 0.3
configs = [(30, 45), (60, -30), (90, 90), (0, 0)]

fig, ax = plt.subplots()
for t1_deg, t2_deg in configs:
    t1, t2 = np.deg2rad(t1_deg), np.deg2rad(t2_deg)
    elbow = rot(t1) @ np.array([L1, 0.0])
    tip = elbow + rot(t1 + t2) @ np.array([L2, 0.0])
    reach = np.linalg.norm(tip)
    print(f"theta1={t1_deg:>3}, theta2={t2_deg:>3}: elbow={elbow.round(3)}, tip={tip.round(3)}, reach={reach:.3f} m")
    ax.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], marker="o",
            label=f"({t1_deg}, {t2_deg})")

print("Maximum possible reach:", L1 + L2, "m")
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Task 5 extension: two-link robot arm")
ax.legend(title="joint angles (deg)", fontsize=8)
ax.grid(True)
fig.savefig(os.path.join(FIG, "task05_ext_robot_arm.png"), dpi=150, bbox_inches="tight")
