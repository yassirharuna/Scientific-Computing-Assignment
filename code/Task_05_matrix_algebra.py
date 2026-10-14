"""Task 5 - matrix algebra: rotate and scale a square."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def transform(points, angle_deg, sx, sy):
    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])
    scale = np.array([[sx, 0.0], [0.0, sy]])
    T = rotation @ scale          # scale first, then rotate
    return points @ T.T, T

# ---------- Part 1: original program ----------
points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
new_pts, T = transform(points, 45, 2.0, 1.5)
print("=== Part 1: 45 deg, scale (2.0, 1.5) ===")
print("Original points:\n", points)
print("Transformation matrix:\n", T)
print("Transformed points:\n", new_pts)
print("Determinant:", np.linalg.det(T))

# ---------- Part 2: changed parameters ----------
pts2, T2 = transform(points, 90, 1.0, 3.0)
print("\n=== Part 2: 90 deg, scale (1.0, 3.0) ===")
print("Transformed points:\n", np.round(pts2, 4))
print("Determinant:", np.linalg.det(T2))

# Reflection: determinant becomes negative
reflect = np.array([[1.0, 0.0], [0.0, -1.0]])
print("Determinant of reflection about x-axis:", np.linalg.det(reflect))

def closed(p):
    return np.vstack([p, p[0]])

fig, ax = plt.subplots()
for p, lab in [(points, "Original"), (new_pts, "45 deg, scale (2, 1.5)"), (pts2, "90 deg, scale (1, 3)")]:
    c = closed(p)
    ax.plot(c[:, 0], c[:, 1], marker="o", label=lab)
ax.set_aspect("equal")
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.set_title("Task 5: matrix transformations of a square")
ax.legend(fontsize=8)
ax.grid(True)
fig.savefig(os.path.join(FIG, "task05_transformations.png"), dpi=150, bbox_inches="tight")
