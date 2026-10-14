"""Task 9 extension - PCA on student marks using eigenvectors of the covariance matrix.

New case: 8 students, marks in 3 courses (Maths, Physics, Programming). PCA finds the
direction along which the marks vary most. Assumption: marks are out of 100 and
roughly linearly related; we centre the data first.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

marks = np.array([
    [82, 78, 90],
    [65, 60, 70],
    [90, 88, 95],
    [50, 55, 48],
    [72, 70, 75],
    [58, 52, 66],
    [95, 91, 85],
    [45, 40, 52],
], dtype=float)
courses = ["Maths", "Physics", "Programming"]

centred = marks - marks.mean(axis=0)
cov = np.cov(centred, rowvar=False)
vals, vecs = np.linalg.eigh(cov)          # eigh: symmetric matrix, sorted ascending
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]

# eigenvectors are only defined up to sign; flip so loadings are positive
for k in range(vecs.shape[1]):
    if vecs[:, k].sum() < 0:
        vecs[:, k] *= -1

explained = vals / vals.sum()
print("Eigenvalues:", np.round(vals, 2))
print("Explained variance ratio:", np.round(explained, 4))
print("First principal component (loadings):", {c: float(round(v, 3)) for c, v in zip(courses, vecs[:, 0])})

scores = centred @ vecs[:, :2]
print("PC1 scores (overall ability):", np.round(scores[:, 0], 1))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(["PC1", "PC2", "PC3"], explained * 100)
ax[0].set_ylabel("Variance explained (%)")
ax[0].set_title("Scree plot")
ax[1].scatter(scores[:, 0], scores[:, 1])
for i, (x, y) in enumerate(scores):
    ax[1].annotate(f"S{i+1}", (x, y), textcoords="offset points", xytext=(4, 4))
ax[1].set_xlabel("PC1")
ax[1].set_ylabel("PC2")
ax[1].set_title("Students in PC space")
ax[1].grid(True)
fig.suptitle("Task 9 extension: PCA of course marks")
fig.savefig(os.path.join(FIG, "task09_ext_pca.png"), dpi=150, bbox_inches="tight")
