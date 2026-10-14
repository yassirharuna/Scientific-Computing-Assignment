"""Task 9 - eigenvalues / eigenvectors and network centrality."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def centrality(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    index = np.argmax(np.real(eigenvalues))
    value = np.real(eigenvalues[index])
    vec = np.real(eigenvectors[:, index])
    c = np.abs(vec) / np.abs(vec).sum()
    err = np.linalg.norm(A @ vec - value * vec)
    return value, c, err, np.sort(np.real(eigenvalues))[::-1]

nodes = ["A", "B", "C", "D"]

# ---------- Part 1: original program ----------
A = np.array([[0., 1., 1., 0.],
              [1., 0., 1., 1.],
              [1., 1., 0., 1.],
              [0., 1., 1., 0.]])
val, c, err, allvals = centrality(A)
print("=== Part 1 ===")
print("All eigenvalues:", np.round(allvals, 4))
print("Dominant eigenvalue:", val)
for n, s in zip(nodes, c):
    print(f"{n}: {s:.4f}")
print("Verification error:", err)

# ---------- Part 2: changed input ----------
# Add a link between A and D so the network becomes a complete graph K4
A2 = A.copy()
A2[0, 3] = A2[3, 0] = 1.0
val2, c2, err2, allvals2 = centrality(A2)
print("\n=== Part 2: link A-D added (complete graph) ===")
print("All eigenvalues:", np.round(allvals2, 4))
print("Dominant eigenvalue:", val2)
for n, s in zip(nodes, c2):
    print(f"{n}: {s:.4f}")

# Remove the B-C link instead
A3 = A.copy()
A3[1, 2] = A3[2, 1] = 0.0
val3, c3, _, _ = centrality(A3)
print("\n=== Part 2b: link B-C removed ===")
print("Dominant eigenvalue:", val3)
for n, s in zip(nodes, c3):
    print(f"{n}: {s:.4f}")

x = np.arange(4)
w = 0.27
plt.bar(x - w, c, w, label="Original")
plt.bar(x, c2, w, label="A-D link added")
plt.bar(x + w, c3, w, label="B-C link removed")
plt.xticks(x, nodes)
plt.ylabel("Eigenvector centrality (normalised)")
plt.title("Task 9: node importance")
plt.legend()
plt.savefig(os.path.join(FIG, "task09_centrality.png"), dpi=150, bbox_inches="tight")
