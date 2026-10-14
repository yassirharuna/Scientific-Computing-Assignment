"""Task 3 - vectorization vs. Python loops."""
import os
import numpy as np
import timeit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---------- Part 1: original program ----------
rng = np.random.default_rng(42)
latency_ms = rng.uniform(5, 250, 1_000_000)

def loop_version(data):
    squared = []
    for value in data:
        seconds = value / 1000.0
        squared.append(seconds ** 2)
    return squared

def vectorized_version(data):
    seconds = data / 1000.0
    return seconds ** 2

loop_time = timeit.timeit(lambda: loop_version(latency_ms), number=1)
vector_time = timeit.timeit(lambda: vectorized_version(latency_ms), number=3) / 3
result = vectorized_version(latency_ms)

print("=== Part 1: 1,000,000 values ===")
print("First five squared latency values:", result[:5])
print(f"Loop time: {loop_time:.4f} s")
print(f"Vectorized time: {vector_time:.4f} s")
print(f"Approximate speed-up: {loop_time/vector_time:.1f}x")

# ---------- Part 2: change the input size ----------
print("\n=== Part 2: changing the number of measurements ===")
sizes = [1_000, 10_000, 100_000, 1_000_000]
loop_times, vec_times = [], []
for n in sizes:
    data = latency_ms[:n]
    lt = timeit.timeit(lambda: loop_version(data), number=3) / 3
    vt = timeit.timeit(lambda: vectorized_version(data), number=20) / 20
    loop_times.append(lt)
    vec_times.append(vt)
    print(f"n={n:>9,}: loop={lt:.6f}s  vectorized={vt:.8f}s  speed-up={lt/vt:.1f}x")

plt.loglog(sizes, loop_times, marker="o", label="Python loop")
plt.loglog(sizes, vec_times, marker="s", label="NumPy vectorized")
plt.xlabel("Number of measurements")
plt.ylabel("Time (s)")
plt.title("Task 3: loop vs vectorized running time")
plt.legend()
plt.grid(True, which="both")
plt.savefig(os.path.join(FIG, "task03_timing.png"), dpi=150, bbox_inches="tight")
