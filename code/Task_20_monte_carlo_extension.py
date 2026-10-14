"""Task 20 extension - project-completion time risk and a queue simulation.

New case A: a small software project has 3 sequential tasks with uncertain durations
(triangular distributions in days). What is the chance it takes longer than 30 days?
New case B: a clinic with one doctor; patients arrive on average every 10 min and take
8 min on average (exponential). What does the average wait look like?
Assumptions: durations independent, FIFO queue, 8 hours per day.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
rng = np.random.default_rng(7)

# ---- A: project duration ----
N = 200_000
design = rng.triangular(5, 8, 14, N)
build = rng.triangular(8, 12, 22, N)
test = rng.triangular(3, 5, 10, N)
total = design + build + test
print(f"Mean duration: {total.mean():.2f} days, 90th percentile: {np.percentile(total, 90):.2f} days")
print(f"P(project > 30 days) = {(total > 30).mean():.3%}")
print(f"P(project > 25 days) = {(total > 25).mean():.3%}")

# ---- B: single-server queue ----
def clinic(mean_arr, mean_srv, n_patients=5000, seed=0):
    r = np.random.default_rng(seed)
    arrivals = np.cumsum(r.exponential(mean_arr, n_patients))
    service = r.exponential(mean_srv, n_patients)
    start = np.zeros(n_patients)
    finish = np.zeros(n_patients)
    for i in range(n_patients):
        start[i] = arrivals[i] if i == 0 else max(arrivals[i], finish[i - 1])
        finish[i] = start[i] + service[i]
    return start - arrivals

for mean_srv in [6, 8, 9, 9.5]:
    waits = np.mean([clinic(10, mean_srv, seed=s).mean() for s in range(20)])
    print(f"Average service time {mean_srv} min -> average wait {waits:.1f} min")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(total, bins=60)
ax[0].axvline(30, color="r", linestyle="--", label="30-day deadline")
ax[0].set_xlabel("Project duration (days)")
ax[0].set_ylabel("Count")
ax[0].legend()
srv = np.array([5, 6, 7, 8, 9, 9.5])
w = [np.mean([clinic(10, s, seed=k).mean() for k in range(10)]) for s in srv]
ax[1].plot(srv, w, marker="o")
ax[1].set_xlabel("Average service time (min)")
ax[1].set_ylabel("Average wait (min)")
ax[1].grid(True)
fig.suptitle("Task 20 extension: project risk and clinic queue")
fig.savefig(os.path.join(FIG, "task20_ext_project_queue.png"), dpi=150, bbox_inches="tight")
