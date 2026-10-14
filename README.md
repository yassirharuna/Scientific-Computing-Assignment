# BIT 2118 Scientific Computing – Group Work

Student: _your name and registration number_
Group: _group name/number_

Python programs, figures and a written report for Tasks 2–20 of the BIT 2118 group work. For each task I ran the given program, changed an input to see what happens, and then wrote a new program for a related real-world case.

## Folders

| Folder | What is in it |
|---|---|
| `code/` | One script per task (`Task_XX_name.py`) and one for the new case (`Task_XX_name_extension.py`) |
| `figures/` | PNG graphs saved by the scripts |
| `docs/` | `Scientific_Computing_Report.md` – explanations, parameter changes, interpretations, new cases |
| `outputs/` | Text output of each script as it ran |

## Running it

```bash
pip install -r requirements.txt
python code/Task_08_iterative_linear.py     # any single task
./run_all.sh                                # everything
```

Scripts save their graphs into `figures/` instead of opening windows, so they also work on a machine without a display.

## Tasks and new cases

| Task | Topic | New case I added |
|---|---|---|
| 2 | NumPy arrays | Greenhouse soil-moisture sensors |
| 3 | Vectorization | Audio gain and clipping |
| 4 | Floating-point errors | Money sums and compound interest |
| 5 | Matrix algebra | Two-link robot arm |
| 6 | Linear systems | Balancing chemical equations |
| 7 | Gaussian elimination / LU | Heated metal rod, four heat-source scenarios |
| 8 | Iterative methods | 2-D heated plate |
| 9 | Eigenvalues | PCA of student marks |
| 10 | Bisection / fixed point | Bakery break-even |
| 11 | Newton / secant | Implied interest rate on a loan |
| 12 | Nonlinear equations | Ball thrown at a hillside |
| 13 | Interpolation | Filling a gap in river-level data |
| 14 | Curve fitting | Logistic growth of app subscribers |
| 15 | Numerical differentiation | Elevator acceleration |
| 16 | Numerical integration | Water pumped from a borehole |
| 17 | Euler's method | SIR epidemic in a school |
| 18 | `solve_ivp` | Predator–prey populations |
| 19 | Optimization | Minimum-cost animal feed |
| 20 | Monte Carlo | Project deadline risk and a clinic queue |

Task 1 (Data Science Essentials with Python) was done separately on the course platform.

## Report

The full write-up is in [`docs/Scientific_Computing_Report.md`](docs/Scientific_Computing_Report.md).
