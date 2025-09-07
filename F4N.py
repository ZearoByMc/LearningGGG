# -*- coding: utf-8 -*-
"""
SCA on Griewank (f4) + Average/Std graph
---------------------------------------
1) Run 10 trials (D=30) -> show mean/std of minima
2) Run D=2 for visualization:
   - contour plot
   - agent slides (20 iterations)
   - convergence graph (iter 5..100)
3) Plot average minimum value with ± std deviation (over multiple runs)
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Benchmark: Griewank f4
# =========================
def griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    s = np.sum(x**2) / 4000.0
    i = np.arange(1, x.size + 1, dtype=float)
    p = np.prod(np.cos(x / np.sqrt(i)))
    return 1.0 + s - p


# =========================
# Sine Cosine Algorithm
# =========================
def sca(objective_function, lb, ub, dim, num_agents, max_iter, seed=None):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(lb, ub, size=(num_agents, dim))
    dest_pos = np.zeros(dim)
    dest_fitness = float("inf")

    convergence = np.zeros(max_iter)
    pop_history_2d = []

    a = 2.0
    for t in range(max_iter):
        positions = np.clip(positions, lb, ub)
        fitnesses = np.apply_along_axis(objective_function, 1, positions)

        idx = np.argmin(fitnesses)
        if fitnesses[idx] < dest_fitness:
            dest_fitness = float(fitnesses[idx])
            dest_pos = positions[idx].copy()

        convergence[t] = dest_fitness

        if positions.shape[1] >= 2 and t < 20:
            pop_history_2d.append(positions[:, :2].copy())

        r1 = a - t * (a / max_iter)
        for i in range(num_agents):
            for j in range(dim):
                r2 = rng.uniform(0, 2*np.pi)
                r3 = rng.uniform(0, 2)
                r4 = rng.uniform(0, 1)
                if r4 < 0.5:
                    positions[i, j] = positions[i, j] + r1*np.sin(r2)*abs(r3*dest_pos[j] - positions[i, j])
                else:
                    positions[i, j] = positions[i, j] + r1*np.cos(r2)*abs(r3*dest_pos[j] - positions[i, j])

    logs = {"convergence": convergence, "pop_history_2d": pop_history_2d}
    return dest_pos, dest_fitness, logs


# =========================
# Plot helpers
# =========================
def plot_griewank_contour_2d(xmin=-600, xmax=600, ymin=-600, ymax=600, levels=50, fname="f4_contour.png"):
    xs = np.linspace(xmin, xmax, 400)
    ys = np.linspace(ymin, ymax, 400)
    XX, YY = np.meshgrid(xs, ys)
    Z = 1 + (XX**2 + YY**2)/4000.0 - (np.cos(XX/np.sqrt(1)) * np.cos(YY/np.sqrt(2)))
    plt.figure(figsize=(7,6))
    cs = plt.contour(XX, YY, Z, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.title("Griewank f4 (D=2) Contour")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.scatter([0],[0], marker="*", s=120, label="global min (0,0)")
    plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=200); plt.close()


def save_agent_slides_on_contour(pop_history_2d, xmin=-600, xmax=600, ymin=-600, ymax=600, outdir="slides_f4"):
    os.makedirs(outdir, exist_ok=True)
    xs = np.linspace(xmin, xmax, 400)
    ys = np.linspace(ymin, ymax, 400)
    XX, YY = np.meshgrid(xs, ys)
    Z = 1 + (XX**2 + YY**2)/4000.0 - (np.cos(XX/np.sqrt(1)) * np.cos(YY/np.sqrt(2)))

    for it, pop in enumerate(pop_history_2d, start=1):
        plt.figure(figsize=(7,6))
        cs = plt.contour(XX, YY, Z, levels=50)
        plt.clabel(cs, inline=True, fontsize=8)
        plt.scatter(pop[:,0], pop[:,1], s=25, c=np.linspace(0,1,pop.shape[0]))
        plt.scatter([0],[0], marker="*", s=120, label="global min (0,0)")
        plt.title(f"Agent positions at iter {it}")
        plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
        plt.tight_layout()
        fname = os.path.join(outdir, f"f4_agents_iter_{it:02d}.png")
        plt.savefig(fname, dpi=200); plt.close()


def plot_convergence(curve, start_iter=5, end_iter=100, fname="f4_convergence.png"):
    n = len(curve)
    s = max(0, min(start_iter-1, n-1))
    e = max(1, min(end_iter, n))
    xs = np.arange(s+1, e+1)
    plt.figure(figsize=(7,5))
    plt.plot(xs, curve[s:e])
    plt.xlabel("Iteration"); plt.ylabel("Best objective value")
    plt.title("Convergence (best-so-far)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(fname, dpi=200); plt.close()


def plot_mean_std(mean_curve, std_curve, title="Average Minimum Value with Std", fname="f4_mean_std.png"):
    iterations = np.arange(len(mean_curve))
    plt.figure(figsize=(10,6))
    plt.plot(iterations, mean_curve, label="Average minimum value")
    plt.fill_between(iterations,
                     mean_curve-std_curve,
                     mean_curve+std_curve,
                     color="lightblue", alpha=0.5,
                     label="±1 std. deviation")
    plt.xlabel("Iterations"); plt.ylabel("Objective value")
    plt.title(title); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()


# =========================
# Experiment runners
# =========================
def run_multiple_trials(objective_function, lb, ub, dim, num_agents, max_iter, num_trials=10):
    all_curves = []
    final_mins = []

    for trial in range(num_trials):
        _, best_fitness, logs = sca(objective_function, lb, ub, dim, num_agents, max_iter, seed=1234+trial)
        all_curves.append(logs["convergence"])
        final_mins.append(best_fitness)
        print(f"[Trial {trial+1:02d}] best min = {best_fitness:.6e}")

    all_curves = np.array(all_curves)
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)

    print("\n=== Summary over runs ===")
    print(f"Mean final min = {np.mean(final_mins):.6e}")
    print(f"Std  final min = {np.std(final_mins, ddof=1):.6e}")

    return mean_curve, std_curve


def demo_plots_for_D2():
    D = 2
    lb, ub = -600.0, 600.0
    num_agents = 20
    max_iter = 120
    seed = 2025

    plot_griewank_contour_2d(xmin=lb, xmax=ub, ymin=lb, ymax=ub)
    _, best, logs = sca(griewank, lb, ub, D, num_agents, max_iter, seed=seed)
    print(f"[D=2 demo] best min = {best:.6e}")
    save_agent_slides_on_contour(logs["pop_history_2d"], xmin=lb, xmax=ub, ymin=lb, ymax=ub)
    plot_convergence(logs["convergence"], start_iter=5, end_iter=100)


# =========================
# Main
# =========================
if __name__ == "__main__":
    # 1) Run multiple trials (D=30)
    D = 30
    lb, ub = -600.0, 600.0
    num_agents = 20
    max_iter = 300
    num_trials = 10

    mean_curve, std_curve = run_multiple_trials(
        griewank, lb, ub, D, num_agents, max_iter, num_trials
    )
    plot_mean_std(mean_curve, std_curve, title="Griewank f4: Average Min ± Std")

    # 2) Visualization demo (D=2)
    demo_plots_for_D2()

    print("\nOutput files:")
    print(" - f4_contour.png")
    print(" - slides_f4/f4_agents_iter_01.png ... f4_agents_iter_20.png")
    print(" - f4_convergence.png")
    print(" - f4_mean_std.png")
