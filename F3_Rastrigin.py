import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# =========================
# Benchmark: Griewank f4
# =========================

def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function (f3): sum[x_i^2 - 10 cos(2πx_i) + 10]."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0))


# =========================
# Sine Cosine Algorithm
# =========================
def sca(objective_function, lb, ub, dim, num_agents, max_iter, seed=None):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(lb, ub, size=(num_agents, dim))
    dest_pos = np.zeros(dim)
    dest_fitness = float("inf")

    # --- Logging variables ---
    convergence = np.zeros(max_iter)
    pop_history_2d = []
    avg_fitness_history = np.zeros(max_iter)
    first_agent_traj_x1 = np.zeros(max_iter)
    full_pop_history = []

    a = 2.0
    for t in range(max_iter):
        # Evaluate
        positions = np.clip(positions, lb, ub)
        fitnesses = np.apply_along_axis(objective_function, 1, positions)

        # --- Log data for this iteration ---
        avg_fitness_history[t] = np.mean(fitnesses)
        first_agent_traj_x1[t] = positions[0, 0]
        full_pop_history.append(positions.copy())

        # Update best
        idx = np.argmin(fitnesses)
        if fitnesses[idx] < dest_fitness:
            dest_fitness = float(fitnesses[idx])
            dest_pos = positions[idx].copy()

        convergence[t] = dest_fitness

        # Save for 2D slides (first 20 iterations)
        if dim >= 2 and t < 20:
            pop_history_2d.append(positions[:, :2].copy())

        # Move agents
        r1 = a - t * (a / max_iter)
        for i in range(num_agents):
            for j in range(dim):
                r2 = rng.uniform(0, 2 * np.pi)
                r3 = rng.uniform(0, 2)
                r4 = rng.uniform(0, 1)
                if r4 < 0.5:
                    positions[i, j] = positions[i, j] + r1 * np.sin(r2) * abs(r3 * dest_pos[j] - positions[i, j])
                else:
                    positions[i, j] = positions[i, j] + r1 * np.cos(r2) * abs(r3 * dest_pos[j] - positions[i, j])

    logs = {
        "convergence": convergence,
        "pop_history_2d": pop_history_2d,
        "avg_fitness": avg_fitness_history,
        "first_agent_traj_x1": first_agent_traj_x1,
        "full_pop_history": full_pop_history
    }
    return dest_pos, dest_fitness, logs

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

def plot_rastrigin_contour_2d(xmin=-5, xmax=5, ymin=-5, ymax=5, levels=15, fname="f3_contour.png"):
    xs = np.linspace(xmin, xmax, 400)
    ys = np.linspace(ymin, ymax, 400)
    X, Y = np.meshgrid(xs, ys)

    # Rastrigin f3 for D=2: sum over dimensions
    Z = (X**2 - 10.0 * np.cos(2.0 * np.pi * X) + 10.0) + \
        (Y**2 - 10.0 * np.cos(2.0 * np.pi * Y) + 10.0)

    plt.figure(figsize=(7, 6))
    cs = plt.contour(X, Y, Z, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.title("Rastrigin f3 (D=2) Contour")
    plt.xlabel("x1"); plt.ylabel("x2")
    # global minimum at (0, 0) with value 0
    plt.scatter([0], [0], marker="*", s=120, label="global min (0,0)")
    plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=200); plt.close()

def save_agent_slides_on_contour(pop_history_2d, xmin=-5, xmax=5, ymin=-5, ymax=5, outdir="slides_f3"):
    os.makedirs(outdir, exist_ok=True)
    xs = np.linspace(xmin, xmax, 400)
    ys = np.linspace(ymin, ymax, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = (X**2 - 10.0 * np.cos(2.0 * np.pi * X) + 10.0) + \
        (Y**2 - 10.0 * np.cos(2.0 * np.pi * Y) + 10.0)

    for it, pop in enumerate(pop_history_2d, start=1):
        plt.figure(figsize=(7,6))
        cs = plt.contour(X, Y, Z, levels=15)
        plt.clabel(cs, inline=True, fontsize=8)
        plt.scatter(pop[:,0], pop[:,1], s=25, c=np.linspace(0,1,pop.shape[0]))
        plt.scatter([0],[0], marker="*", s=120, label="global min (0,0)")
        plt.title(f"Agent positions at iter {it}")
        plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
        plt.tight_layout()
        fname = os.path.join(outdir, f"f3_agents_iter_{it:02d}.png")
        plt.savefig(fname, dpi=200); plt.close()


def plot_convergence(curve, start_iter=5, end_iter=100, fname="f3_convergence.png"):
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

def plot_search_history(history, x_range=(-2, 2), y_range=(-2, 2), fname="F3_search_history.png"):
    """ 1. Plots the trajectory of all search agents on the contour plot (first 100 iters). """
    xs = np.linspace(x_range[0], x_range[1], 400)
    ys = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(xs, ys)
    Z = (X**2 - 10.0 * np.cos(2.0 * np.pi * X) + 10.0) + \
        (Y**2 - 10.0 * np.cos(2.0 * np.pi * Y) + 10.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    
    # Slice history to the first 100 iterations
    history_sliced = history[:100]
    agent_trajectories = np.array(history_sliced).transpose(1, 0, 2)
    
    for agent_path in agent_trajectories:
        points = agent_path.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, color='red', alpha=0.3, linewidth=1)
        ax.add_collection(lc)

    ax.scatter([1], [1], marker="*", s=150, c='gold', edgecolors='black', zorder=5, label="Global Min (0,0)")
    ax.set_title("Search History of All Agents (First 100 Iterations)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.legend()
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()

def plot_first_agent_trajectory(trajectory, fname="first_agent_trajectory.png"):
    """ 2. Plots the trajectory of x1 for the first agent (first 100 iters). """
    plt.figure(figsize=(8, 5))
    # Slice trajectory to the first 100 iterations
    plt.plot(trajectory[:100])
    plt.xlabel("Iteration")
    plt.ylabel("Value of x1 for the first agent")
    plt.title("Trajectory of the First Variable (x1) of the First Agent (First 100 Iterations)")
    plt.grid(True)
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()

def plot_average_fitness(avg_fitness, fname="average_fitness.png"):
    """ 3. Plots the average fitness of all search agents (first 100 iters). """
    plt.figure(figsize=(8, 5))
    # Slice fitness data to the first 100 iterations
    plt.plot(avg_fitness[:100])
    plt.xlabel("Iteration")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness of Search Agents (First 100 Iterations)")
    plt.grid(True)
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()

def plot_full_convergence_curve(curve, out="convergence_curve.png"):
    """ 4. Plots the convergence curve (first 100 iters). """
    plt.figure(figsize=(8, 5))
    # Slice convergence data to the first 100 iterations
    plt.plot(curve[:100])
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Value (Fitness)")
    plt.title("Convergence Curve (First 100 Iterations)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def demo_plots_for_D2():
    D = 2
    lb, ub = -5.0, 5.0
    num_agents = 20
    max_iter = 120
    seed = 2025

    _, best, logs = sca(rastrigin, lb, ub, D, num_agents, max_iter, seed=seed)
    print(f"[D=2 demo] best min = {best:.6e}")
    plot_rastrigin_contour_2d(xmin=lb, xmax=ub, ymin=lb, ymax=ub)

    plot_convergence(logs["convergence"], start_iter=5, end_iter=100)

    save_agent_slides_on_contour(logs["pop_history_2d"], xmin=lb, xmax=ub, ymin=lb, ymax=ub)

    plot_search_history(logs["pop_history_2d"], x_range=(lb, ub), y_range=(lb, ub), fname="F3_search_history.png")

    plot_first_agent_trajectory(logs["first_agent_traj_x1"], fname="F3_first_agent_trajectory.png")

    plot_average_fitness(logs["avg_fitness"],fname="F3_average_fitness.png")

    plot_full_convergence_curve(logs["convergence"], "F3_convergence_curve.png")

if __name__ == "__main__":
    # 1) Run multiple trials (D=30)
    D = 20
    lb, ub = -5.0, 5.0
    num_agents = 20
    max_iter = 300
    num_trials = 10

    mean_curve, std_curve = run_multiple_trials(
        rastrigin, lb, ub, D, num_agents, max_iter, num_trials
    )

    demo_plots_for_D2()

