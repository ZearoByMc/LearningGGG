# -*- coding: utf-8 -*-
"""
SCA on Griewank (f4) — ตามโจทย์:
1) รัน 10 ครั้งที่ D=30 ในโดเมน [-600, 600]^D แล้วสรุป mean/std ของ best minimum
2) วาด contour plot (D=2)
3) ทำสไลด์โชว์ตำแหน่งเอเจนต์ (20 ตัว) 20 iteration แรกบน contour (D=2)
4) วาดกราฟ convergence ช่วง iteration 5..100

หมายเหตุ:
- ปรับพารามิเตอร์ SCA ได้ที่บล็อค CONFIG
- หากคุณมีอัลกอริทึมของตัวเอง ให้แทนที่ฟังก์ชัน sca(...) ด้วยของคุณได้เลย
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Benchmark: Griewank f4
# =========================
def griewank(x: np.ndarray) -> float:
    """
    Griewank function (min = 0 at x=0)
    f(x) = 1 + (1/4000)*sum(x_i^2) - prod(cos(x_i / sqrt(i)))
    Domain (โจทย์): [-600, 600]^D
    """
    x = np.asarray(x, dtype=float)
    s = np.sum(x**2) / 4000.0
    # i index starts at 1 for the product term
    i = np.arange(1, x.size + 1, dtype=float)
    p = np.prod(np.cos(x / np.sqrt(i)))
    return 1.0 + s - p


# =========================
# Sine Cosine Algorithm
# =========================
def sca(objective_function, lb, ub, dim, num_agents, max_iter, seed=None):
    """
    SCA (minimization) + เก็บโลกรวบรวมสำหรับการ plot

    Returns
    -------
    dest_pos : (dim,)
    dest_fitness : float
    logs : dict with keys:
        - convergence: (max_iter,) best-so-far per iter
        - pop_history_2d: list length <= first_k_iters, each is (num_agents, 2) for D>=2
    """
    rng = np.random.default_rng(seed)

    # init population uniformly in [lb, ub]
    positions = rng.uniform(lb, ub, size=(num_agents, dim))
    dest_pos = np.zeros(dim)
    dest_fitness = float("inf")

    convergence = np.zeros(max_iter)
    pop_history_2d = []  # เก็บเฉพาะมิติแรกๆ 2 มิติไว้ plot การเคลื่อนที่

    a = 2.0  # standard SCA schedule coefficient

    for t in range(max_iter):
        # boundary control + evaluate
        positions = np.clip(positions, lb, ub)
        fitnesses = np.apply_along_axis(objective_function, 1, positions)

        # update global best
        idx = np.argmin(fitnesses)
        if fitnesses[idx] < dest_fitness:
            dest_fitness = float(fitnesses[idx])
            dest_pos = positions[idx].copy()

        convergence[t] = dest_fitness

        # เก็บไว้ทำสไลด์ (เฉพาะ 2D แรก)
        if positions.shape[1] >= 2 and t < 20:
            pop_history_2d.append(positions[:, :2].copy())

        # r1 schedule
        r1 = a - t * (a / max_iter)

        # update positions
        for i in range(num_agents):
            for j in range(dim):
                r2 = rng.uniform(0, 2*np.pi)
                r3 = rng.uniform(0, 2)
                r4 = rng.uniform(0, 1)
                if r4 < 0.5:
                    positions[i, j] = positions[i, j] + r1*np.sin(r2)*abs(r3*dest_pos[j] - positions[i, j])
                else:
                    positions[i, j] = positions[i, j] + r1*np.cos(r2)*abs(r3*dest_pos[j] - positions[i, j])

    logs = {
        "convergence": convergence,
        "pop_history_2d": pop_history_2d
    }
    return dest_pos, dest_fitness, logs


# =========================
# Plot helpers
# =========================
def plot_griewank_contour_2d(xmin=-600, xmax=600, ymin=-600, ymax=600, levels=50, title="Griewank f4 (D=2) Contour", fname="f4_contour.png"):
    xs = np.linspace(xmin, xmax, 400)
    ys = np.linspace(ymin, ymax, 400)
    XX, YY = np.meshgrid(xs, ys)
    Z = 1 + (XX**2 + YY**2)/4000.0 - (np.cos(XX/np.sqrt(1)) * np.cos(YY/np.sqrt(2)))
    plt.figure(figsize=(7,6))
    cs = plt.contour(XX, YY, Z, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter([0],[0], marker="*", s=120, label="global min (0,0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def save_agent_slides_on_contour(pop_history_2d, xmin=-600, xmax=600, ymin=-600, ymax=600, outdir="slides_f4"):
    """
    วาดตำแหน่งเอเจนต์ (2D แรก) ลงบน contour ในแต่ละ iteration (20 ภาพแรก)
    """
    os.makedirs(outdir, exist_ok=True)

    # เตรียม contour พื้นหลังครั้งเดียว
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
        plt.title(f"SCA on Griewank — agent positions at iter {it}")
        plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
        plt.tight_layout()
        fname = os.path.join(outdir, f"f4_agents_iter_{it:02d}.png")
        plt.savefig(fname, dpi=200)
        plt.close()


def plot_convergence(curve, start_iter=5, end_iter=100, title="Convergence (best-so-far)", fname="f4_convergence.png"):
    n = len(curve)
    s = max(0, min(start_iter-1, n-1))
    e = max(1, min(end_iter, n))
    xs = np.arange(s+1, e+1)
    plt.figure(figsize=(7,5))
    plt.plot(xs, curve[s:e])
    plt.xlabel("Iteration")
    plt.ylabel("Best objective value")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


# =========================
# Experiment runners
# =========================
def run_10_trials_griewank_D30():
    """
    ทำตามข้อ 1: รัน 10 ครั้ง (D=30) รายงาน mean และ std ของ minimum ที่ได้
    """
    D = 30
    lb, ub = -600.0, 600.0
    NUM_TRIALS = 10

    num_agents = 20   # ใช้ 20 เอเจนต์เหมือนเงื่อนไข Rosenbrock เพื่อให้สอดคล้อง
    max_iter = 1000   # ปรับได้

    bests = []
    for trial in range(NUM_TRIALS):
        seed = 1234 + trial
        _, best, _ = sca(griewank, lb, ub, D, num_agents, max_iter, seed=seed)
        bests.append(best)
        print(f"[Trial {trial+1:02d}] best min = {best:.6e}")

    bests = np.array(bests, dtype=float)
    print("\n=== Summary (10 runs, D=30) ===")
    print(f"mean best = {bests.mean():.6e}")
    print(f"std  best = {bests.std(ddof=1):.6e}")
    return bests


def demo_plots_for_D2():
    """
    ทำตามข้อ 2 และส่วน visualization เพิ่มเติม:
    - contour plot (D=2)
    - สไลด์ตำแหน่งเอเจนต์ 20 iter แรกบน contour
    - กราฟ convergence ช่วง iter 5..100
    """
    D = 2
    lb, ub = -600.0, 600.0
    num_agents = 20
    max_iter = 120  # ให้เกิน 100 นิดหน่อย เพื่อ plot 5..100 ได้พอดี
    seed = 2025

    # วาด contour อย่างเดียว
    plot_griewank_contour_2d(xmin=lb, xmax=ub, ymin=lb, ymax=ub, fname="f4_contour.png")

    # รัน SCA เพื่อเอาข้อมูลมา plot
    _, best, logs = sca(griewank, lb, ub, D, num_agents, max_iter, seed=seed)
    print(f"[D=2 demo] best min = {best:.6e}")

    # สไลด์โชว์ตำแหน่งเอเจนต์ 20 ภาพแรก
    save_agent_slides_on_contour(logs["pop_history_2d"], xmin=lb, xmax=ub, ymin=lb, ymax=ub, outdir="slides_f4")

    # กราฟ convergence iter 5..100
    plot_convergence(logs["convergence"], start_iter=5, end_iter=100, fname="f4_convergence.png")


# =========================
# Main
# =========================
if __name__ == "__main__":
    # ---------- CONFIG ----------
    # หากต้องการปรับความเข้ม/คุณภาพ ให้เปลี่ยนพารามิเตอร์ต่อไปนี้
    # MAX_ITER_D30 = 1000  # เปลี่ยนใน run_10_trials_griewank_D30 ถ้าต้องการ
    # MAX_ITER_D2  = 120   # เปลี่ยนใน demo_plots_for_D2 ถ้าต้องการ

    # 1) 10 trials @ D=30
    run_10_trials_griewank_D30()

    # 2) contour + slides + convergence @ D=2
    demo_plots_for_D2()

    print("\nภาพที่ได้:")
    print(" - f4_contour.png")
    print(" - slides_f4/f4_agents_iter_01.png ... f4_agents_iter_20.png")
    print(" - f4_convergence.png")
