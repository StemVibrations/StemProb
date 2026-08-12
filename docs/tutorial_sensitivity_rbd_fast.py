"""
Sensitivity analysis tutorial -- RBD-FAST, the alternative to Morris
introduced in the tutorial text.

Runs on the same base model, same nine parameters/bounds, and same output
point as tutorial_sensitivity_morris.py, so the two methods' rankings can be
compared directly. Unlike Morris, RBD-FAST needs an independent (not
trajectory-based) sample -- drawn here with Latin Hypercube Sampling.
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from SALib.analyze import rbd_fast as rbd_fast_analyze

from tutorial_sensitivity_base import run_model

# Same nine parameters and ranges as the Morris problem definition.
problem = {
    "num_vars": 9,
    "names": [
        "clay_density", "clay_young_modulus",
        "sand_density", "sand_young_modulus",
        "embankment_density", "embankment_young_modulus",
        "vertical_load", "rayleigh_k", "rayleigh_m",
    ],
    "bounds": [
        [1000, 3000],
        [20e6, 100e6],
        [1000, 3000],
        [100e6, 400e6],
        [1000, 3000],
        [50e6, 150e6],
        [-40000, -20000],
        [1e-6, 1e-3],
        [0.1, 0.9],
    ],
}


def plot_rbd_fast(Si, problem, path="sensitivity_plots/tutorial_sensitivity_rbd_fast_S1.png"):
    """
    Horizontal bar chart of the first-order Sobol' index S_i, ranked.
    Colour encodes the same importance-ranking convention as the Morris
    mu-vs-sigma plot (1 = largest, shown yellow).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = problem["names"]
    S_i = np.asarray(Si["S1"])
    order = np.argsort(-S_i)
    rank_by_S_i = np.argsort(np.argsort(-S_i)) + 1

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(names)))
    ax.barh([names[i].replace("_", " ") for i in order], S_i[order],
            color=[colors[rank_by_S_i[i] - 1] for i in order], edgecolor="k")
    ax.invert_yaxis()
    ax.set_xlabel(r"$S_i$ (first-order Sobol' index)")
    ax.set_title("RBD-FAST Sensitivity Analysis: ranked $S_i$")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


def plot_rank_comparison(names, morris_mu_star, rbd_S_i,
                         path="sensitivity_plots/tutorial_sensitivity_rank_comparison.png"):
    """
    Grouped column chart comparing Morris and RBD-FAST parameter rankings
    directly -- ranks share one 1-9 scale, unlike the raw mu*/S_i values,
    which are in unrelated units. Bar height is inverted (rank 1 tallest)
    so "more important" reads as "bigger"; the rank number is labelled at
    each bar's tip.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(names)
    morris_rank = np.argsort(np.argsort(-np.abs(np.asarray(morris_mu_star)))) + 1
    rbd_rank = np.argsort(np.argsort(-np.asarray(rbd_S_i))) + 1

    order = np.argsort(morris_rank)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    morris_h = n - morris_rank[order]
    rbd_h = n - rbd_rank[order]
    bars_m = ax.bar(x - width / 2, morris_h, width, color="#2a78d6", label="Morris (μ* rank)")
    bars_r = ax.bar(x + width / 2, rbd_h, width, color="#eb6834", label="RBD-FAST (S_i rank)")

    for bar, rank in zip(bars_m, morris_rank[order]):
        ax.annotate(str(rank), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    for bar, rank in zip(bars_r, rbd_rank[order]):
        ax.annotate(str(rank), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([names[i].replace("_", "\n") for i in order], fontsize=8)
    ax.set_yticks(np.arange(0, n))
    ax.set_yticklabels([str(n - p) for p in range(0, n)])
    ax.set_ylim(0, n)
    ax.set_ylabel("Rank (1 = most important)")
    ax.set_title("Morris vs RBD-FAST -- parameter ranking, max|v_y|")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # RBD-FAST settings -- see the tutorial text for what each one controls.
    # ------------------------------------------------------------------
    N = 100          # independent LHS samples
    M = 10           # harmonics (SALib default)
    seed = 42

    bounds = np.array(problem["bounds"])
    sampler = qmc.LatinHypercube(d=problem["num_vars"], seed=seed)
    samples = qmc.scale(sampler.random(n=N), bounds[:, 0], bounds[:, 1])

    outputs = np.empty(N)
    for i, sample in enumerate(samples):
        print(f"\n{'=' * 70}\nRBD-FAST RUN {i + 1}/{N}\n{'=' * 70}")
        outputs[i] = run_model(sample)["v_y_max"]
        print(f"  max|v_y| = {outputs[i] * 1000:.4f} mm/s")

    Si = rbd_fast_analyze.analyze(problem, samples, outputs, M=M, print_to_console=True)

    os.makedirs("sensitivity_plots", exist_ok=True)
    with open("sensitivity_plots/tutorial_sensitivity_rbd_fast_v_y_results.json", "w") as f:
        json.dump({
            "names": problem["names"],
            "n_samples": N,
            "M": M,
            "S1": np.asarray(Si["S1"]).tolist(),
        }, f, indent=2)

    plot_rbd_fast(Si, problem)

    morris_file = "../legacy/sensitivity_plots/tutorial_sensitivity_morris_v_y_results.json"
    if os.path.exists(morris_file):
        with open(morris_file) as f:
            morris_results = json.load(f)
        plot_rank_comparison(problem["names"], morris_results["mu_star"], Si["S1"])
