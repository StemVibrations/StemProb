"""
Sensitivity analysis tutorial -- Step 2: the Morris method.

Runs the Morris design on top of the shared base model (build_model /
run_model, see tutorial_sensitivity_base.py) and plots the resulting
mu* / sigma sensitivity ranking.

The quantity of interest is run_model(...)["v_y_max"], the peak absolute
vertical velocity, at the toe-of-embankment / track-midpoint output point
used throughout this tutorial: (3.0, 2.0, 25.0).
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

from tutorial_sensitivity_base import run_model

# ============================================================================
# Morris problem definition: the same nine parameters and ranges as the
# table in Step 1.
# ============================================================================
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


def plot_mu_sigma(Si, problem, path="sensitivity_plots/tutorial_sensitivity_morris_mu_sigma.png"):
    """
    mu vs sigma scatter, labelled points. Colour encodes the importance
    ranking by |mu| alone (1 = largest mean effect), ignoring sigma/
    nonlinearity -- a simple ranking cue on top of the two axes.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = problem["names"]
    mu, sigma = np.asarray(Si["mu"]), np.asarray(Si["sigma"])
    rank_by_abs_mu = np.argsort(np.argsort(-np.abs(mu))) + 1  # 1 = largest |mu|

    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(mu, sigma, c=rank_by_abs_mu, cmap="viridis_r", s=110,
                     edgecolors="k", linewidths=0.6, zorder=3)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1, zorder=1)
    for name, x, y in zip(names, mu, sigma):
        ax.annotate(name.replace("_", "\n"), (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel(r"$\mu$ (Mean Elementary Effect)")
    ax.set_ylabel(r"$\sigma$ (Standard Deviation of Elementary Effect)")
    ax.set_title("Morris Sensitivity Analysis: " + r"$\mu$ vs $\sigma$")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax, ticks=range(1, len(names) + 1))
    cbar.set_label("Importance ranking by |mu| (1 = largest)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Morris settings -- see the tutorial text for what each one controls.
    # ------------------------------------------------------------------
    N = 10           # number of trajectories (r)
    num_levels = 4    # levels in the p-level design
    sampling_seed = 42
    analysis_seed = 42

    samples = morris_sample.sample(problem, N=N, num_levels=num_levels, seed=sampling_seed)
    n_runs = len(samples)
    print(f"Morris design: {n_runs} runs = N({N}) x (num_vars({problem['num_vars']}) + 1)")

    outputs = np.empty(n_runs)
    for i, sample in enumerate(samples):
        print(f"\n{'=' * 70}\nMORRIS RUN {i + 1}/{n_runs}\n{'=' * 70}")
        outputs[i] = run_model(sample)["v_y_max"]
        print(f"  max|v_y| = {outputs[i] * 1000:.4f} mm/s")

    Si = morris_analyze.analyze(
        problem, samples, outputs,
        num_levels=num_levels, scaled=False, seed=analysis_seed, print_to_console=True,
    )

    os.makedirs("sensitivity_plots", exist_ok=True)
    with open("sensitivity_plots/tutorial_sensitivity_morris_v_y_results.json", "w") as f:
        json.dump({
            "names": problem["names"],
            "mu": np.asarray(Si["mu"]).tolist(),
            "mu_star": np.asarray(Si["mu_star"]).tolist(),
            "sigma": np.asarray(Si["sigma"]).tolist(),
        }, f, indent=2)

    plot_mu_sigma(Si, problem)
