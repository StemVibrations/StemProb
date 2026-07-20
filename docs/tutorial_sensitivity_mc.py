"""
Sensitivity analysis tutorial -- Step 3: Monte Carlo / Latin Hypercube sampling.

Unlike Step 2 (Morris), this step is not about ranking which parameters
matter. It propagates realistic parameter uncertainty through the Step 1
base model (build_model / run_model, see tutorial_sensitivity_base.py) to
see what the resulting *distribution* of the response looks like -- e.g. to
answer "how much could v_y actually vary, given what we believe about the
uncertainty in these nine parameters?"

Two sampling strategies are supported: plain (pseudo-)random Monte Carlo,
and Latin Hypercube Sampling (LHS). Both draw from the same marginal
distributions below; only how they cover the 9-dimensional parameter space
differs. Set METHOD to "random" or "lhs".
"""

import os
import json

import numpy as np
from scipy import stats
from scipy.stats import qmc
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from tutorial_sensitivity_base import run_model

# ============================================================================
# Parameter distributions: realistic uncertainty around the Step 1 reference
# values. Unlike the wide Morris screening ranges (used to explore the full
# design space and find important parameters), these are meant to reflect
# actual uncertainty, so that the resulting output distribution is
# meaningful. Material properties are lognormal (always positive); load and
# damping are normal. 5% COV is an example choice, not a validated value.
# ============================================================================
DISTRIBUTIONS = {
    "clay_density":             {"mean": 2000.0, "cov": 0.05, "dist": "lognormal"},
    "clay_young_modulus":       {"mean": 60e6,   "cov": 0.05, "dist": "lognormal"},
    "sand_density":              {"mean": 2000.0, "cov": 0.05, "dist": "lognormal"},
    "sand_young_modulus":       {"mean": 250e6,  "cov": 0.05, "dist": "lognormal"},
    "embankment_density":       {"mean": 2000.0, "cov": 0.05, "dist": "lognormal"},
    "embankment_young_modulus": {"mean": 100e6,  "cov": 0.05, "dist": "lognormal"},
    "vertical_load":             {"mean": -30000.0, "cov": 0.05, "dist": "normal"},
    "rayleigh_k":                {"mean": 5e-4,     "cov": 0.05, "dist": "normal"},
    "rayleigh_m":                {"mean": 0.5,      "cov": 0.05, "dist": "normal"},
}
NAMES = list(DISTRIBUTIONS.keys())


def uniform_to_parameters(u: np.ndarray) -> np.ndarray:
    """
    Map an (n_samples, 9) array of Uniform[0,1] draws -- one column per
    parameter -- to physical parameter values, through each parameter's
    inverse CDF. This is what makes plain random sampling and LHS
    interchangeable: both produce a Uniform[0,1]^9 array, just with
    different coverage of that space; the transform to physical units is
    identical either way.
    """
    out = np.empty_like(u)
    for i, name in enumerate(NAMES):
        d = DISTRIBUTIONS[name]
        if d["dist"] == "lognormal":
            sigma_ln = np.sqrt(np.log(1 + d["cov"] ** 2))
            mu_ln = np.log(d["mean"]) - 0.5 * sigma_ln ** 2
            out[:, i] = stats.lognorm.ppf(u[:, i], s=sigma_ln, scale=np.exp(mu_ln))
        else:  # normal
            std = abs(d["mean"]) * d["cov"]
            out[:, i] = stats.norm.ppf(u[:, i], loc=d["mean"], scale=std)
    return out


def plot_response_distribution(time_list, v_y_list, v_y_max, method, n,
                                out_dir="sensitivity_plots"):
    """
    Two figures: (1) v_y(t) spaghetti plot with mean +/- 1 std envelope,
    (2) histogram of the peak response v_y_max with a lognormal fit.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Spaghetti plot -----------------------------------------------------
    t_fine = np.linspace(time_list[0][0], time_list[0][-1], 400)
    traces = np.array([CubicSpline(t, v)(t_fine) for t, v in zip(time_list, v_y_list)]) * 1000  # mm/s
    mean_trace = traces.mean(axis=0)
    std_trace = traces.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(t_fine, mean_trace - std_trace, mean_trace + std_trace,
                     color="#457b9d", alpha=0.18, zorder=1, label=r"Mean $\pm 1\sigma$ envelope")
    for trace in traces:
        ax.plot(t_fine, trace, color="#adb5bd", lw=0.8, alpha=0.6, zorder=2)
    ax.plot(t_fine, mean_trace, color="#1d3557", lw=2.2, zorder=3, label=f"Mean (N = {n} samples)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Velocity $v_y$ (mm/s)")
    ax.set_title(f"Response variability ({method}, N = {n})")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"tutorial_sensitivity_mc_spaghetti_{method}.png"), dpi=150)
    plt.close(fig)

    # --- Histogram of the peak response -------------------------------------
    v = np.asarray(v_y_max) * 1000  # mm/s
    mu_v, sd_v = v.mean(), v.std(ddof=1)
    ln_shape, ln_loc, ln_scale = stats.lognorm.fit(v, floc=0)
    x_fit = np.linspace(v.min() * 0.7, v.max() * 1.3, 400)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(v, bins=max(8, n // 4), density=True, color="#a8dadc", edgecolor="white",
            alpha=0.85, label=f"{method} samples (N = {n})", zorder=2)
    ax.plot(x_fit, stats.lognorm.pdf(x_fit, ln_shape, ln_loc, ln_scale), color="#1d3557",
            lw=2.2, label=fr"Lognormal fit ($\sigma_\ln$ = {ln_shape:.3f})", zorder=3)
    ax.axvline(mu_v, color="#457b9d", ls=":", lw=1.6,
               label=fr"Mean: {mu_v:.4f} mm/s (CV = {sd_v / mu_v * 100:.0f} %)")
    ax.set_xlabel(r"max$|v_y|$ (mm/s)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Output distribution -- {n} {method} realisations, toe of embankment (z = 25 m)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"tutorial_sensitivity_mc_histogram_{method}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    N = 30
    seed = 42
    method = "lhs"  # "lhs" or "random"

    if method == "lhs":
        sampler = qmc.LatinHypercube(d=len(NAMES), seed=seed)
        u = sampler.random(n=N)
    else:
        rng = np.random.default_rng(seed)
        u = rng.random((N, len(NAMES)))

    samples = uniform_to_parameters(u)

    v_y_max = np.empty(N)
    time_list, v_y_list = [], []
    for i, sample in enumerate(samples):
        print(f"\n{'=' * 70}\n{method.upper()} RUN {i + 1}/{N}\n{'=' * 70}")
        result = run_model(sample)
        v_y_max[i] = result["v_y_max"]
        time_list.append(result["time"])
        v_y_list.append(result["v_y"])
        print(f"  max|v_y| = {v_y_max[i] * 1000:.4f} mm/s")

    os.makedirs("sensitivity_plots", exist_ok=True)
    with open(f"sensitivity_plots/tutorial_sensitivity_mc_{method}_results.json", "w") as f:
        json.dump({
            "method": method,
            "n": N,
            "names": NAMES,
            "samples": samples.tolist(),
            "v_y_max": v_y_max.tolist(),
        }, f, indent=2)

    plot_response_distribution(time_list, v_y_list, v_y_max, method, N)
