"""
Sensitivity analysis tutorial -- Random Field (spatial variability).

Keeps all nine parameters fixed at their Step 1 reference values, and
instead varies the *spatial* distribution of the clay layer's Young's
modulus, using STEM's built-in random field generator. 

Reuses build_model / run_model from tutorial_sensitivity_base.py.
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from tutorial_sensitivity_base import run_model

# Step 1 reference parameter values -- see the Variables table.
REFERENCE_PARAMETERS = [
    2000.0,   # clay_density
    60e6,     # clay_young_modulus (RF mean)
    2000.0,   # sand_density
    250e6,    # sand_young_modulus
    2000.0,   # embankment_density
    100e6,    # embankment_young_modulus
    -30000.0, # vertical_load
    5e-4,     # rayleigh_k
    0.5,      # rayleigh_m
]

# Random field settings -- Gaussian model, applied to soil_layer_2
# (clay) Young's modulus, same mechanism as tutorial 4 in tutorials_RF.rst.
RF_COV = 0.1          # coefficient of variation
RF_ANISOTROPY = 10.0  # horizontal correlation length (m)
REFERENCE_SEED = 14   # the single-example seed used in the tutorial text


def plot_rf_histogram(v_y_max, seeds, seed14_value, deterministic_value,
                       path="sensitivity_plots/tutorial_sensitivity_rf_histogram.png"):
    """Histogram of max|v_y| with reference lines for seed=14 and the
    deterministic (no RF) run at the same reference parameters."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    v = np.asarray(v_y_max) * 1000  # mm/s
    n = len(v)
    ln_shape, ln_loc, ln_scale = stats.lognorm.fit(v, floc=0)
    x_fit = np.linspace(v.min() * 0.7, v.max() * 1.3, 400)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(v, bins=max(8, n // 4), density=True, color="#a8dadc", edgecolor="white",
            alpha=0.85, label=f"rf samples (N = {n})", zorder=2)
    ax.plot(x_fit, stats.lognorm.pdf(x_fit, ln_shape, ln_loc, ln_scale), color="#1d3557",
            lw=2.2, label=fr"Lognormal fit ($\sigma_\ln$ = {ln_shape:.3f})", zorder=3)
    ax.axvline(seed14_value * 1000, color="#e63946", ls="--", lw=1.8,
               label=f"Seed 14: {seed14_value * 1000:.4f} mm/s", zorder=4)
    ax.axvline(deterministic_value * 1000, color="#457b9d", ls="-", lw=1.8,
               label=f"Deterministic (no RF): {deterministic_value * 1000:.4f} mm/s", zorder=4)
    ax.set_xlabel(r"max$|v_y|$ (mm/s)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Output distribution -- {n} RF realisations, toe of embankment (z = 25 m)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    N = 20          # number of random field realisations (seeds)
    seeds = list(range(1, N + 1))
    assert REFERENCE_SEED in seeds

    v_y_max = np.empty(N)
    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 70}\nRF RUN {i + 1}/{N}  (seed={seed})\n{'=' * 70}")
        result = run_model(REFERENCE_PARAMETERS, rf_seed=seed, rf_cov=RF_COV, rf_anisotropy=RF_ANISOTROPY)
        v_y_max[i] = result["v_y_max"]
        print(f"  max|v_y| = {v_y_max[i] * 1000:.4f} mm/s")

    # Deterministic reference: same parameters, no random field.
    print(f"\n{'=' * 70}\nDETERMINISTIC RUN (no RF)\n{'=' * 70}")
    deterministic_result = run_model(REFERENCE_PARAMETERS)
    deterministic_value = deterministic_result["v_y_max"]
    print(f"  max|v_y| = {deterministic_value * 1000:.4f} mm/s")

    seed14_value = float(v_y_max[seeds.index(REFERENCE_SEED)])

    os.makedirs("sensitivity_plots", exist_ok=True)
    with open("sensitivity_plots/tutorial_sensitivity_rf_results.json", "w") as f:
        json.dump({
            "rf_cov": RF_COV,
            "rf_anisotropy": RF_ANISOTROPY,
            "rf_property": "clay_young_modulus (soil_layer_2)",
            "reference_parameters": REFERENCE_PARAMETERS,
            "seeds": seeds,
            "v_y_max": v_y_max.tolist(),
            "seed14_value": seed14_value,
            "deterministic_value": deterministic_value,
        }, f, indent=2)

    plot_rf_histogram(v_y_max, seeds, seed14_value, deterministic_value)
