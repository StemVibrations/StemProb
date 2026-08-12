"""
Spatial sensitivity analysis on the tutorial's small base model.

Follow-up to "Extending to multiple locations" in tutorial_sensitivity.rst:
one Morris design (N=10 trajectories -> 10*(9+1) = 100 STEM runs), evaluated
at every grid point in a single run each, so covering 18 locations costs the
same as covering one. Same model and same 9 parameters as
tutorial_sensitivity_base.py, just with output recorded at a grid instead of
one point.

Produces:
    results_sa_distribution.json
    plots/dominant_parameter_map.png
    plots/morris_mu_star_v_y_max.png
    plots/morris_mu_star_V_eff_max.png
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import MovingLoad
from stem.boundary import DisplacementConstraint
from stem.output import NodalOutput, JsonOutputParameters
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.stem import Stem

from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

_DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _DOCS_DIR)
from process_data import process_response_data  # noqa: E402

OUTPUT_DIR      = os.path.dirname(os.path.abspath(__file__))
RUN_NAME        = os.environ.get('SA_RUN_NAME', 'n10')
PLOTS_DIR       = os.path.join(OUTPUT_DIR, 'plots', RUN_NAME)
RESULTS_FILE    = os.path.join(OUTPUT_DIR, f'results_sa_distribution_{RUN_NAME}.json')
INPUT_FILES_DIR = os.path.join(OUTPUT_DIR, f'sa_distribution_model_{RUN_NAME}')

# Grid of output points: cross-track x (just beyond the embankment toe,
# at x=3 and x=4) x along-track z (5..45 m, every 5 m) -- 2x9 = 18 points.
GRID_X       = [3.0, 4.0]
GRID_Z       = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
GRID_POINTS  = [(x, 2.0, z) for x in GRID_X for z in GRID_Z]
POINT_LABELS = [f"x{int(x)}_z{int(z)}" for x, _, z in GRID_POINTS]
METRICS      = ['v_y_max', 'v_eff_max']

# Physical reference lines for the plan-view plots (see plot_dominant_parameter):
# embankment toe (this model's toe sits at x=3, one of the grid_x values) and
# the track / load line (x=0.75).
TOE_X   = 3.0
TRACK_X = 0.75

_PARAM_SHORT = {
    'clay_density':             'rho_clay',
    'clay_young_modulus':       'E_clay',
    'sand_density':             'rho_sand',
    'sand_young_modulus':       'E_sand',
    'embankment_density':       'rho_emb',
    'embankment_young_modulus': 'E_emb',
    'vertical_load':            'Load',
    'rayleigh_k':               'Ray_k',
    'rayleigh_m':               'Ray_m',
}

# Same 9 parameters and ranges as the rest of the tutorial.
problem = {
    "num_vars": 9,
    "names": [
        "clay_density", "clay_young_modulus",
        "sand_density", "sand_young_modulus",
        "embankment_density", "embankment_young_modulus",
        "vertical_load", "rayleigh_k", "rayleigh_m",
    ],
    "bounds": [
        [1000, 3000], [20e6, 100e6],
        [1000, 3000], [100e6, 400e6],
        [1000, 3000], [50e6, 150e6],
        [-40000, -20000], [1e-6, 1e-3], [0.1, 0.9],
    ],
}


def build_model(clay_density, clay_young_modulus, sand_density, sand_young_modulus,
                embankment_density, embankment_young_modulus, vertical_load,
                rayleigh_k, rayleigh_m) -> Model:
    """Same small embankment model as tutorial_sensitivity_base.py, with
    output recorded at GRID_POINTS instead of a single point."""
    ndim = 3
    model = Model(ndim)

    sand_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=sand_density, POROSITY=0.3)
    sand_law = LinearElasticSoil(YOUNG_MODULUS=sand_young_modulus, POISSON_RATIO=0.2)
    sand_material = SoilMaterial("sand", sand_formulation, sand_law, SaturatedBelowPhreaticLevelLaw())

    clay_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=clay_density, POROSITY=0.3)
    clay_law = LinearElasticSoil(YOUNG_MODULUS=clay_young_modulus, POISSON_RATIO=0.2)
    clay_material = SoilMaterial("clay", clay_formulation, clay_law, SaturatedBelowPhreaticLevelLaw())

    embankment_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=embankment_density, POROSITY=0.3)
    embankment_law = LinearElasticSoil(YOUNG_MODULUS=embankment_young_modulus, POISSON_RATIO=0.2)
    embankment_material = SoilMaterial("embankment", embankment_formulation, embankment_law,
                                       SaturatedBelowPhreaticLevelLaw())

    soil1_coordinates = [(0.0, -2.0, 0.0), (5.0, -2.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    soil2_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    embankment_coordinates = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0.0, 3.0, 0.0)]

    model.extrusion_length = 50.0
    model.add_soil_layer_by_coordinates(soil1_coordinates, sand_material, "soil_layer_1")
    model.add_soil_layer_by_coordinates(soil2_coordinates, clay_material, "soil_layer_2")
    model.add_soil_layer_by_coordinates(embankment_coordinates, embankment_material, "embankment_layer")

    load_coordinates = [(0.75, 3.0, 0.0), (0.75, 3.0, 50.0)]
    moving_load = MovingLoad(load=[0.0, vertical_load, 0.0], direction_signs=[1, 1, 1],
                             velocity=30, origin=[0.75, 3.0, 0.0], offset=0.0)
    model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")

    model.add_output_settings_by_coordinates(
        coordinates=GRID_POINTS,
        part_name="sensitivity_output",
        output_parameters=JsonOutputParameters(
            output_interval=0.1,  # matched to delta_time below -- finer has no effect
            nodal_results=[NodalOutput.VELOCITY],
            gauss_point_results=[],
        ),
        output_dir="output",
        output_name="sensitivity_output",
    )
    model.synchronise_geometry()

    no_displacement_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(is_fixed=[True, False, True], value=[0, 0, 0])
    model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17],
                                                 roller_displacement_parameters, "sides_roller")

    model.set_mesh_size(element_size=1.0)

    time_integration = TimeIntegration(start_time=0.0, end_time=2.0, delta_time=0.1,
                                       reduction_factor=1.0, increase_factor=1.0)
    convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                            displacement_absolute_tolerance=1.0e-9)
    solver_settings = SolverSettings(analysis_type=AnalysisType.MECHANICAL, solution_type=SolutionType.DYNAMIC,
                                     stress_initialisation_type=StressInitialisationType.NONE,
                                     time_integration=time_integration,
                                     is_stiffness_matrix_constant=True, are_mass_and_damping_constant=True,
                                     convergence_criteria=convergence_criterion,
                                     strategy_type=NewtonRaphsonStrategy(), scheme=NewmarkScheme(),
                                     linear_solver_settings=Amgcl(),
                                     rayleigh_k=rayleigh_k, rayleigh_m=rayleigh_m)

    model.project_parameters = Problem(problem_name="sa_distribution", number_of_threads=1,
                                       settings=solver_settings)
    return model


def run_model(parameters) -> dict:
    """Run one STEM model; return {point_label: {v_y_max, v_eff_max}} (mm/s)."""
    model = build_model(*parameters)
    stem = Stem(model, INPUT_FILES_DIR)
    stem.write_all_input_files()
    stem.run_calculation()

    output_path = os.path.join(INPUT_FILES_DIR, "output", "sensitivity_output.json")
    with open(output_path) as f:
        raw = json.load(f)

    time = np.asarray(raw["TIME"], dtype=float)
    node_keys = [k for k in raw if k.startswith("NODE_")]

    def _coord(k):
        return np.array(raw[k].get("COORDINATES", [0.0, 0.0, 0.0]), dtype=float)

    out = {}
    for point, label in zip(GRID_POINTS, POINT_LABELS):
        target = np.array(point, dtype=float)
        nearest = min(node_keys, key=lambda k: np.linalg.norm(_coord(k) - target))
        v_y = np.asarray(raw[nearest]["VELOCITY_Y"], dtype=float)
        pd = process_response_data(time, v_y)
        out[label] = {"v_y_max": pd["V_y_max"], "v_eff_max": pd["V_eff_max"]}
    return out


# ── Plotting ───────────────────────────────────────────────────────────────

def _edges(values):
    """Cell edges for pcolormesh from sorted sample centres (midpoints between
    consecutive samples, extrapolated by half the end spacing at both ends)."""
    v = sorted(values)
    if len(v) == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    mids = [(v[i] + v[i + 1]) / 2 for i in range(len(v) - 1)]
    return np.array([v[0] - (mids[0] - v[0])] + mids + [v[-1] + (v[-1] - mids[-1])])


def _plan_view(ax, xs, zs, values, title, cmap="viridis"):
    """Plan-view grid with numeric cell annotations and toe/track reference
    lines, matching output_distribution/plot_distribution.py's _plan_view."""
    norm = mcolors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    im = ax.pcolormesh(_edges(zs), _edges(xs), values, cmap=cmap, norm=norm,
                       shading="flat", alpha=0.85)

    for ix, x in enumerate(xs):
        for iz, z in enumerate(zs):
            val = values[ix, iz]
            if np.isfinite(val):
                ax.text(z, x, f"{val:.3g}", ha="center", va="center", fontsize=6.5,
                        color="white" if norm(val) > 0.45 else "black", fontweight="bold")

    ax.axhline(TOE_X, color="gray", lw=0.7, ls="--", label="embankment toe")
    ax.axhline(TRACK_X, color="orange", lw=1.0, ls="-", label=f"load path (x={TRACK_X})")
    ax.set_xlim(0, max(zs) + 5)
    ax.set_ylim(0, max(xs) + 1)
    ax.set_xlabel("Along-track position z (m)", fontsize=8)
    ax.set_ylabel("Cross-track position x (m)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=7)
    return im


def plot_single_parameter_map(results_metric: dict, param_name: str, metric: str, path: str):
    """Plan-view mu* map for one parameter -- the building block behind the
    3x3 grid and the dominant-parameter map below."""
    j = problem["names"].index(param_name)
    values = np.array([[results_metric[f"x{int(x)}_z{int(z)}"]["mu_star"][j] for z in GRID_Z]
                       for x in GRID_X])

    fig, ax = plt.subplots(figsize=(8, 4))
    im = _plan_view(ax, GRID_X, GRID_Z, values, title=f"{_PARAM_SHORT[param_name]} -- mu* ({metric})")
    fig.colorbar(im, ax=ax, label="mu_star")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dominant_parameter_single(results_metric: dict, metric: str, path: str):
    """Plan-view map for one metric: colour = index of the parameter with
    the highest Morris mu* at each grid point."""
    names = problem["names"]
    n_p = len(names)
    cmap = get_cmap("tab10", n_p)
    x_edges, z_edges = _edges(GRID_X), _edges(GRID_Z)

    dom = np.array([[np.argmax(results_metric[f"x{int(x)}_z{int(z)}"]["mu_star"])
                     for z in GRID_Z] for x in GRID_X], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.pcolormesh(z_edges, x_edges, dom, cmap=cmap, vmin=-0.5, vmax=n_p - 0.5, shading="flat")
    for ix, x in enumerate(GRID_X):
        for iz, z in enumerate(GRID_Z):
            ax.text(z, x, _PARAM_SHORT[names[int(dom[ix, iz])]], ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    ax.axhline(TOE_X, color="gray", lw=0.8, ls="--")
    ax.axhline(TRACK_X, color="orange", lw=1.2, ls="-")
    ax.set_xlim(0, max(GRID_Z) + 5)
    ax.set_ylim(0, max(GRID_X) + 1)
    ax.set_xlabel("Along-track z (m)")
    ax.set_ylabel("Cross-track x (m)")
    ax.set_title(f"Dominant parameter -- {metric}")
    patches = [mpatches.Patch(color=cmap(j), label=_PARAM_SHORT[names[j]]) for j in range(n_p)]
    fig.legend(handles=patches, fontsize=7, loc="lower center", ncol=n_p, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mu_star_map(results_metric: dict, metric: str, path: str):
    """3x3 plan-view grid: Morris mu* of each parameter at every grid point."""
    names = problem["names"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 13.5))
    fig.suptitle(f"Morris mu* -- {metric}", fontsize=13, fontweight="bold")
    for j, (ax, name) in enumerate(zip(axes.flat, names)):
        values = np.array([[results_metric[f"x{int(x)}_z{int(z)}"]["mu_star"][j] for z in GRID_Z]
                           for x in GRID_X])
        im = _plan_view(ax, GRID_X, GRID_Z, values, title=_PARAM_SHORT[name])
        fig.colorbar(im, ax=ax, label="mu_star", pad=0.01)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dominant_parameter(results: dict, path: str):
    """Plan-view map, one panel per metric: colour = index of the parameter
    with the highest Morris mu* at each grid point. Reference lines mark the
    embankment toe (x=3) and the track / load line (x=0.75)."""
    names = problem["names"]
    n_p = len(names)
    cmap = get_cmap("tab10", n_p)
    x_edges, z_edges = _edges(GRID_X), _edges(GRID_Z)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(7 * len(METRICS), 4))
    for ax, metric in zip(axes, METRICS):
        dom = np.array([[np.argmax(results[metric][f"x{int(x)}_z{int(z)}"]["mu_star"])
                         for z in GRID_Z] for x in GRID_X], dtype=float)

        ax.pcolormesh(z_edges, x_edges, dom, cmap=cmap, vmin=-0.5, vmax=n_p - 0.5, shading="flat")
        for ix, x in enumerate(GRID_X):
            for iz, z in enumerate(GRID_Z):
                ax.text(z, x, _PARAM_SHORT[names[int(dom[ix, iz])]], ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

        ax.axhline(TOE_X, color="gray", lw=0.8, ls="--")
        ax.axhline(TRACK_X, color="orange", lw=1.2, ls="-")
        ax.set_xlim(0, max(GRID_Z) + 5)
        ax.set_ylim(0, max(GRID_X) + 1)
        ax.set_xlabel("Along-track z (m)", fontsize=8)
        ax.set_ylabel("Cross-track x (m)", fontsize=8)
        ax.set_title(f"Dominant parameter -- {metric}", fontsize=9)

    patches = [mpatches.Patch(color=cmap(j), label=_PARAM_SHORT[names[j]]) for j in range(n_p)]
    fig.legend(handles=patches, fontsize=7, loc="lower center", ncol=n_p, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N = 10
    num_levels = 4
    seed = 42

    samples = morris_sample.sample(problem, N=N, num_levels=num_levels, seed=seed)
    n_runs = len(samples)
    print(f"Morris design: {n_runs} runs, {len(GRID_POINTS)} grid points recorded per run")

    outputs = {label: {m: np.empty(n_runs) for m in METRICS} for label in POINT_LABELS}
    for i, sample in enumerate(samples):
        print(f"\nMorris run {i + 1}/{n_runs}")
        point_results = run_model(sample)
        for label in POINT_LABELS:
            for m in METRICS:
                outputs[label][m][i] = point_results[label][m]

    results = {m: {} for m in METRICS}
    for m in METRICS:
        for label in POINT_LABELS:
            Si = morris_analyze.analyze(problem, samples, outputs[label][m],
                                        num_levels=num_levels, print_to_console=False)
            results[m][label] = {"mu_star": Si["mu_star"].tolist()}

    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "param_names": problem["names"],
            "point_labels": POINT_LABELS,
            "grid_points": GRID_POINTS,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {RESULTS_FILE}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_dominant_parameter(results, os.path.join(PLOTS_DIR, "dominant_parameter_map.png"))
    plot_dominant_parameter_single(results["v_y_max"], "v_y_max",
                                   os.path.join(PLOTS_DIR, "dominant_parameter_map_v_y_max.png"))
    plot_mu_star_map(results["v_y_max"], "v_y_max", os.path.join(PLOTS_DIR, "morris_mu_star_v_y_max.png"))
    plot_mu_star_map(results["v_eff_max"], "v_eff_max", os.path.join(PLOTS_DIR, "morris_mu_star_V_eff_max.png"))
    plot_single_parameter_map(results["v_y_max"], "clay_young_modulus", "v_y_max",
                              os.path.join(PLOTS_DIR, "mu_star_E_clay_v_y_max.png"))
    plot_single_parameter_map(results["v_y_max"], "embankment_density", "v_y_max",
                              os.path.join(PLOTS_DIR, "mu_star_embankment_density_v_y_max.png"))
