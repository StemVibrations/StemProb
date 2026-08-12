"""
Sensitivity analysis tutorial -- Step 1: base model.

Builds and runs, once, the deterministic 3D embankment model that will later
be driven repeatedly (once per Morris sample) in Step 2. All numbers used
here are the reference ("mid-range") values of the parameters that will be
varied by the Morris method -- see REFERENCE_PARAMETERS and MORRIS_BOUNDS
below.

Run this file directly to write the STEM input files and execute a single
calculation, so the model setup can be checked before moving on to Step 2.
"""

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import MovingLoad
from stem.boundary import DisplacementConstraint
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.stem import Stem

input_files_dir = "tutorial_sensitivity_model"

# ============================================================================
# Reference (mean) parameter values and Morris screening ranges.
#
# These are the nine parameters that Step 2 will perturb with the Morris
# method. The reference value is simply the midpoint of each range and is
# only used here, in Step 1, to run one deterministic "base case".
# Poisson's ratio (0.3) and porosity (0.3) are kept fixed for every layer
# and are not part of the screening.
# ============================================================================
REFERENCE_PARAMETERS = {
    "clay_density": 2000.0,               # kg/m3
    "clay_young_modulus": 60e6,           # Pa
    "sand_density": 2000.0,               # kg/m3
    "sand_young_modulus": 250e6,          # Pa
    "embankment_density": 2000.0,         # kg/m3
    "embankment_young_modulus": 100e6,    # Pa
    "vertical_load": -30000.0,            # N
    "rayleigh_k": 5e-4,                   # stiffness-proportional damping
    "rayleigh_m": 0.5,                    # mass-proportional damping
}

MORRIS_BOUNDS = {
    "clay_density": [1000, 3000],
    "clay_young_modulus": [20e6, 100e6],
    "sand_density": [1000, 3000],
    "sand_young_modulus": [100e6, 400e6],
    "embankment_density": [1000, 3000],
    "embankment_young_modulus": [50e6, 150e6],
    "vertical_load": [-40000, -20000],
    "rayleigh_k": [1e-6, 1e-3],
    "rayleigh_m": [0.1, 0.9],
}

p = REFERENCE_PARAMETERS

# ============================================================================
# Model
# ============================================================================
ndim = 3
model = Model(ndim)

# --- Materials -------------------------------------------------------------
# Sand: bottom layer. Clay: layer directly under the embankment.
# Embankment: sloped fill on top. Poisson's ratio and porosity fixed at 0.3.
sand_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=p["sand_density"], POROSITY=0.3)
sand_law = LinearElasticSoil(YOUNG_MODULUS=p["sand_young_modulus"], POISSON_RATIO=0.2)
sand_material = SoilMaterial("sand", sand_formulation, sand_law, SaturatedBelowPhreaticLevelLaw())

clay_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=p["clay_density"], POROSITY=0.3)
clay_law = LinearElasticSoil(YOUNG_MODULUS=p["clay_young_modulus"], POISSON_RATIO=0.2)
clay_material = SoilMaterial("clay", clay_formulation, clay_law, SaturatedBelowPhreaticLevelLaw())

embankment_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=p["embankment_density"], POROSITY=0.3)
embankment_law = LinearElasticSoil(YOUNG_MODULUS=p["embankment_young_modulus"], POISSON_RATIO=0.2)
embankment_material = SoilMaterial("embankment", embankment_formulation, embankment_law, SaturatedBelowPhreaticLevelLaw())

# --- Geometry ----------------------------------------------------------------
# Cross-section (x-y plane), extruded 50 m in z. Deliberately small: 5 m wide,
# 50 m long, so that dozens of Morris runs stay computationally affordable.
soil1_coordinates = [(0.0, -2.0, 0.0), (5.0, -2.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]        # sand
soil2_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]           # clay
embankment_coordinates = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0.0, 3.0, 0.0)]

model.extrusion_length = 50.0
model.add_soil_layer_by_coordinates(soil1_coordinates, sand_material, "soil_layer_1")
model.add_soil_layer_by_coordinates(soil2_coordinates, clay_material, "soil_layer_2")
model.add_soil_layer_by_coordinates(embankment_coordinates, embankment_material, "embankment_layer")

# --- Load --------------------------------------------------------------------
# A single moving point load on the embankment crest represents the train.
# No rail, sleepers or UVEC vehicle model are included here.
load_coordinates = [(0.75, 3.0, 0.0), (0.75, 3.0, 50.0)]
moving_load = MovingLoad(load=[0.0, p["vertical_load"], 0.0], direction_signs=[1, 1, 1],
                          velocity=30, origin=[0.75, 3.0, 0.0], offset=0.0)
model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")

# --- Output points -------------------------------------------------------
# Example point: toe of the embankment, roughly midway along the track
# (z = 50/2). For an actual design, choose the output point(s) based on
# where the response actually matters (e.g. a building or receiver location).
output_coordinates = [
    (3.0, 2.0, 25.0),
]
nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]

model.add_output_settings_by_coordinates(
    coordinates=output_coordinates,
    part_name="sensitivity_output",
    output_parameters=JsonOutputParameters(
        output_interval=0.1,  # matched to delta_time below -- finer has no effect
        nodal_results=nodal_results,
        gauss_point_results=[],
    ),
    output_dir="output",
    output_name="sensitivity_output",
)

model.synchronise_geometry()
# Uncomment to inspect surface ids before assigning boundary conditions:
# model.show_geometry(show_surface_ids=True)

# --- Boundary conditions ---------------------------------------------------
# Base fixed in all directions; sides on rollers. No absorbing boundaries --
# wave reflections from the side boundaries are NOT damped in this reduced
# model (see the tutorial text for why that matters).
no_displacement_parameters = DisplacementConstraint(is_fixed=[True, True, True], value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(is_fixed=[True, False, True], value=[0, 0, 0])

model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17],
                                              roller_displacement_parameters, "sides_roller")

# --- Mesh --------------------------------------------------------------------
# A coarse 1 m element size keeps each run fast; not accurate enough for a
# real vibration assessment.
model.set_mesh_size(element_size=1.0)

# --- Solver settings -----------------------------------------------------
# Short duration (2 s) and a relatively large time step (0.1 s) again trade
# accuracy for run time, since Step 2 will repeat this run 50-500 times.
analysis_type = AnalysisType.MECHANICAL
solution_type = SolutionType.DYNAMIC
time_integration = TimeIntegration(start_time=0.0, end_time=2.0, delta_time=0.1,
                                    reduction_factor=1.0, increase_factor=1.0)
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                          displacement_absolute_tolerance=1.0e-9)
strategy_type = NewtonRaphsonStrategy()
scheme_type = NewmarkScheme()
linear_solver_settings = Amgcl()
stress_initialisation_type = StressInitialisationType.NONE

solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                  stress_initialisation_type=stress_initialisation_type,
                                  time_integration=time_integration,
                                  is_stiffness_matrix_constant=True, are_mass_and_damping_constant=True,
                                  convergence_criteria=convergence_criterion,
                                  strategy_type=strategy_type, scheme=scheme_type,
                                  linear_solver_settings=linear_solver_settings,
                                  rayleigh_k=p["rayleigh_k"], rayleigh_m=p["rayleigh_m"])

problem = Problem(problem_name="tutorial_sensitivity_base_model", number_of_threads=1,
                   settings=solver_settings)
model.project_parameters = problem

# --- VTK output (for visual inspection in Paraview) -------------------------
model.add_output_settings(
    part_name="porous_computational_model_part",
    output_name="vtk_output",
    output_dir="output",
    output_parameters=VtkOutputParameters(
        output_interval=1,
        nodal_results=nodal_results,
        gauss_point_results=[],
        output_control_type="step",
    ),
)

if __name__ == "__main__":
    stem = Stem(model, input_files_dir)
    stem.write_all_input_files()
    stem.run_calculation()
