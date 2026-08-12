"""
Sensitivity/probabilistic analysis tutorial -- shared base model.

Provides ``build_model`` and ``run_model``, the parametrised version of the
Step 1 model (see tutorial_sensitivity_model.py). Every later step of this
tutorial that needs to run the model many times with different parameter
values -- Morris screening, Monte Carlo / Latin Hypercube sampling, Random
Field studies -- imports these two functions rather than redefining the
model.

``run_model`` returns the full ``v_y`` time history plus one example
scalar reduction, max(|v_y|). That reduction is only an example: depending
on what a given study needs, a different output point, a different recorded
quantity, or a different way of summarising the time series may be more
appropriate.
"""

import os
import json

import numpy as np
from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import MovingLoad
from stem.boundary import DisplacementConstraint
from stem.output import NodalOutput, JsonOutputParameters
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria, \
    NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.stem import Stem

# RF helpers make RandomFieldGenerator/ParameterFieldParameters construction
# robust across STEM builds (see the modules themselves for why).
from random_field_utils import create_random_field_generator
from parameter_field_utils import create_parameter_field_parameters

input_files_dir = "tutorial_sensitivity_model"
OUTPUT_POINT = (3.0, 2.0, 25.0)  # toe of embankment, track midpoint -- same point as Step 1


def build_model(clay_density, clay_young_modulus, sand_density, sand_young_modulus,
                 embankment_density, embankment_young_modulus, vertical_load,
                 rayleigh_k, rayleigh_m,
                 rf_seed=None, rf_cov=0.1, rf_anisotropy=10.0) -> Model:
    """
    Build the Step 1 base model with all nine table parameters free.

    If ``rf_seed`` is given, a spatial random field (Gaussian, coefficient of
    variation ``rf_cov``, horizontal correlation length ``rf_anisotropy``
    metres) is additionally applied to the Young's modulus of the clay layer
    (``soil_layer_2``), on top of the ``clay_young_modulus`` mean value --
    the same mechanism used in tutorial 4 of ``tutorials_RF.rst``.
    """
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

    if rf_seed is not None:
        random_field_generator = create_random_field_generator(
            dim=3, cov=rf_cov, model_name="Gaussian", v_scale_fluctuation=1,
            anisotropy=[rf_anisotropy], angle=[0], seed=rf_seed,
        )
        field_parameters = create_parameter_field_parameters(
            property_name="YOUNG_MODULUS", function_type="json_file",
            field_generator=random_field_generator,
        )
        model.add_field(part_name="soil_layer_2", field_parameters=field_parameters)

    load_coordinates = [(0.75, 3.0, 0.0), (0.75, 3.0, 50.0)]
    moving_load = MovingLoad(load=[0.0, vertical_load, 0.0], direction_signs=[1, 1, 1],
                             velocity=30, origin=[0.75, 3.0, 0.0], offset=0.0)
    model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")

    # Only VELOCITY is needed: every study built on this function reduces v_y.
    model.add_output_settings_by_coordinates(
        coordinates=[OUTPUT_POINT],
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

    model.project_parameters = Problem(problem_name="tutorial_sensitivity_probabilistic", number_of_threads=1,
                                       settings=solver_settings)
    return model


def run_model(parameters, rf_seed=None, rf_cov=0.1, rf_anisotropy=10.0) -> dict:
    """
    Build, run and post-process one STEM model for a single set of
    parameter values. See ``build_model`` for the ``rf_*`` arguments.

    Returns
    -------
    dict with keys:
        time    -- time array (s)
        v_y     -- vertical velocity at the output point (m/s), full history
        v_y_max -- max(|v_y|) (m/s) -- an example scalar quantity of interest
    """
    model = build_model(*parameters, rf_seed=rf_seed, rf_cov=rf_cov, rf_anisotropy=rf_anisotropy)

    stem = Stem(model, input_files_dir)
    stem.write_all_input_files()
    stem.run_calculation()

    output_path = os.path.join(input_files_dir, "output", "sensitivity_output.json")
    with open(output_path) as f:
        results = json.load(f)

    node_key = next(k for k in results if k.startswith("NODE_"))
    time = np.asarray(results["TIME"], dtype=float)
    v_y = np.asarray(results[node_key]["VELOCITY_Y"], dtype=float)
    return {"time": time, "v_y": v_y, "v_y_max": float(np.max(np.abs(v_y)))}
