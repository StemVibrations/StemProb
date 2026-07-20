import json
import os

import stem
from stem.additional_processes import ParameterFieldParameters
from stem.boundary import DisplacementConstraint
from stem.field_generator import RandomFieldGenerator
from stem.load import PointLoad, MovingLoad
from stem.model import Model
import stem
from stem.output import GaussPointOutput, NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.solver import AnalysisType, TimeIntegration, DisplacementConvergenceCriteria, NewtonRaphsonStrategy, \
    NewmarkScheme, StressInitialisationType, Amgcl, SolverSettings, Problem, SolutionType
from stem.stem import Stem
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


"""
Contains test/example cases of models with defined soil layers
and project parameters
"""

def default_2d_soil_material() -> stem.soil_material.SoilMaterial:
    ndim = 2
    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
    soil_material = SoilMaterial(name="soil",
                                 soil_formulation=soil_formulation,
                                 constitutive_law=constitutive_law,
                                 retention_parameters=SaturatedBelowPhreaticLevelLaw())
    return soil_material


def test_case_2d_model():
    ndim = 2
    model = Model(ndim)
    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
    soil_material = SoilMaterial(name="soil",
                                 soil_formulation=soil_formulation,
                                 constitutive_law=constitutive_law,
                                 retention_parameters=SaturatedBelowPhreaticLevelLaw())
    model.add_soil_layer_by_coordinates(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        soil_material, "layer1"
    )

    load_coordinates = [
        (0.5, 1, 0)
    ]
    moving_load_coordinates = [
        (0, 1, 0), (1, 1, 0),
    ]
    output_coordinates = [
        (0.0, 1.0, 0.0),
        (0.5, 1.0, 0.0),
    ]
    no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, True, True], value=[0, 0, 0])
    roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, False, True], value=[0, 0, 0])

    # %%%%%%
    # DEFINE THE CALCULATION OUTPUT
    # %%%%%%
    # nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]
    nodal_results = [NodalOutput.DISPLACEMENT]
    gauss_point_results = [GaussPointOutput.YOUNG_MODULUS]

    point_load = PointLoad(active=[False, True, False], value=[0, -1000, 0])
    model.add_load_by_coordinates(load_coordinates, point_load, f"point_load")
    model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(1, [2, 4],
                                                 roller_displacement_parameters, "sides_roller")
    return model


def test_case_2d_model_project_params():
    analysis_type = AnalysisType.MECHANICAL
    solution_type = SolutionType.DYNAMIC
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(start_time=0.0, end_time=10.0, delta_time=0.1, reduction_factor=1.0,
                                       increase_factor=1.0)
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
                                     linear_solver_settings=linear_solver_settings, rayleigh_k=0.12,
                                     rayleigh_m=0.0001)

    # Set up the problem data
    problem = Problem(problem_name="calculate_moving_load_on_embankment_2d", number_of_threads=1,
                      settings=solver_settings)
    return problem


def ttest_uq_2d(soil_material: stem.soil_material.SoilMaterial = default_2d_soil_material()):
    model = Model(2)

    model.add_soil_layer_by_coordinates(
        [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)
        ],
        soil_material, "layer1"
    )
    model.set_mesh_size(1)

    random_field_generator = RandomFieldGenerator(
        n_dim=2,
        cov=0.1,
        model_name='Gaussian',
        v_scale_fluctuation=1,
        anisotropy=[0.5],
        angle=[0],
        seed=42,
    )
    field_parameters_json = ParameterFieldParameters(property_name="YOUNG_MODULUS",
                                                     function_type="json_file",
                                                     field_generator=random_field_generator)
    model.add_field(part_name="layer1", field_parameters=field_parameters_json)
    model.synchronise_geometry()
    model.generate_mesh()
    pass


def plot_response_output(responses_dict: dict,
                         disp_coord: str,
                         name_of_the_model: str | None = None,
                         NUM_SIMS: int | None = None):
    fig = plt.figure(layout='constrained')
    gs = GridSpec(2, 2, figure=fig)
    response_time_ax = fig.add_subplot(gs[0, :])
    hist_ax = fig.add_subplot(gs[1, 0])
    ax_statplot = fig.add_subplot(gs[1, 1])

    responses = responses_dict[f'DISPLACEMENT_{disp_coord}']
    time = responses_dict['TIME']

    responses_np = np.array(responses)

    # Compute the mean and 95% confidence interval of each column
    mean = np.mean(responses_np, axis=0)
    ci_lower = np.percentile(responses_np, 2.5, axis=0)
    ci_upper = np.percentile(responses_np, 97.5, axis=0)

    # Plot the response over time
    for i in range(NUM_SIMS):
        response_time_ax.plot(time, responses_np[i, :], c='k', alpha=0.05)
    response_time_ax.set_xlabel('Time [s]')
    response_time_ax.set_ylabel(f'Displacement {disp_coord} [m]')
    response_time_ax.set_xlabel('Time [s]')
    # response_time_ax.set_ylim([0.5 * np.min(responses_np), 3* np.max(responses_np)])

    # Plot the mean and 95% confidence interval
    ax_statplot.plot(time, mean, c='k', label='Mean')
    ax_statplot.fill_between(time, ci_lower, ci_upper, alpha=0.4, color='darkgray', label='95% CI')
    ax_statplot.set_xlabel('Time [s]')
    ax_statplot.set_ylabel(f'Displacement {disp_coord} [m]')
    # Set the limits of the y axis
    # ax_statplot.set_ylim([0.5 * np.min(responses_np), 3 * np.max(responses_np)])

    # Plot a histogram of the responses
    means_calculations = np.mean(responses_np, axis=1)
    mean_means = np.mean(means_calculations)
    hist_ax.hist(means_calculations, bins=50, color='blue', alpha=0.5)
    hist_ax.axvline(mean_means, color='k', linestyle='dashed', linewidth=1)
    hist_ax.set_xlabel(f'Mean displacement {disp_coord} [m]')
    hist_ax.set_ylabel('Frequency')

    if name_of_the_model is not None and NUM_SIMS is not None:
        fig.suptitle(f'{name_of_the_model} - {NUM_SIMS} simulations')
    plt.show()
    pass


def run_2d_model():
    NUM_SIMULATIONS = 100
    # SEEDS = np.random.randint(0, 1000, NUM_SIMULATIONS)
    SEEDS = np.arange(NUM_SIMULATIONS)

    input_files_dir = 'random_field_mc'
    ndim = 2

    responses_var = {}
    responses_var['DISPLACEMENT_Y'] = []
    responses_var['DISPLACEMENT_X'] = []

    for i in range(NUM_SIMULATIONS):

        model = Model(ndim)

        soil = default_2d_soil_material()
        model.add_soil_layer_by_coordinates(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
            soil, "layer1"
        )

        load_coordinates = [
            (0.5, 1, 0)
        ]
        moving_load_coordinates = [
            (0, 1, 0), (1, 1, 0),
        ]
        output_coordinates = [
            (0.0, 1.0, 0.0),
            (0.5, 1.0, 0.0),
        ]

        no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, True, True], value=[0, 0, 0])
        roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                                is_fixed=[True, False, True], value=[0, 0, 0])

        # %%%%%%
        # DEFINE THE CALCULATION OUTPUT
        # %%%%%%
        # nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]
        nodal_results = [NodalOutput.DISPLACEMENT]
        gauss_point_results = [GaussPointOutput.YOUNG_MODULUS]

        point_load = PointLoad(active=[False, True, False], value=[0, -1000, 0])

        moving_load = MovingLoad(load=[0.0, -1000.0, 0.0],
                                 direction=[1, 0, 0],
                                 velocity=0.1,
                                 origin=[0.0, 1.0, 0.0],
                                 offset=0.0)
        model.add_load_by_coordinates(load_coordinates, point_load, f"point_load")
        # model.add_load_by_coordinates(moving_load_coordinates, moving_load, f"moving_load")

        model.synchronise_geometry()
        # model.show_geometry(show_surface_ids=True, show_line_ids=True)

        model.add_boundary_condition_by_geometry_ids(1, [1], no_displacement_parameters, "base_fixed")
        model.add_boundary_condition_by_geometry_ids(1, [2, 4],
                                                     roller_displacement_parameters, "sides_roller")

        delta_time = 0.0005
        analysis_type = AnalysisType.MECHANICAL
        solution_type = SolutionType.DYNAMIC
        # Set up start and end time of calculation, time step and etc
        time_integration = TimeIntegration(start_time=0.0, end_time=10.0, delta_time=0.1, reduction_factor=1.0,
                                           increase_factor=1.0)
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
                                         linear_solver_settings=linear_solver_settings, rayleigh_k=0.12,
                                         rayleigh_m=0.0001)

        # Set up the problem data
        problem = Problem(problem_name="calculate_moving_load_on_embankment_2d", number_of_threads=1,
                          settings=solver_settings)
        model.project_parameters = problem

        random_field_generator = RandomFieldGenerator(
            n_dim=3,
            cov=0.1,
            model_name='Gaussian',
            v_scale_fluctuation=1,
            anisotropy=[0.5],
            angle=[0],
            seed=SEEDS[i],
            # seed=128
        )
        field_parameters_json = ParameterFieldParameters(property_name="YOUNG_MODULUS",
                                                         function_type="json_file",
                                                         field_generator=random_field_generator)

        model.add_field(part_name="layer1", field_parameters=field_parameters_json)

        model.add_output_settings_by_coordinates(
            coordinates=output_coordinates,
            part_name="midline_output",
            output_parameters=JsonOutputParameters(
                output_interval=delta_time - 1e-8,
                nodal_results=nodal_results,
                gauss_point_results=[],
            ),
            output_dir="output",
            output_name=f"json_output_{i}",
        )
        pass
        # model.synchronise_geometry()
        # model.generate_mesh()

        # model.add_output_settings(
        #     part_name="porous_computational_model_part",
        #     output_name="vtk_output",
        #     output_dir="output",
        #     output_parameters=VtkOutputParameters(
        #         output_interval=1,
        #         file_format="ascii",
        #         nodal_results=nodal_results,
        #         gauss_point_results=gauss_point_results,
        #         output_control_type="step"
        #     )
        # )

        stem = Stem(model, input_files_dir)
        stem.write_all_input_files()
        stem.run_calculation()

        path_to_results = os.path.join(input_files_dir, 'output', f"json_output_{i}.json")

        with open(path_to_results) as f:
            calculated_response = json.load(f)
            if i == 0:
                responses_var['TIME'] = calculated_response['TIME']
            results_node = calculated_response['NODE_5']
            responses_var['DISPLACEMENT_Y'].append(results_node['DISPLACEMENT_Y'])
            responses_var['DISPLACEMENT_X'].append(results_node['DISPLACEMENT_X'])

        # Delete the output files
        os.remove(path_to_results)

    plot_response_output(responses_dict=responses_var, disp_coord='Y', name_of_the_model='test',
                         NUM_SIMS=NUM_SIMULATIONS)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # for i in range(NUM_SIMULATIONS):
    #     ax.plot(responses_var['TIME'], responses_var['DISPLACEMENT_Y'][i], color='darkgray', alpha=0.3)
    # # ax.plot(calculated_response['TIME'], calculated_response['NODE_5']['DISPLACEMENT_Y'], label='DISPLACEMENT_Y')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Displacement Y [m]')
    # plt.show()

    pass


if __name__ == "__main__":
    run_2d_model()
