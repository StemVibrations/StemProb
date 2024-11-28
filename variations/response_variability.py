import json
import os
from copy import deepcopy

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from stem.additional_processes import ParameterFieldParameters
from stem.field_generator import RandomFieldGenerator
import stem.model
from stem.output import JsonOutputParameters

from examples.example_models import test_case_2d_model, test_case_2d_model_project_params
from stem.model_part import BodyModelPart, ModelPart

from stem.stem import Stem
from stem.output import NodalOutput


class UserVariations():
    """"
    Class for running uncertainty quantification on a model using user-defined stochastic parameters.

    Attributes
    ----------
    model : stem.model.Model
        The defined model to run the uncertainty quantification on.
    project_params : stem.solver.Problem
        The project parameters (calculation properties) for the model.
    model_part : str
        The name of the model part that is considered to have uncertain properties.
    input_files_dir : str
        The directory where the input files are stored.
    output_var_names : list
        The name of the variables to be recorded to output. Needs to be one of the following:
        'disp', 'vel', 'acc', 'total_disp', 'water'.
    model_part: str
        The name of the model part to be considered for the uncertainty quantification.
        Either a stem.model_part.BodyModelPart (for studying soil spatial variability) or a
        stem.model_part.ModelPart (for a process model part).
        TODO: Implement the functionality for a list of model parts.
    unc_type : str | list
        The type of uncertainty distribution to be considered for the random variates of a stem.model_part.ModelPart.
        Either 'Gaussian', 'Lognormal', 'Uniform'.
        TODO: Implement the functionality for a list of model parts.
    output_coordinates : list
        The coordinates of the nodes to be given to output.
    num_simulations : int
        The number of simulations to be run.
    property_name : str
        The name of the property to be treated as uncertain when considering spatial variability of soil.
        E.g. 'YOUNG_MODULUS'.
    dist_params : tuple
        The parameters of the uncertainty distribution to be considered for the random variates of a stem.model_part.ModelPart.
    random_state: int | None
        The random state to be used for generating the random variates of a stem.model_part.ModelPart
    load_direction: str | None
        The direction of the load to be considered as uncertain. Either 'X', 'Y', 'Z'.
    output_files_dir: str
        The directory where the output files are stored.
    output_time_interval: float | None
        The time interval for the calculation using random fields on a soil layer. If None, the default value is 0.0005.
    response_vars: str | list[str]
        The response variables to be considered for the uncertainty quantification.
        If None, the default value is ['TIME', 'DISPLACEMENT_X', 'DISPLACEMENT_Y', 'DISPLACEMENT_Z', 'ACCELERATION_X',
        'ACCELERATION_Y', 'ACCELERATION_Z'].
    """
    def __init__(
            self,
            stem_model: stem.model.Model,
            project_params: stem.solver.Problem,
            input_files_dir: str,
            output_var_names: list,
            model_part: str,
            unc_type: str,
            output_coordinates: list,
            num_simulations: int = 100,
            property_name: str | list | None = None,
            dist_params:  tuple | None = None,
            random_state: int = None,
            load_direction: str | None = None,
            output_files_dir: str = None,
            output_time_interval: float | None = None,
            response_vars: str | list[str] = None,
    ):
        self.model = stem_model
        self.project_params = project_params
        self.model_part = model_part
        self.input_files_dir = input_files_dir
        # self.input_files_dir = input_files_dir
        if output_files_dir is None:
            self.output_files_dir = os.path.join(os.getcwd(), os.path.join(input_files_dir, 'json_output.json'))
        else:
            self.output_files_dir = output_files_dir

        if dist_params is None:
            self.dist_params = (0, 1)
        else:
            self.dist_params = dist_params
        self.load_direction = self._parse_load_direction(load_direction)
        self.output_time_interval = 0.0005 if output_time_interval is None else output_time_interval
        self.output_coordinates = self._validate_output_coordinates(output_coordinates)
        self.output_var_names = self._parse_output_var_names(output_var_names)
        self.response_vars = self._parse_response_vars(response_vars)
        self.calculation_results = []
        self.num_simulations = num_simulations
        self.property_name = property_name
        self.unc_type_str = unc_type
        self.random_state = random_state

    def run(self,
            sim_type: str,
            ):

        if not isinstance(self.model_part, list):
            # Check the model part
            model_part_stem = self._get_model_part_by_name(self.model_part)
            # Check which model part needs to be simulated
            if type(model_part_stem) is BodyModelPart:
                # Perform quantification of spatial variability on the soil property
                random_fields = self._set_up_rfs()
                # self.model.synchronise_geometry()
                self.model.add_output_settings_by_coordinates(
                    coordinates=self.output_coordinates,
                    part_name="midline_output",
                    output_parameters=JsonOutputParameters(
                        output_interval=self.output_time_interval - 1e-8,
                        nodal_results=self.output_var_names,
                        gauss_point_results=[],
                    ),
                    output_dir=os.path.dirname(self.output_files_dir),
                    output_name=os.path.basename(self.output_files_dir),
                )

                for random_field in random_fields:
                    loop_model = test_case_2d_model()

                    loop_model.add_output_settings_by_coordinates(
                        coordinates=self.output_coordinates,
                        part_name="midline_output",
                        output_parameters=JsonOutputParameters(
                            output_interval=self.output_time_interval - 1e-8,
                            nodal_results=self.output_var_names,
                            gauss_point_results=[],
                        ),
                        output_dir=os.path.dirname(self.output_files_dir),
                        output_name=os.path.basename(self.output_files_dir),
                    )

                    field_params_json = ParameterFieldParameters(
                        property_name=self.property_name,
                        function_type="json_file",
                        field_generator=random_field)

                    # self.model.add_field(
                    #     part_name=self.model_part,
                    #     field_parameters=field_params_json)
                    loop_model.add_field(
                        part_name=self.model_part,
                        field_parameters=field_params_json
                    )
                    # Set the project (calculation) parameters
                    # self.model.project_parameters = self.project_params
                    loop_model.project_parameters = self.project_params

                    # Run the analysis
                    analysis = Stem(loop_model, self.input_files_dir)
                    analysis.write_all_input_files()
                    analysis.run_calculation()
                    self._run_post_processing()
                    os.remove(self.output_files_dir)
                    pass
            elif type(model_part_stem) is ModelPart:
                random_variates = self._set_up_rvs(model_part_stem)

                for rv in random_variates:
                    loop_model = test_case_2d_model()

                    loop_model.add_output_settings_by_coordinates(
                        coordinates=self.output_coordinates,
                        part_name="midline_output",
                        output_parameters=JsonOutputParameters(
                            output_interval=self.output_time_interval - 1e-8,
                            nodal_results=self.output_var_names,
                            gauss_point_results=[],
                        ),
                        output_dir=os.path.dirname(self.output_files_dir),
                        output_name=os.path.basename(self.output_files_dir),
                    )

                    # Find the model part
                    for process_model_part in loop_model.process_model_parts:
                        if process_model_part.name != model_part_stem.name:
                            continue
                        else:
                            # Change the load magnitude to the random variate
                            process_model_part.parameters.value[self.load_direction] = rv

                    loop_model.project_parameters = self.project_params
                    analysis = Stem(loop_model, self.input_files_dir)
                    analysis.write_all_input_files()
                    analysis.run_calculation()
                    self._run_post_processing()
                    os.remove(self.output_files_dir)
                    pass

        else:
            raise ValueError("Model part not found in the model")

    # else:
    #     pass
    # match sim_type:
    #     case 'MC' | 'Monte Carlo':
    #         self._monte_carlo_sim(num_simulations)
    pass

    def plot_results(self,
                     disp_coord: str,
                     node_number: int,
                     property: str = 'disp',
                     name_of_the_model: str | None = None,
                     save_fig: bool = False,
                     show_plot: bool = False):
        fig = plt.figure(layout='constrained')
        gs = GridSpec(2, 2, figure=fig)
        response_time_ax = fig.add_subplot(gs[0, :])
        hist_ax = fig.add_subplot(gs[1, 0])
        ax_statplot = fig.add_subplot(gs[1, 1])

        if 'disp' in property:
            response_variable = f'DISPLACEMENT_{disp_coord}'
        elif 'vel' in property:
            response_variable = f'VELOCITY_{disp_coord}'
        elif 'acc' in property:
            response_variable = f'ACCELERATION_{disp_coord}'
        else:
            raise ValueError("Result property name must be 'disp', 'vel' or 'acc'")

        # Parse the simulation results into a numpy array
        results_stacked = []
        for i in range(len(self.calculation_results)):
            results = self.calculation_results[i]
            results_node_i = results[f'NODE_{node_number}'][response_variable]

            results_stacked.append(results_node_i)
            pass

        results_stacked_np = np.array(results_stacked)

        # Compute the mean and 95% confidence interval of each column
        mean = np.mean(results_stacked_np, axis=0)
        ci_lower = np.percentile(results_stacked_np, 2.5, axis=0)
        ci_upper = np.percentile(results_stacked_np, 97.5, axis=0)

        time = self.calculation_results[0]['TIME']

        # Plot the response over time - loop over the rows of results_stacked_np
        for i in range(len(results_stacked_np)):
            response_time_ax.plot(time, results_stacked_np[i], c='k', alpha=0.05)
        response_time_ax.set_xlabel('Time [s]')
        response_time_ax.set_ylabel(f'{response_variable} [m]')

        # Plot the mean and 95% confidence interval
        ax_statplot.plot(time, mean, c='k', label='Mean')
        ax_statplot.fill_between(time, ci_lower, ci_upper, alpha=0.4, color='darkgray', label='95% CI')
        ax_statplot.set_xlabel('Time [s]')
        ax_statplot.set_ylabel(f'{response_variable} [m]')

        # Plot a histogram of the responses
        means_calculations = np.mean(results_stacked_np, axis=1)
        bin_width = 2 * (np.percentile(means_calculations, 75) - np.percentile(means_calculations, 25)) / (
                len(means_calculations) ** (1 / 3))
        bins = int((means_calculations.max() - means_calculations.min()) / bin_width)

        mean_means = np.mean(means_calculations)
        hist_ax.hist(means_calculations, bins=bins, color='blue', alpha=0.5)
        hist_ax.axvline(mean_means, color='k', linestyle='dashed', linewidth=1)
        hist_ax.set_xlabel(f'Mean {response_variable} [m]')
        hist_ax.set_ylabel('Frequency')

        if name_of_the_model is not None:
            fig.suptitle(f'{response_variable} at node {node_number}')

        output_dir = os.path.dirname(self.output_files_dir)

        if save_fig:
            fig.savefig(os.path.join(output_dir, f'{response_variable}_node_{node_number}.jpeg'),
                        dpi=300,
                        bbox_inches='tight'
                        )
        if show_plot:
            plt.show()
        plt.close(fig)

    def calculate_soil_variability(self):
        # Calculate soil variability
        pass

    def _parse_response_vars(
            self,
            response_vars_names: str | list[str] = None
    ):
        if response_vars_names is None:
            if self.model.ndim > 2:
                return ['TIME', 'DISPLACEMENT_X', 'DISPLACEMENT_Y', 'DISPLACEMENT_Z', 'ACCELERATION_X',
                        'ACCELERATION_Y', 'ACCELERATION_Z']
            else:
                return ['TIME', 'DISPLACEMENT_X', 'DISPLACEMENT_Y', 'ACCELERATION_X', 'ACCELERATION_Y']
        else:
            if isinstance(response_vars_names, str):
                return [response_vars_names]
            else:
                return response_vars_names

    def _generate_random_field(self, matching_part: stem.model.BodyModelPart):
        pass

    def _set_up_rfs(self):
        rfs = []
        max = 5_557_383
        seeds = np.random.choice(range(max), self.num_simulations, replace=False)

        for i in range(len(seeds)):
            rf_generator = RandomFieldGenerator(
                n_dim=3,
                cov=0.1,
                model_name='Gaussian',
                v_scale_fluctuation=1,
                anisotropy=[0.5],
                angle=[0],
                seed=seeds[i],
            )
            rfs.append(rf_generator)
        return rfs

    def _get_model_part_by_name(self, model_part_name: str):
        for model_part in self.model.body_model_parts + self.model.process_model_parts:
            if model_part.name == model_part_name:
                return model_part
        raise ValueError(f"Model part {model_part} not found in the model")

    def _validate_output_coordinates(self, output_coordinates: list):
        model_points = self.model.gmsh_io.geo_data['points']

        # Extract the coordinates of the model points
        x_vals = [point[0] for point in model_points.values()]
        y_vals = [point[1] for point in model_points.values()]
        z_vals = [point[2] for point in model_points.values()]

        # Check for the min and max values
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        z_min, z_max = min(z_vals), max(z_vals)

        for output_point in output_coordinates:
            is_within_bounds = (
                    x_min <= output_point[0] <= x_max and
                    y_min <= output_point[1] <= y_max and
                    z_min <= output_point[2] <= z_max
            )
            if not is_within_bounds:
                raise ValueError(f"Output point {output_point} is outside the model bounds")
        return output_coordinates

    def _parse_output_var_names(self, output_var_names: list):
        parsed_variable_names = []

        for var_name in output_var_names:
            match var_name.lower().strip():
                case 'disp' | 'displacement' | 'd':
                    parsed_variable_names.append(NodalOutput.DISPLACEMENT)
                case 'vel' | 'velocity' | 'v':
                    parsed_variable_names.append(NodalOutput.VELOCITY)
                case 'acc' | 'acceleration' | 'a':
                    parsed_variable_names.append(NodalOutput.ACCELERATION)
                case 'total_disp' | 'total_displacement' | 'td':
                    parsed_variable_names.append(NodalOutput.TOTAL_DISPLACEMENT)
                case 'water' | 'water_pressure' | 'wp':
                    parsed_variable_names.append(NodalOutput.WATER_PRESSURE)
        return parsed_variable_names

    def _run_post_processing(self, ):
        output_file = self.output_files_dir

        with open(output_file, 'r') as f:
            results = json.load(f)
            self.calculation_results.append(results)

    def _prepare_calculation_results(self):
        d = self.response_vars
        temp = {}
        for var in d:
            temp[var] = []
        return temp

    def _set_up_rvs(self, model_part_stem: list | stem.model_part.ModelPart):
        if type(model_part_stem) is stem.model_part.ModelPart:
            unc_dist_type = self.unc_type_str
            match unc_dist_type.lower():
                case 'gaussian' | 'normal' | 'gauss':
                    dist = scipy.stats.norm(*self.dist_params)
                case 'lognormal' | 'log_normal' | 'log_norm':
                    dist = scipy.stats.lognorm(*self.dist_params)
                case 'uniform' | 'uni':
                    dist = scipy.stats.uniform(*self.dist_params)
                case _:
                    raise NotImplementedError(f"Uncertainty distribution type {unc_dist_type} not implemented")
            return dist.rvs(size=self.num_simulations,
                            random_state=self.random_state)
        else:
            # Sample random variates for the ModelPart object using Latin Hypercube Sampling
            raise NotImplementedError("Latin Hypercube Sampling not implemented yet")
        pass

    def _parse_load_direction(self, load_direction: str | None):
        if load_direction is None:
            return 1
        else:
            match load_direction.lower():
                case 'x':
                    return 0
                case 'y':
                    return 1
                case 'z':
                    return 2
                case _:
                    raise ValueError(f"Load direction {load_direction} not recognized")


if __name__ == '__main__':
    model = test_case_2d_model()
    project_params = test_case_2d_model_project_params()
    # Example usage for quantifying spatial variability of soil properties
    # user_variations = UserVariations(stem_model=model,
    #                                  project_params=project_params,
    #                                  input_files_dir='random_field_mc',
    #                                  output_var_names=['disp', 'acceleration'],
    #                                  model_part='layer1',
    #                                  property_name='YOUNG_MODULUS',
    #                                  unc_type='Gaussian',
    #                                  output_coordinates=[
    #                                      (0.0, 1.0, 0.0),
    #                                      (0.5, 1.0, 0.0),
    #                                  ],
    #                                  num_simulations=50)

    # Example usage for quantifying variability of load magnitude
    user_variations = UserVariations(stem_model=model,
                                     project_params=project_params,
                                     input_files_dir='random_field_mc',
                                     output_var_names=['disp', 'acceleration'],
                                     model_part='point_load',
                                     load_direction='Y',
                                     dist_params=(2, 1),
                                     unc_type='lognormal',
                                     output_coordinates=[
                                         (0.0, 1.0, 0.0),
                                         (0.5, 1.0, 0.0),
                                     ],
                                     num_simulations=5)
    user_variations.run(sim_type='MC')
    user_variations.plot_results(disp_coord='Y',
                                 node_number=4,
                                 show_plot=True)
    pass
