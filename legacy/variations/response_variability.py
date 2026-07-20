import json
import os
from copy import deepcopy

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import stem.model
import stem.model_part
import stem.solver
from stem.model_part import BodyModelPart, ModelPart
from stem.output import JsonOutputParameters, NodalOutput
from stem.stem import Stem

# Use version-agnostic adapters from clean/ so this works across STEM builds.
from clean.random_field_utils import create_random_field_generator
from clean.parameter_field_utils import create_parameter_field_parameters


class UserVariations():
    """
    Class for running uncertainty quantification on a model using user-defined stochastic parameters.

    Attributes
    ----------
    model : stem.model.Model
        Template model. A deep copy is made per simulation — the original is never mutated.
    project_params : stem.solver.Problem
        Calculation settings applied to each simulation copy.
    model_part : str
        Name of the model part with uncertain properties.
        BodyModelPart  → spatial random field on a soil property.
        ModelPart      → scalar random variate on a load/BC component.
    input_files_dir : str
        Directory where STEM writes input files and reads output.
    output_var_names : list[str]
        Variables to record. Accepted values: 'disp', 'vel', 'acc', 'total_disp', 'water'.
    unc_type : str
        Distribution for ModelPart variates: 'Gaussian', 'Lognormal', 'Uniform'.
    output_coordinates : list[tuple]
        Measurement point coordinates.
    num_simulations : int
        Number of Monte Carlo realisations.
    property_name : str | None
        STEM property name for BodyModelPart RF (e.g. 'YOUNG_MODULUS').
    dist_params : tuple
        Parameters forwarded to scipy.stats distribution (loc, scale, …).
    random_state : int | None
        Master seed. Controls both RF seed generation and scalar RV sampling,
        so results are fully reproducible when set.
    load_direction : str | None
        Load component to vary: 'X', 'Y', or 'Z'.
    output_files_dir : str | None
        Full path to the JSON output file. Defaults to <input_files_dir>/json_output.json.
    output_time_interval : float | None
        Output interval in seconds. Defaults to 0.0005.
    response_vars : str | list[str] | None
        Variable names to store in calculation_results. Defaults to displacement + acceleration.
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
            dist_params: tuple | None = None,
            random_state: int | None = None,
            load_direction: str | None = None,
            output_files_dir: str | None = None,
            output_time_interval: float | None = None,
            response_vars: str | list[str] | None = None,
    ):
        self.model = stem_model
        self.project_params = project_params
        self.model_part = model_part
        self.input_files_dir = input_files_dir

        if output_files_dir is None:
            self.output_files_dir = os.path.join(input_files_dir, 'json_output.json')
        else:
            self.output_files_dir = output_files_dir

        self.dist_params = (0, 1) if dist_params is None else dist_params
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

    def run(self, sim_type: str):
        if isinstance(self.model_part, list):
            raise NotImplementedError("Lists of model parts are not yet implemented")

        model_part_stem = self._get_model_part_by_name(self.model_part)

        if type(model_part_stem) is BodyModelPart:
            # Spatial random field on a soil property.
            rf_generators = self._set_up_rfs()

            for rf_generator in rf_generators:
                # Each iteration gets a fresh copy of the user's model.
                loop_model = deepcopy(self.model)

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

                field_params = create_parameter_field_parameters(
                    property_name=self.property_name,
                    function_type="json_file",
                    field_generator=rf_generator,
                )
                loop_model.add_field(part_name=self.model_part, field_parameters=field_params)
                loop_model.project_parameters = self.project_params

                analysis = Stem(loop_model, self.input_files_dir)
                analysis.write_all_input_files()
                analysis.run_calculation()
                self._run_post_processing()
                os.remove(self.output_files_dir)

        elif type(model_part_stem) is ModelPart:
            # Scalar random variate on a load / boundary condition.
            random_variates = self._set_up_rvs(model_part_stem)

            for rv in random_variates:
                loop_model = deepcopy(self.model)

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

                for process_model_part in loop_model.process_model_parts:
                    if process_model_part.name == model_part_stem.name:
                        process_model_part.parameters.value[self.load_direction] = rv
                        break

                loop_model.project_parameters = self.project_params
                analysis = Stem(loop_model, self.input_files_dir)
                analysis.write_all_input_files()
                analysis.run_calculation()
                self._run_post_processing()
                os.remove(self.output_files_dir)

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

        results_stacked = []
        for result in self.calculation_results:
            results_stacked.append(result[f'NODE_{node_number}'][response_variable])

        results_stacked_np = np.array(results_stacked)
        mean = np.mean(results_stacked_np, axis=0)
        ci_lower = np.percentile(results_stacked_np, 2.5, axis=0)
        ci_upper = np.percentile(results_stacked_np, 97.5, axis=0)
        time = self.calculation_results[0]['TIME']

        for row in results_stacked_np:
            response_time_ax.plot(time, row, c='k', alpha=0.05)
        response_time_ax.set_xlabel('Time [s]')
        response_time_ax.set_ylabel(f'{response_variable} [m]')

        ax_statplot.plot(time, mean, c='k', label='Mean')
        ax_statplot.fill_between(time, ci_lower, ci_upper, alpha=0.4, color='darkgray', label='95% CI')
        ax_statplot.set_xlabel('Time [s]')
        ax_statplot.set_ylabel(f'{response_variable} [m]')

        means_calculations = np.mean(results_stacked_np, axis=1)
        iqr = np.percentile(means_calculations, 75) - np.percentile(means_calculations, 25)
        bin_width = 2 * iqr / (len(means_calculations) ** (1 / 3))
        bins = max(1, int((means_calculations.max() - means_calculations.min()) / bin_width))

        hist_ax.hist(means_calculations, bins=bins, color='blue', alpha=0.5)
        hist_ax.axvline(np.mean(means_calculations), color='k', linestyle='dashed', linewidth=1)
        hist_ax.set_xlabel(f'Mean {response_variable} [m]')
        hist_ax.set_ylabel('Frequency')

        if name_of_the_model is not None:
            fig.suptitle(f'{response_variable} at node {node_number}')

        if save_fig:
            output_dir = os.path.dirname(self.output_files_dir)
            fig.savefig(os.path.join(output_dir, f'{response_variable}_node_{node_number}.jpeg'),
                        dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_up_rfs(self):
        """
        Create one RandomFieldGenerator per simulation.
        Seeds are derived from self.random_state so results are reproducible.
        The RF dimension matches the model (2D or 3D).
        """
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 5_557_383, size=self.num_simulations)

        rfs = []
        for seed in seeds:
            rf = create_random_field_generator(
                dim=self.model.ndim,
                cov=0.1,
                model_name='Gaussian',
                v_scale_fluctuation=1,
                anisotropy=[0.5],
                angle=[0],
                seed=int(seed),
            )
            rfs.append(rf)
        return rfs

    def _set_up_rvs(self, model_part_stem: stem.model_part.ModelPart):
        unc_dist_type = self.unc_type_str
        match unc_dist_type.lower():
            case 'gaussian' | 'normal' | 'gauss':
                dist = stats.norm(*self.dist_params)
            case 'lognormal' | 'log_normal' | 'log_norm':
                dist = stats.lognorm(*self.dist_params)
            case 'uniform' | 'uni':
                dist = stats.uniform(*self.dist_params)
            case _:
                raise NotImplementedError(f"Distribution type '{unc_dist_type}' not implemented")
        return dist.rvs(size=self.num_simulations, random_state=self.random_state)

    def _get_model_part_by_name(self, model_part_name: str):
        for mp in self.model.body_model_parts + self.model.process_model_parts:
            if mp.name == model_part_name:
                return mp
        raise ValueError(f"Model part '{model_part_name}' not found in the model")

    def _validate_output_coordinates(self, output_coordinates: list):
        model_points = self.model.gmsh_io.geo_data['points']
        x_vals = [p[0] for p in model_points.values()]
        y_vals = [p[1] for p in model_points.values()]
        z_vals = [p[2] for p in model_points.values()]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        z_min, z_max = min(z_vals), max(z_vals)

        for point in output_coordinates:
            if not (x_min <= point[0] <= x_max and
                    y_min <= point[1] <= y_max and
                    z_min <= point[2] <= z_max):
                raise ValueError(f"Output point {point} is outside model bounds")
        return output_coordinates

    def _parse_output_var_names(self, output_var_names: list):
        parsed = []
        for var_name in output_var_names:
            match var_name.lower().strip():
                case 'disp' | 'displacement' | 'd':
                    parsed.append(NodalOutput.DISPLACEMENT)
                case 'vel' | 'velocity' | 'v':
                    parsed.append(NodalOutput.VELOCITY)
                case 'acc' | 'acceleration' | 'a':
                    parsed.append(NodalOutput.ACCELERATION)
                case 'total_disp' | 'total_displacement' | 'td':
                    parsed.append(NodalOutput.TOTAL_DISPLACEMENT)
                case 'water' | 'water_pressure' | 'wp':
                    parsed.append(NodalOutput.WATER_PRESSURE)
        return parsed

    def _parse_response_vars(self, response_vars_names: str | list[str] | None):
        if response_vars_names is None:
            if self.model.ndim > 2:
                return ['TIME', 'DISPLACEMENT_X', 'DISPLACEMENT_Y', 'DISPLACEMENT_Z',
                        'ACCELERATION_X', 'ACCELERATION_Y', 'ACCELERATION_Z']
            else:
                return ['TIME', 'DISPLACEMENT_X', 'DISPLACEMENT_Y',
                        'ACCELERATION_X', 'ACCELERATION_Y']
        if isinstance(response_vars_names, str):
            return [response_vars_names]
        return response_vars_names

    def _run_post_processing(self):
        with open(self.output_files_dir, 'r') as f:
            self.calculation_results.append(json.load(f))

    def _parse_load_direction(self, load_direction: str | None):
        if load_direction is None:
            return 1
        match load_direction.lower():
            case 'x':
                return 0
            case 'y':
                return 1
            case 'z':
                return 2
            case _:
                raise ValueError(f"Load direction '{load_direction}' not recognised")
