"""
Model Runner module for sensitivity analysis.

This module provides classes to run the 3D geotechnical model with specific
parameter sets for sensitivity analysis purposes.
"""

import numpy as np
import json
import os
from typing import Dict, Any, Optional, Tuple
from stem.boundary import DisplacementConstraint
from stem.load import MovingLoad
from stem.model import Model
from stem.output import NodalOutput, JsonOutputParameters
from stem.solver import AnalysisType, TimeIntegration, DisplacementConvergenceCriteria, NewtonRaphsonStrategy, \
    NewmarkScheme, StressInitialisationType, Amgcl, SolverSettings, Problem, SolutionType
from stem.stem import Stem

# Import processing function
# Optional dependency: sensitivity post-processing relies on SignalProcessingTools.
# This repository is also used for "model check" runs where we only need to write VTK.
# Therefore we import post-processing utilities lazily/defensively.
try:
    from process_data import process_response_data
except ModuleNotFoundError:  # e.g. missing `SignalProcessingTools`
    process_response_data = None  # type: ignore[assignment]

# RF helper makes RandomFieldGenerator construction robust across STEM builds.
from random_field_utils import create_random_field_generator

from parameter_field_utils import create_parameter_field_parameters


class ModelRunner:
    """
    Class to run the 3D geotechnical model with specific parameter sets.

    This class is designed to work with sensitivity analysis by providing
    a clean interface to run the model with given parameter values.
    """

    # Geometry constants — override in subclasses to switch model size
    SOIL_X_MAX   = 5.0    # cross-track soil width (m)
    TRACK_LENGTH = 50.0   # along-track extrusion length (m)
    END_TIME     = 2.0    # analysis end time (s)

    def __init__(self, input_files_dir: str = 'random_field_mc', simulation_id: int = 0):
        """
        Initialize the model runner.
        
        Parameters:
        -----------
        input_files_dir : str
            Directory for input files
        simulation_id : int
            ID for this simulation run
        """
        self.input_files_dir = input_files_dir
        self.simulation_id = simulation_id
        self.ndim = 3
        self.rf_seed = 14     # random-field seed; override to vary RF realisations
        self.apply_rf = False # set to True only for explicit RF ensemble studies
        
    def run_model_with_parameters(self, parameters: np.ndarray,
                                 output_variable: str = 'DISPLACEMENT_Y',
                                 extract_method: str = 'max',
                                 output_mode: str = 'velocity') -> Dict[str, Any]:
        """
        Run the 3D model and return all response metrics in one dict.

        output_mode controls which metric becomes 'summary_value' (the SA target).
          'velocity'     -> summary_value = V_eff_max  (default)
          'v_y_max'      -> summary_value = V_y_max
          'psd'          -> summary_value = PSD_max
          'displacement' -> summary_value = max |DISPLACEMENT_Y|

        Returned keys (always present):
          summary_value, disp_y_max, response_y, response_x,
          V_y_max, V_eff_max, PSD_max, Freq_PSD_max,
          time, velocity_y, v_eff, frequency_Pxx, Pxx
        """
        clay_density, clay_young_modulus, sand_density, sand_young_modulus, \
        embankment_density, embankment_young_modulus, vertical_load, \
        rayleigh_k, rayleigh_m = parameters

        model = self._create_model_with_parameters(
            clay_density, clay_young_modulus, sand_density, sand_young_modulus,
            embankment_density, embankment_young_modulus, vertical_load,
            rayleigh_k, rayleigh_m
        )

        try:
            results = self._run_analysis(model)
            time_array, velocity_y_array = self._extract_velocity_from_results(results)

            # --- Displacement ---
            _, _, response_y, response_x = self._extract_output_value(
                results, 'DISPLACEMENT_Y', extract_method
            )
            disp_y_max = float(np.nanmax(np.abs(response_y))) if response_y is not None else np.nan

            # --- Velocity metrics from process_data ---
            if process_response_data is not None:
                pd = process_response_data(time_array, velocity_y_array)
            else:
                pd = None

            # --- summary_value for sensitivity analysis ---
            # output_mode options:
            #   'velocity'     -> V_eff_max  (root-mean-square effective velocity, mm/s)
            #   'v_y_max'      -> V_y_max    (peak instantaneous velocity in Y, mm/s)
            #   'psd'          -> PSD_max    (peak power spectral density, (mm/s)^2/Hz)
            #   'displacement' -> max |DISPLACEMENT_Y| (m)
            mode = output_mode.lower()
            if mode == 'displacement':
                summary_value = disp_y_max
            elif mode == 'v_y_max':
                summary_value = pd['V_y_max'] if pd is not None else np.nan
            elif mode == 'psd':
                summary_value = pd['PSD_max'] if pd is not None else np.nan
            else:  # 'velocity' -> V_eff_max (default)
                summary_value = pd['V_eff_max'] if pd is not None else np.nan

            return {
                'summary_value': summary_value,
                # displacement
                'disp_y_max': disp_y_max,
                'response_y': response_y,
                'response_x': response_x,
                # velocity metrics (from process_data)
                'V_y_max':       pd['V_y_max']        if pd else np.nan,
                'V_eff_max':     pd['V_eff_max']       if pd else np.nan,
                'PSD_max':       pd['PSD_max']         if pd else np.nan,
                'Freq_PSD_max':  pd['Freq_PSD_max']    if pd else np.nan,
                'time':          time_array,
                'velocity_y':    pd['velocity_y']      if pd else velocity_y_array * 1000,
                'v_eff':         pd['v_eff']           if pd else None,
                'frequency_Pxx': pd['frequency_Pxx']   if pd else None,
                'Pxx':           pd['Pxx']             if pd else None,
            }

        except Exception as e:
            print(f"Error running model with parameters: {e}")
            import traceback
            traceback.print_exc()
            return {
                'summary_value': np.nan,
                'disp_y_max': np.nan,
                'response_y': None,
                'response_x': None,
                'V_y_max': np.nan,
                'V_eff_max': np.nan,
                'PSD_max': np.nan,
                'Freq_PSD_max': np.nan,
                'time': None,
                'velocity_y': None,
                'v_eff': None,
                'frequency_Pxx': None,
                'Pxx': None,
            }
    
    def _create_model_with_parameters(self, clay_density: float, clay_young_modulus: float,
                                    sand_density: float, sand_young_modulus: float,
                                    embankment_density: float, embankment_young_modulus: float,
                                    vertical_load: float, rayleigh_k: float,
                                    rayleigh_m: float) -> Model:
        """
        Create the 3D model with specified parameters.
        
        Returns:
        --------
        Model : Configured STEM model
        """
        # Create model
        model = Model(self.ndim)
        model.set_mesh_size(1)
        
        # Import the helper function to reduce code duplication
        from example_models_3d import create_material_with_properties
        
        # Create materials with specified properties using shared helper function
        clay_material = create_material_with_properties(
            name="clay_sensitivity",
            density=clay_density,
            young_modulus=clay_young_modulus,
            poisson_ratio=0.3,
            porosity=0.3,
            ndim=3
        )
        
        sand_material = create_material_with_properties(
            name="sand_sensitivity",
            density=sand_density,
            young_modulus=sand_young_modulus,
            poisson_ratio=0.3,
            porosity=0.3,
            ndim=3
        )
        
        embankment_material = create_material_with_properties(
            name="embankment_sensitivity",
            density=embankment_density,
            young_modulus=embankment_young_modulus,
            poisson_ratio=0.3,
            porosity=0.3,
            ndim=3
        )
        
        # Define geometry coordinates
        soil1_coordinates = [(0.0, -2.0, 0.0), (self.SOIL_X_MAX, -2.0, 0.0), (self.SOIL_X_MAX, 1.0, 0.0), (0.0, 1.0, 0.0)]
        soil2_coordinates = [(0.0, 1.0, 0.0), (self.SOIL_X_MAX, 1.0, 0.0), (self.SOIL_X_MAX, 2.0, 0.0), (0.0, 2.0, 0.0)]
        embankment_coordinates = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0, 3.0, 0.0)]

        # Add soil layers
        model.extrusion_length = self.TRACK_LENGTH
        model.add_soil_layer_by_coordinates(soil1_coordinates, sand_material, "soil_layer_1")
        model.add_soil_layer_by_coordinates(soil2_coordinates, clay_material, "soil_layer_2")
        model.add_soil_layer_by_coordinates(embankment_coordinates, embankment_material, "embankment_layer")

        # Spatial random field for Young's modulus on `soil_layer_2`
        if self.apply_rf:
            random_field_generator = create_random_field_generator(
                dim=3,
                cov=0.1,
                model_name="Gaussian",
                v_scale_fluctuation=1,
                anisotropy=[10.0],
                angle=[0],
                seed=self.rf_seed,
            )
            field_parameters_json = create_parameter_field_parameters(
                property_name="YOUNG_MODULUS",
                function_type="json_file",
                field_generator=random_field_generator,
            )
            model.add_field(part_name="soil_layer_2", field_parameters=field_parameters_json)
        
        # Add loads
        load_coordinates = [(0.75, 3.0, 0.0), (0.75, 3.0, self.TRACK_LENGTH)]
        moving_load = MovingLoad(load=[0.0, vertical_load, 0.0], direction_signs=[1, 1, 1],
                               velocity=30, origin=[0.75, 3.0, 0.0], offset=0.0)
        model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")
        
        # Add boundary conditions
        no_displacement_parameters = DisplacementConstraint(
            is_fixed=[True, True, True], value=[0, 0, 0]
        )
        roller_displacement_parameters = DisplacementConstraint(
            is_fixed=[True, False, True], value=[0, 0, 0]
        )
        
        model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
        model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17],
                                                   roller_displacement_parameters, "sides_roller")
        
        # Set mesh size
        model.set_mesh_size(element_size=1.0)
        
        # Configure solver settings
        analysis_type = AnalysisType.MECHANICAL
        solution_type = SolutionType.DYNAMIC
        time_integration = TimeIntegration(start_time=0.0, end_time=self.END_TIME, delta_time=0.1,
                                         reduction_factor=1.0, increase_factor=1.0)
        convergence_criterion = DisplacementConvergenceCriteria(
            displacement_relative_tolerance=1.0e-4, displacement_absolute_tolerance=1.0e-9
        )
        strategy_type = NewtonRaphsonStrategy()
        scheme_type = NewmarkScheme()
        linear_solver_settings = Amgcl()
        stress_initialisation_type = StressInitialisationType.NONE
        
        solver_settings = SolverSettings(
            analysis_type=analysis_type, solution_type=solution_type,
            stress_initialisation_type=stress_initialisation_type,
            time_integration=time_integration,
            is_stiffness_matrix_constant=True, are_mass_and_damping_constant=True,
            convergence_criteria=convergence_criterion,
            strategy_type=strategy_type, scheme=scheme_type,
            linear_solver_settings=linear_solver_settings, rayleigh_k=rayleigh_k, rayleigh_m=rayleigh_m
        )
        
        # Set up problem
        problem = Problem(problem_name=f"sensitivity_analysis_{self.simulation_id}", 
                         number_of_threads=1, settings=solver_settings)
        model.project_parameters = problem
        
        return model
    
    def _run_analysis(self, model: Model) -> Dict[str, Any]:
        """
        Run the analysis and return results.
        
        Parameters:
        -----------
        model : Model
            Configured STEM model
            
        Returns:
        --------
        Dict[str, Any] : Analysis results
        """
        # Define output coordinates (updated from original location for better measurement)
        output_coordinates = [
            (3.0, 2.0, 0.0),                       # start of embankment
            (3.0, 2.0, self.TRACK_LENGTH / 2.0),   # midpoint
        ]
        
        # Add output settings
        nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]
        
        model.add_output_settings_by_coordinates(
            coordinates=output_coordinates,
            part_name="sensitivity_output",
            output_parameters=JsonOutputParameters(
                output_interval=0.05,
                nodal_results=nodal_results,
                gauss_point_results=[],
            ),
            output_dir="output",
            output_name=f"sensitivity_output_{self.simulation_id}",
        )
        
        # Run STEM
        stem_runner = Stem(model, self.input_files_dir)
        stem_runner.write_all_input_files()
        stem_runner.run_calculation()
        
        # Read results
        path_to_results = os.path.join(self.input_files_dir, 'output', f"sensitivity_output_{self.simulation_id}.json")
        
        try:
            with open(path_to_results) as f:
                results = json.load(f)
            return results
        except FileNotFoundError:
            print(f"Results file not found: {path_to_results}")
            return {}
    
    def _extract_velocity_from_results(self, results: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract time and velocity Y from STEM results.
        Prefers VELOCITY_Y; if absent, derives velocity from DISPLACEMENT_Y.
        """
        if not results:
            raise ValueError("Empty results dictionary")

        # STEM assigns node IDs internally and they vary between run types.
        # Select the node with the highest z-coordinate (midpoint of the extrusion).
        node_keys = [k for k in results if k.startswith('NODE_') and isinstance(results[k], dict)]
        if not node_keys:
            raise ValueError(f"No node data found. Keys: {list(results.keys())}")

        def _z(key):
            coords = results[key].get('COORDINATES', [0, 0, 0])
            return coords[2] if len(coords) > 2 else 0.0

        node_key = max(node_keys, key=_z)
        node_data = results[node_key]
        if node_data is None:
            raise ValueError(f"No node data found. Keys: {list(results.keys())}")

        time_values = results.get('TIME')
        if time_values is None:
            raise ValueError("TIME not found in results")

        time_array = np.asarray(time_values, dtype=float)
        velocity_y = node_data.get('VELOCITY_Y')

        if velocity_y is not None:
            velocity_y_array = np.asarray(velocity_y, dtype=float)
        else:
            disp_y = node_data.get('DISPLACEMENT_Y')
            if disp_y is None:
                raise ValueError(f"Neither VELOCITY_Y nor DISPLACEMENT_Y in {node_key}")
            d = np.asarray(disp_y, dtype=float)
            velocity_y_array = np.gradient(d, time_array)

        return time_array, velocity_y_array

    def _extract_and_process_velocity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract velocity, process via process_response_data. Kept for backward compatibility."""
        time_array, velocity_y_array = self._extract_velocity_from_results(results)
        if process_response_data is None:
            raise ModuleNotFoundError(
                "process_response_data is unavailable (likely missing dependency "
                "`SignalProcessingTools`). Install it to enable velocity/PSD post-processing."
            )
        return process_response_data(time_array, velocity_y_array)

    def _extract_output_value(self, results: Dict[str, Any], output_variable: str, 
                            extract_method: str) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract response information from the time series results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Analysis results
        output_variable : str
            Variable to extract
        extract_method : str
            Method to extract single value ('max', 'min', 'mean', 'final')
            
        Returns:
        --------
        tuple : (summary_value, time_values, response_y, response_x)
        """
        if not results:
            return np.nan, None, None, None
        
        # Select the node with the highest z-coordinate (midpoint of the extrusion).
        node_keys = [k for k in results if k.startswith('NODE_') and isinstance(results[k], dict)]
        if not node_keys:
            return np.nan, None, None, None

        def _z(key):
            coords = results[key].get('COORDINATES', [0, 0, 0])
            return coords[2] if len(coords) > 2 else 0.0

        node_key = max(node_keys, key=_z)
        node_data = results.get(node_key)
        
        if node_data is None or output_variable not in node_data:
            print(f"Output variable {output_variable} not found in results")
            return np.nan, None, None, None
        
        # Extract time series data
        time_series = node_data.get(output_variable)
        time_values = results.get('TIME', None)
        response_y = node_data.get('DISPLACEMENT_Y')
        response_x = node_data.get('DISPLACEMENT_X')

        if not time_series:
            return np.nan, None, None, None
        
        # Extract summary value based on specified method
        if extract_method == 'max':
            summary_value = float(np.max(time_series))  # Maximum value over time
        elif extract_method == 'min':
            summary_value = float(np.min(time_series))  # Minimum value over time
        elif extract_method == 'mean':
            summary_value = float(np.mean(time_series))  # Mean value over time
        elif extract_method == 'final':
            summary_value = float(time_series[-1])  # Final value in time series
        else:
            print(f"Unknown extract method: {extract_method}, using final value")
            summary_value = float(time_series[-1])

        # Convert to numpy arrays for consistency
        response_array_y = np.asarray(response_y if response_y is not None else time_series, dtype=float)
        response_array_x = np.asarray(response_x, dtype=float) if response_x is not None else None

        # Handle time array: use provided or create default
        if time_values is not None:
            time_array = np.asarray(time_values, dtype=float)
        else:
            # Create default time array if not provided
            length = len(time_series)
            time_array = np.linspace(0, length - 1, num=length)

        return summary_value, time_array, response_array_y, response_array_x


class FastModelRunner(ModelRunner):
    """
    Fast version of the model runner that skips file I/O for sensitivity analysis.
    
    This version is optimized for sensitivity analysis where many model runs
    are needed and file I/O can be a bottleneck.
    """
    
    def __init__(self, input_files_dir: str = 'random_field_mc', simulation_id: int = 0):
        super().__init__(input_files_dir, simulation_id)
        self.results_cache = {}
    
    def run_model_with_parameters(self, parameters: np.ndarray,
                                 output_variable: str = 'DISPLACEMENT_Y',
                                 extract_method: str = 'max',
                                 output_mode: str = 'velocity') -> Dict[str, Any]:
        """
        Fast version that caches results by parameter hash.
        Cache key includes output_mode so different modes don't collide.
        """
        param_hash = hash((tuple(parameters), output_mode))

        if param_hash in self.results_cache:
            return self.results_cache[param_hash]

        result = super().run_model_with_parameters(parameters, output_variable, extract_method, output_mode)

        self.results_cache[param_hash] = result
        return result