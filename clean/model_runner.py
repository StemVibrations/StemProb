"""
Model Runner module for sensitivity analysis.

This module provides classes to run the 3D geotechnical model with specific
parameter sets for sensitivity analysis purposes.
"""

import numpy as np
import json
import os
from typing import Dict, Any, Optional, Tuple
import stem
from stem.additional_processes import ParameterFieldParameters
from stem.boundary import DisplacementConstraint
from stem.field_generator import RandomFieldGenerator
from stem.load import MovingLoad
from stem.model import Model
from stem.output import GaussPointOutput, NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.solver import AnalysisType, TimeIntegration, DisplacementConvergenceCriteria, NewtonRaphsonStrategy, \
    NewmarkScheme, StressInitialisationType, Amgcl, SolverSettings, Problem, SolutionType
from stem.stem import Stem

# Import processing function
from process_data import process_response_data


class ModelRunner:
    """
    Class to run the 3D geotechnical model with specific parameter sets.
    
    This class is designed to work with sensitivity analysis by providing
    a clean interface to run the model with given parameter values.
    """
    
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
        
    def run_model_with_parameters(self, parameters: np.ndarray, 
                                 output_variable: str = 'DISPLACEMENT_Y',
                                 extract_method: str = 'max') -> Dict[str, Any]:
        """
        Run the 3D model with given parameters and return processed response information.
        
        Parameters:
        -----------
        parameters : np.ndarray
            Array of 9 parameters [clay_density, clay_young_modulus, sand_density,
                                  sand_young_modulus, embankment_density,
                                  embankment_young_modulus, vertical_load,
                                  rayleigh_k, rayleigh_m]
        output_variable : str
            Variable to extract from results (default: 'DISPLACEMENT_Y')
            Note: For velocity-based processing, use 'VELOCITY_Y'
        extract_method : str
            Method to extract single value from time series ('max', 'min', 'mean', 'final')
            Note: This is kept for backward compatibility but velocity processing is now used
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'V_y_max': Maximum absolute velocity Y (mm/s)
            - 'V_eff_max': Maximum effective velocity (mm/s)
            - 'PSD_max': Maximum PSD ((mm/s)^2/Hz)
            - 'Freq_PSD_max': Frequency at max PSD (Hz)
            - 'summary_value': V_eff_max (for backward compatibility with sensitivity analysis)
            - 'time': np.ndarray of time values
            - 'response_y': np.ndarray of velocity Y values over time (mm/s)
            - 'response_x': np.ndarray of displacement X values over time (for backward compatibility)
            - 'processed_data': Full processed data dictionary from process_response_data
        """
        # Unpack parameters
        clay_density, clay_young_modulus, sand_density, sand_young_modulus, \
        embankment_density, embankment_young_modulus, vertical_load, \
        rayleigh_k, rayleigh_m = parameters
        
        # Create model
        model = self._create_model_with_parameters(
            clay_density, clay_young_modulus, sand_density, sand_young_modulus,
            embankment_density, embankment_young_modulus, vertical_load,
            rayleigh_k, rayleigh_m
        )
        
        # Run analysis
        try:
            results = self._run_analysis(model)
            
            # Extract and process velocity data
            processed_data = self._extract_and_process_velocity(results)

            # Return processed metrics
            return {
                'V_y_max': processed_data['V_y_max'],
                'V_eff_max': processed_data['V_eff_max'],
                'PSD_max': processed_data['PSD_max'],
                'Freq_PSD_max': processed_data['Freq_PSD_max'],
                'summary_value': processed_data['V_eff_max'],  # For backward compatibility
                'time': processed_data['time'],
                'response_y': processed_data['velocity_y'],  # Velocity Y in mm/s
                'response_x': None,  # Not used for velocity processing
                'processed_data': processed_data
            }
            
        except Exception as e:
            print(f"Error running model with parameters: {e}")
            import traceback
            traceback.print_exc()
            return {
                'V_y_max': np.nan,
                'V_eff_max': np.nan,
                'PSD_max': np.nan,
                'Freq_PSD_max': np.nan,
                'summary_value': np.nan,
                'time': None,
                'response_y': None,
                'response_x': None,
                'processed_data': None
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
        soil1_coordinates = [(0.0, -2.0, 0.0), (5.0, -2.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        soil2_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
        embankment_coordinates = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0, 3.0, 0.0)]
        
        # Add soil layers
        model.extrusion_length = 50
        model.add_soil_layer_by_coordinates(soil1_coordinates, sand_material, "soil_layer_1")
        model.add_soil_layer_by_coordinates(soil2_coordinates, clay_material, "soil_layer_2")
        model.add_soil_layer_by_coordinates(embankment_coordinates, embankment_material, "embankment_layer")
        
        # Add loads
        load_coordinates = [(0.75, 3.0, 0.0), (0.75, 3.0, 50.0)]
        moving_load = MovingLoad(load=[0.0, vertical_load, 0.0], direction=[1, 1, 1], 
                               velocity=30, origin=[0.75, 3.0, 0.0], offset=0.0)
        model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")
        
        # Add boundary conditions
        no_displacement_parameters = DisplacementConstraint(
            active=[True, True, True], is_fixed=[True, True, True], value=[0, 0, 0]
        )
        roller_displacement_parameters = DisplacementConstraint(
            active=[True, True, True], is_fixed=[True, False, True], value=[0, 0, 0]
        )
        
        model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
        model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17],
                                                   roller_displacement_parameters, "sides_roller")
        
        # Set mesh size
        model.set_mesh_size(element_size=1.0)
        
        # Configure solver settings
        analysis_type = AnalysisType.MECHANICAL
        solution_type = SolutionType.DYNAMIC
        time_integration = TimeIntegration(start_time=0.0, end_time=2, delta_time=0.1, 
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
            (3.0, 2.0, 0.0),    # Surface point at start of embankment
            (3.0, 2.0, 25.0),   # Surface point at midpoint (extrusion length = 50m)
        ]
        
        # Add output settings
        nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]
        gauss_point_results = [GaussPointOutput.YOUNG_MODULUS]
        
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
    
    def _extract_and_process_velocity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract velocity data from results and process it to compute key metrics.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Analysis results containing TIME and node data with VELOCITY_Y
            
        Returns:
        --------
        Dict[str, Any]
            Processed data dictionary from process_response_data
        """
        if not results:
            raise ValueError("Empty results dictionary")
        
        # Find the node with data (typically NODE_22 for 3D model at output coordinates)
        node_key = None
        preferred_node_id = "NODE_22"

        if preferred_node_id in results:
            node_key = preferred_node_id
        else:
            # Fallback: search for any node key
            for key in results:
                if key.startswith('NODE_') and isinstance(results[key], dict):
                    node_key = key
                    break

        node_data = results.get(node_key) if node_key else None
        
        if node_data is None:
            raise ValueError(f"No node data found in results. Available keys: {list(results.keys())}")
        
        # Extract time and velocity data
        time_values = results.get('TIME', None)
        velocity_y = node_data.get('VELOCITY_Y', None)
        
        if time_values is None:
            raise ValueError("TIME not found in results")
        if velocity_y is None:
            raise ValueError(f"VELOCITY_Y not found in node {node_key}. Available keys: {list(node_data.keys())}")
        
        # Convert to numpy arrays
        time_array = np.asarray(time_values, dtype=float)
        velocity_y_array = np.asarray(velocity_y, dtype=float)
        
        # Process the velocity data
        processed_data = process_response_data(time_array, velocity_y_array)
        
        return processed_data
    
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
        
        # Find the node with data (typically NODE_22 for 3D model at output coordinates)
        node_key = None
        preferred_node_id = "NODE_22"

        if preferred_node_id in results:
            node_key = preferred_node_id
        else:
            # Fallback: search for any node key
            for key in results:
                if key.startswith('NODE_') and isinstance(results[key], dict):
                    node_key = key
                    break

        node_data = results.get(node_key) if node_key else None
        
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
                                 extract_method: str = 'max') -> float:
        """
        Fast version that uses cached results when possible.
        """
        # Create a hash of parameters for caching
        param_hash = hash(tuple(parameters))
        
        if param_hash in self.results_cache:
            return self.results_cache[param_hash]
        
        # Run the full model
        result = super().run_model_with_parameters(parameters, output_variable, extract_method)
        
        # Cache the result
        self.results_cache[param_hash] = result
        
        return result