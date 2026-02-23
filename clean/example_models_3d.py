import json
import os
import stem
from stem.additional_processes import ParameterFieldParameters
from stem.boundary import DisplacementConstraint
from stem.field_generator import RandomFieldGenerator
from stem.load import PointLoad, MovingLoad
from stem.model import Model
from stem.output import GaussPointOutput, NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.solver import AnalysisType, TimeIntegration, DisplacementConvergenceCriteria, NewtonRaphsonStrategy, \
    NewmarkScheme, StressInitialisationType, Amgcl, SolverSettings, Problem, SolutionType
from stem.stem import Stem
import numpy as np
import scipy.stats as stats
from scipy.stats import qmc

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import sensitivity analysis modules
from sensitivity_analysis import MorrisSensitivityAnalysis, SensitivityAnalysisRunner
from model_runner import ModelRunner, FastModelRunner
# Import sampling comparison module
from sampling_comparison import SamplingComparisonRunner
# Import plotting and post-processing utilities
from plotting import PlottingUtilities
from postprocessing import PostProcessingUtilities


"""
Contains test/example cases of models with defined soil layers
and project parameters.

This module provides functions for creating 3D geotechnical models with Monte Carlo
simulation capabilities, sensitivity analysis, and various sampling methods.
"""


# ============================================================================
# MATERIAL DEFINITION FUNCTIONS
# ============================================================================

def default_2d_soil_material() -> stem.soil_material.SoilMaterial:
    """
    Create a default 2D soil material with standard properties.
    
    Returns:
    --------
    SoilMaterial : Default soil material (density=2650 kg/m³, E=100 MPa, ν=0.3)
    """
    ndim = 2
    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=2650, POROSITY=0.3)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
    soil_material = SoilMaterial(name="soil",
                                 soil_formulation=soil_formulation,
                                 constitutive_law=constitutive_law,
                                 retention_parameters=SaturatedBelowPhreaticLevelLaw())
    return soil_material

def embankment_material() -> stem.soil_material.SoilMaterial:
    ndim = 2
    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1900, POROSITY=0.3)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
    soil_material = SoilMaterial(name="embankment",
                                 soil_formulation=soil_formulation,
                                 constitutive_law=constitutive_law,
                                 retention_parameters=SaturatedBelowPhreaticLevelLaw())
    return soil_material

def sand_soil() -> stem.soil_material.SoilMaterial:
    ndim = 2
    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1900, POROSITY=0.3)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=350e6, POISSON_RATIO=0.3)
    soil_material = SoilMaterial(name="sand",
                                 soil_formulation=soil_formulation,
                                 constitutive_law=constitutive_law,
                                 retention_parameters=SaturatedBelowPhreaticLevelLaw())
    return soil_material

def peat_soil() -> stem.soil_material.SoilMaterial:
    ndim = 2
    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1400, POROSITY=0.3)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=100e6, POISSON_RATIO=0.3)
    soil_material = SoilMaterial(name="peat",
                                 soil_formulation=soil_formulation,
                                 constitutive_law=constitutive_law,
                                 retention_parameters=SaturatedBelowPhreaticLevelLaw())
    return soil_material

def clay_soil() -> stem.soil_material.SoilMaterial:
    """Create clay material (density=1500 kg/m³, E=50 MPa, ν=0.3)."""
    ndim = 2
    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=1500, POROSITY=0.3)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=50e6, POISSON_RATIO=0.3)
    soil_material = SoilMaterial(name="clay",
                                 soil_formulation=soil_formulation,
                                 constitutive_law=constitutive_law,
                                 retention_parameters=SaturatedBelowPhreaticLevelLaw())
    return soil_material    
    
    
def create_material_with_properties(name: str, density: float, young_modulus: float, 
                                    poisson_ratio: float = 0.3, porosity: float = 0.3,
                                    ndim: int = 3) -> stem.soil_material.SoilMaterial:
    """
    Create a soil material with specified properties (helper function to reduce duplication).
    
    Parameters:
    -----------
    name : str
        Material name
    density : float
        Solid density in kg/m³
    young_modulus : float
        Young's modulus in Pa
    poisson_ratio : float, optional
        Poisson's ratio (default: 0.3)
    porosity : float, optional
        Porosity (default: 0.3)
    ndim : int, optional
        Number of dimensions (default: 3)
        
    Returns:
    --------
    SoilMaterial : Configured soil material
    """
    soil_formulation = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=density, POROSITY=porosity)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=young_modulus, POISSON_RATIO=poisson_ratio)
    return SoilMaterial(name=name, soil_formulation=soil_formulation,
                                 constitutive_law=constitutive_law,
                                 retention_parameters=SaturatedBelowPhreaticLevelLaw())


# ============================================================================
# PLOTTING AND POST-PROCESSING FUNCTIONS MOVED TO SEPARATE CLASSES
# See: plotting.py (PlottingUtilities class) and postprocessing.py (PostProcessingUtilities class)
# ============================================================================



def get_variable_definitions():
    """
    Get variable definitions for sampling. This function can be extended 
    to include more parameters in the future.
    
    Returns:
    --------
    dict : Dictionary containing variable definitions with their distributions
    """
    variables = {
        'clay_density': {'mean': 1500, 'cov': 0.05, 'dist': 'lognormal'},           # kg/m³
        'clay_young_modulus': {'mean': 50e6, 'cov': 0.05, 'dist': 'lognormal'},     # Pa
        'sand_density': {'mean': 2100, 'cov': 0.05, 'dist': 'lognormal'},           # kg/m³
        'sand_young_modulus': {'mean': 250e6, 'cov': 0.05, 'dist': 'lognormal'},    # Pa
        'embankment_density': {'mean': 1650, 'cov': 0.05, 'dist': 'lognormal'},     # kg/m³
        'embankment_young_modulus': {'mean': 100e6, 'cov': 0.05, 'dist': 'lognormal'}, # Pa
        'vertical_load': {'mean': -30000, 'std': 400, 'dist': 'normal'},          # N (Gaussian) - kept constant in sensitivity analysis
        'rayleigh_k': {'mean': 0.0002, 'cov': 0.05, 'dist': 'normal'},           # Rayleigh damping parameter k
        'rayleigh_m': {'mean': 0.5, 'cov': 0.05, 'dist': 'normal'}               # Rayleigh damping parameter m
    }
    return variables

         
def convert_uniform_samples_to_distributions(uniform_samples, variables):
    """
    Convert uniform [0,1] samples to their respective distributions.
    
    Parameters:
    -----------
    uniform_samples : np.ndarray
        Uniform samples of shape (n_samples, n_variables)
    variables : dict
        Variable definitions from get_variable_definitions()
        
    Returns:
    --------
    dict : Dictionary containing samples for each variable
    """
    results = {}
    
    for i, (var_name, params) in enumerate(variables.items()):
        if params['dist'] == 'normal':
            # Gaussian distribution: convert uniform [0,1] samples to normal distribution
            mean_val = params['mean']
            # Support both 'std' (for vertical_load) and 'cov' (for rayleigh parameters)
            if 'std' in params:
                std_val = params['std']  # Direct standard deviation (for vertical_load)
            elif 'cov' in params:
                std_val = mean_val * params['cov']  # COV-based standard deviation
            else:
                raise ValueError(f"Normal distribution for {var_name} requires either 'std' or 'cov'")
            
            # Use inverse CDF (percent point function) to convert uniform to normal
            normal_samples = stats.norm.ppf(uniform_samples[:, i], 
                                          loc=mean_val, 
                                          scale=std_val)
            results[var_name] = normal_samples
            
        elif params['dist'] == 'lognormal':
            # Lognormal distribution: used for material properties (always positive)
            mean_val = params['mean']
            cov = params['cov']  # Coefficient of variation
            
            # Calculate lognormal parameters from mean and COV
            # For lognormal: if X ~ LN(μ, σ²), then E[X] = exp(μ + σ²/2), Var[X] = E[X]²(exp(σ²) - 1)
            # We solve for μ and σ given mean and COV
            sigma_ln = np.sqrt(np.log(1 + cov**2))  # Standard deviation in log space
            mu_ln = np.log(mean_val) - 0.5 * sigma_ln**2  # Mean in log space
            
            # Convert uniform samples to lognormal using inverse CDF
            lognormal_samples = stats.lognorm.ppf(uniform_samples[:, i], 
                                                 s=sigma_ln,  # Shape parameter (std in log space)
                                                 scale=np.exp(mu_ln))  # Scale parameter (exp of mean in log space)
            results[var_name] = lognormal_samples
    
    return results


# Note: All sampling generation functions have been moved to sampling_comparison.py:
# - generate_random_samples() -> sampling_comparison.generate_random_samples()
# - generate_lhs_samples() -> sampling_comparison.generate_lhs_samples()
# - generate_sobol_samples() -> sampling_comparison.generate_sobol_samples()
# - generate_hammersley_samples() -> sampling_comparison.generate_hammersley_samples()
# - generate_optimized_lhs_samples() -> sampling_comparison.generate_optimized_lhs_samples()
# Use SamplingComparisonRunner class instead (from sampling_comparison import SamplingComparisonRunner)


# Note: generate_random_samples() has been moved to sampling_comparison.py
# Use SamplingComparisonRunner.generate_samples(method='random', ...) instead


# Note: print_statistics_table() has been moved to PostProcessingUtilities class in postprocessing.py
# Use PostProcessingUtilities.print_statistics_table() instead


def compare_sampling_methods(num_simulations=10, lhs_seed=42, show_plots=True, show_detailed_plots=True):
    """
    Compare multiple sampling methods: Random, LHS, Sobol, Halton, and Optimized LHS.
    
    This is a wrapper function that uses the SamplingComparisonRunner class.
    
    Parameters:
    -----------
    num_simulations : int
        Number of simulations to run for each method
    lhs_seed : int
        Seed for sampling reproducibility
    show_plots : bool, optional
        If True, show basic displacement plots. Default is True.
    show_detailed_plots : bool, optional
        If True, show detailed statistical analysis plots. Default is True.
        
    Returns:
    --------
    dict : Dictionary containing samples from all methods
    """
    # Use SamplingComparisonRunner class
    comparison_runner = SamplingComparisonRunner(get_variable_definitions_func=get_variable_definitions)
    
    # Compare all available methods
    all_samples = comparison_runner.compare_sampling_methods(
        num_simulations=num_simulations,
        seed=lhs_seed,
        show_plots=show_plots,
        show_detailed_plots=show_detailed_plots
    )
    
    return all_samples


# Note: All plotting functions have been moved to PlottingUtilities class in plotting.py:
# - create_comparison_plots_multiple() -> PlottingUtilities.create_comparison_plots_multiple()
# - create_histogram_comparison_multiple() -> PlottingUtilities.create_histogram_comparison_multiple()
# - create_pair_plot() -> PlottingUtilities.create_pair_plot()
# - create_comparison_plots() -> Use PlottingUtilities methods instead (legacy function removed)
# - create_histogram_comparison() -> Use PlottingUtilities methods instead (legacy function removed)
#
# Note: Post-processing functions have been moved to PostProcessingUtilities class in postprocessing.py:
# - print_summary_statistics() -> PostProcessingUtilities.print_summary_statistics()


def run_sensitivity_analysis(num_trajectories=10, num_levels=4, seed=42, 
                           output_variable='DISPLACEMENT_Y', extract_method='max',
                           use_fast_runner=True):
    """
    Run Morris sensitivity analysis on the 3D geotechnical model.
    
    Parameters:
    -----------
    num_trajectories : int
        Number of Morris trajectories (default: 10)
    num_levels : int
        Number of levels for Morris sampling (default: 4)
    seed : int
        Random seed for reproducibility (default: 42)
    output_variable : str
        Output variable to analyze (default: 'DISPLACEMENT_Y')
    extract_method : str
        Method to extract single value from time series ('max', 'min', 'mean', 'final')
    use_fast_runner : bool
        Whether to use the fast model runner with caching
        
    Returns:
    --------
    Dict : Sensitivity analysis results
    """
    print("="*80)
    print("MORRIS SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Initialize model runner
    if use_fast_runner:
        model_runner = FastModelRunner(input_files_dir='random_field_mc', simulation_id=0)
        print("Using FastModelRunner with caching")
    else:
        model_runner = ModelRunner(input_files_dir='random_field_mc', simulation_id=0)
        print("Using standard ModelRunner")
    
    # Create a wrapper function for the sensitivity analysis
    def model_wrapper(parameters):
        """Wrapper function for sensitivity analysis."""
        return model_runner.run_model_with_parameters(
            parameters, output_variable, extract_method
        )
    
    # Initialize sensitivity analysis runner
    sensitivity_runner = SensitivityAnalysisRunner(
        model_runner_func=model_wrapper,
        output_variable=output_variable
    )
    
    # Run Morris analysis
    sensitivity_results = sensitivity_runner.run_morris_analysis(
        num_trajectories=num_trajectories,
        num_levels=num_levels,
        seed=seed
    )
    
    # Print summary using post-processing utilities
    summary = sensitivity_runner.get_results_summary()
    PostProcessingUtilities.print_sensitivity_summary(summary)
    
    return sensitivity_results


def run_3d_model(use_lhs=True, lhs_seed=42, num_simulations=10, check_model_only=False, 
                 show_plots=True, show_detailed_plots=True, sampling_method=None):
    """
    Run 3D model with Monte Carlo simulations.
    
    Parameters:
    -----------
    use_lhs : bool, optional
        If True, use Latin Hypercube Sampling. If False, use random sampling.
        Default is True. (Deprecated: use sampling_method instead)
    lhs_seed : int, optional
        Seed for sampling reproducibility. Default is 42.
    num_simulations : int, optional
        Number of simulations to run. Default is 10.
    check_model_only : bool, optional
        If True, only run one simulation to check model setup. Default is False.
    show_plots : bool, optional
        If True, show basic displacement plots. Default is True.
    show_detailed_plots : bool, optional
        If True, show detailed statistical analysis plots. Default is True.
    sampling_method : str, optional
        Sampling method to use: 'random', 'lhs', 'sobol', 'halton', 'optimized_lhs'.
        If None, uses use_lhs parameter for backward compatibility.
    """
    NUM_SIMULATIONS = 1 if check_model_only else num_simulations
    # SEEDS = np.random.randint(0, 1000, NUM_SIMULATIONS)
    SEEDS = np.arange(NUM_SIMULATIONS)
    
    
    input_files_dir = 'random_field_mc'
    ndim = 3

    responses_var = {}
    responses_var['DISPLACEMENT_Y'] = []
    responses_var['DISPLACEMENT_X'] = []

    # Determine sampling method (backward compatibility with use_lhs)
    if sampling_method is None:
        sampling_method = 'lhs' if use_lhs else 'random'
    
    sampling_method = sampling_method.lower()
    
    # Use SamplingComparisonRunner to generate samples
    comparison_runner = SamplingComparisonRunner(get_variable_definitions_func=get_variable_definitions)
    all_samples = comparison_runner.generate_samples(sampling_method, NUM_SIMULATIONS, seed=lhs_seed)
    method_name = comparison_runner.method_map[sampling_method][0]
    
    # Extract VerticalLoad
    VerticalLoad = all_samples['vertical_load']
    
    # Create sampling method info
    sampling_method_name = method_name
    sampling_seed = lhs_seed if sampling_method != 'random' else "Multiple seeds per variable"
    
    # Initialize dictionary to collect all random variables from all simulations
    all_simulation_variables = {}

    for i in range(NUM_SIMULATIONS):
        if check_model_only:
            print(f"  Running model check simulation {i+1}/{NUM_SIMULATIONS}...")

        model = Model(ndim)

        soil = default_2d_soil_material()
        clay = clay_soil()
        sand = sand_soil()
        embankment = embankment_material()
        
        if sampling_method != 'random':
            # Use pre-generated samples (for all quasi-random methods)
            random_values = {
                "clay_density": all_samples['clay_density'][i],
                "clay_young_modulus": all_samples['clay_young_modulus'][i],
                "sand_density": all_samples['sand_density'][i],
                "sand_young_modulus": all_samples['sand_young_modulus'][i],
                "embankment_density": all_samples['embankment_density'][i],
                "embankment_young_modulus": all_samples['embankment_young_modulus'][i],
                "rayleigh_k": all_samples['rayleigh_k'][i],
                "rayleigh_m": all_samples['rayleigh_m'][i]
            }
        else:
            # Generate random values using the original method
            # Default mean values from material functions
            mean_density_clay = clay.soil_formulation.DENSITY_SOLID        # 1500 kg/m³
            mean_young_modulus_clay = clay.constitutive_law.YOUNG_MODULUS  # 50e6 Pa
            mean_density_sand = sand.soil_formulation.DENSITY_SOLID        # 1900 kg/m³
            mean_young_modulus_sand = sand.constitutive_law.YOUNG_MODULUS  # 350e6 Pa
            mean_density_embankment = embankment.soil_formulation.DENSITY_SOLID        # 1900 kg/m³
            mean_young_modulus_embankment = embankment.constitutive_law.YOUNG_MODULUS  # 100e6 Pa
            
            cov = 0.05  # 5% coefficient of variation for all properties
            
            # Generate all 6 random values using different seeds for independence
            seeds_offset = [100, 200, 300, 400, 500, 600]
            
            # Generate lognormal random values for all 6 variables
            random_values = {}
            for j, (material, property_type) in enumerate([
                ("clay", "density"), ("clay", "young_modulus"),
                ("sand", "density"), ("sand", "young_modulus"), 
                ("embankment", "density"), ("embankment", "young_modulus")
            ]):
                rng = np.random.default_rng(SEEDS[i] + seeds_offset[j])
                
                # Get mean value
                if material == "clay" and property_type == "density":
                    mean_val = mean_density_clay
                elif material == "clay" and property_type == "young_modulus":
                    mean_val = mean_young_modulus_clay
                elif material == "sand" and property_type == "density":
                    mean_val = mean_density_sand
                elif material == "sand" and property_type == "young_modulus":
                    mean_val = mean_young_modulus_sand
                elif material == "embankment" and property_type == "density":
                    mean_val = mean_density_embankment
                elif material == "embankment" and property_type == "young_modulus":
                    mean_val = mean_young_modulus_embankment
                
                # Calculate lognormal parameters
                sigma_ln = np.sqrt(np.log(1 + cov**2))
                mu_ln = np.log(mean_val) - 0.5 * sigma_ln**2
                
                # Generate random value
                random_values[f"{material}_{property_type}"] = rng.lognormal(mu_ln, sigma_ln)
            
            # Generate rayleigh_k and rayleigh_m (normal distribution with COV 5%)
            rng_rayleigh = np.random.default_rng(SEEDS[i] + 700)
            mean_rayleigh_k = 0.0002
            mean_rayleigh_m = 0.6
            std_rayleigh_k = mean_rayleigh_k * cov
            std_rayleigh_m = mean_rayleigh_m * cov
            random_values["rayleigh_k"] = rng_rayleigh.normal(loc=mean_rayleigh_k, scale=std_rayleigh_k)
            random_values["rayleigh_m"] = rng_rayleigh.normal(loc=mean_rayleigh_m, scale=std_rayleigh_m)

        # Override with deterministic values when running a single model check
        if check_model_only:
            random_values = {
                "clay_density": clay.soil_formulation.DENSITY_SOLID,
                "clay_young_modulus": clay.constitutive_law.YOUNG_MODULUS,
                "sand_density": sand.soil_formulation.DENSITY_SOLID,
                "sand_young_modulus": sand.constitutive_law.YOUNG_MODULUS,
                "embankment_density": embankment.soil_formulation.DENSITY_SOLID,
                "embankment_young_modulus": embankment.constitutive_law.YOUNG_MODULUS,
                "rayleigh_k": 0.0002,
                "rayleigh_m": 0.6
            }
            vertical_load_value = -30000.0
        else:
            vertical_load_value = VerticalLoad[i]
        
        # Collect all 9 variables for this simulation (7 original + 2 rayleigh)
        simulation_data = {
            "clay_density": random_values["clay_density"],
            "clay_young_modulus": random_values["clay_young_modulus"],
            "sand_density": random_values["sand_density"],
            "sand_young_modulus": random_values["sand_young_modulus"],
            "embankment_density": random_values["embankment_density"],
            "embankment_young_modulus": random_values["embankment_young_modulus"],
            "vertical_load": vertical_load_value,
            "rayleigh_k": random_values["rayleigh_k"],
            "rayleigh_m": random_values["rayleigh_m"]
        }
        
        # Store in the collection dictionary
        all_simulation_variables[f"simulation_{i}"] = simulation_data
        
        # Create materials with random properties using helper function
        clay_with_random_properties = create_material_with_properties(
            name="clay_random_properties",
            density=random_values["clay_density"],
            young_modulus=random_values["clay_young_modulus"],
            poisson_ratio=0.3,
            porosity=0.3,
            ndim=3
        )
        
        sand_with_random_properties = create_material_with_properties(
            name="sand_random_properties",
            density=random_values["sand_density"],
            young_modulus=random_values["sand_young_modulus"],
            poisson_ratio=0.3,
            porosity=0.3,
            ndim=3
        )
        
        embankment_with_random_properties = create_material_with_properties(
            name="embankment_random_properties",
            density=random_values["embankment_density"],
            young_modulus=random_values["embankment_young_modulus"],
            poisson_ratio=0.3,
            porosity=0.3,
            ndim=3
        )
        soil1_coordinates = [(-0.0, -2.0, 0.0), (5.0, -2.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        soil2_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
        embankment_coordinates = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0, 3.0, 0.0)]
        model.extrusion_length = 50
        model.add_soil_layer_by_coordinates(
            soil2_coordinates,
            clay_with_random_properties, "soil_layer_2")
        model.add_soil_layer_by_coordinates(
            soil1_coordinates,
            sand_with_random_properties, "soil_layer_1")
        model.add_soil_layer_by_coordinates(
            embankment_coordinates,
            embankment_with_random_properties, "embankment_layer"
        )

        load_coordinates = [(0.75, 3.0, 0.0), (0.75, 3.0, 50.0)]

        moving_load = MovingLoad(load=[0.0, vertical_load_value, 0.0], direction=[1, 1, 1], velocity=30, origin=[0.75, 3.0, 0.0],
                         offset=0.0)
        model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")
        
        # Define output coordinates (measurement points for displacement tracking)
        # TODO: verify coordinates match actual geometry
        output_coordinates = [
            (1.4, 3.0, 0.0),   # Surface point near embankment top
            (1.4, 3.0, 25.0),  # Surface point at midpoint (extrusion length = 50m)
        ]

        # Synchronize geometry and display for verification
        model.synchronise_geometry()
        model.show_geometry(show_surface_ids=True, show_line_ids=True)  
        
        # Define boundary conditions
        # Fixed boundary: all DOFs fixed (base of model)
        no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                            is_fixed=[True, True, True], value=[0, 0, 0])
        # Roller boundary: X and Z fixed, Y free (side boundaries)
        roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                                is_fixed=[True, False, True], value=[0, 0, 0])

        # Define output variables to track
        nodal_results = [NodalOutput.DISPLACEMENT]  # Track displacement at nodes
        gauss_point_results = [GaussPointOutput.YOUNG_MODULUS]  # Track Young's modulus at Gauss points   
       
        # Apply boundary conditions to geometry
        # Base is fully fixed (surface ID 1)
        model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
        # Sides have roller constraints (surface IDs: 2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17)
        model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17],
                                             roller_displacement_parameters, "sides_roller")
        model.set_mesh_size(element_size=1)  # Set mesh element size to 1m

        # Configure solver settings
        delta_time = 0.1  # Time step (seconds)
        analysis_type = AnalysisType.MECHANICAL  # Mechanical analysis (no flow)
        solution_type = SolutionType.DYNAMIC     # Dynamic analysis
        
        # Time integration parameters
        time_integration = TimeIntegration(start_time=0.0, end_time=2, delta_time=0.1, 
                                          reduction_factor=1.0, increase_factor=1.0)
        
        # Convergence criteria
        convergence_criterion = DisplacementConvergenceCriteria(
            displacement_relative_tolerance=1.0e-4,  # Relative tolerance
            displacement_absolute_tolerance=1.0e-9   # Absolute tolerance
        )
        
        # Solution strategy
        strategy_type = NewtonRaphsonStrategy()  # Nonlinear solver strategy
        scheme_type = NewmarkScheme()            # Time integration scheme
        linear_solver_settings = Amgcl()         # Linear solver (AMGCL)
        stress_initialisation_type = StressInitialisationType.NONE  # No initial stress
        
        # Configure solver with all settings including Rayleigh damping
        solver_settings = SolverSettings(
            analysis_type=analysis_type, 
            solution_type=solution_type,
                                 stress_initialisation_type=stress_initialisation_type,
                                 time_integration=time_integration,
            is_stiffness_matrix_constant=True,      # Assume constant stiffness
            are_mass_and_damping_constant=True,     # Assume constant mass/damping
                                 convergence_criteria=convergence_criterion,
            strategy_type=strategy_type, 
            scheme=scheme_type,
                                 linear_solver_settings=linear_solver_settings, 
            rayleigh_k=random_values["rayleigh_k"],  # Rayleigh damping parameter k (stiffness proportional)
            rayleigh_m=random_values["rayleigh_m"])  # Rayleigh damping parameter m (mass proportional)
        
        # Set up the problem
        problem = Problem(problem_name="calculate_moving_load_on_embankment_3d", 
                         number_of_threads=1, settings=solver_settings)
        model.project_parameters = problem
        
        # Note: Material properties (density and Young's modulus) are now generated 
        # as single random values per simulation, not as spatial random fields

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
        

        model.add_output_settings(
            part_name="porous_computational_model_part",
            output_name="vtk_output",
            output_dir="output",
            output_parameters=VtkOutputParameters(
                output_interval=1,
                file_format="ascii",
                nodal_results=nodal_results,
                gauss_point_results=gauss_point_results,
                output_control_type="step"
            )
        )

        stem = Stem(model, input_files_dir)
        stem.write_all_input_files()
        stem.run_calculation()

        path_to_results = os.path.join(input_files_dir, 'output', f"json_output_{i}.json")

        with open(path_to_results) as f:
            calculated_response = json.load(f)
            if i == 0:
                responses_var['TIME'] = calculated_response['TIME']
            results_node = calculated_response['NODE_22']     
            responses_var['DISPLACEMENT_Y'].append(results_node['DISPLACEMENT_Y'])
            responses_var['DISPLACEMENT_X'].append(results_node['DISPLACEMENT_X'])

        # Delete the output files
        # os.remove(path_to_results)

    # Save all simulation variables to a single JSON file
    final_variables_file = os.path.join(input_files_dir, "all_simulation_variables.json")
    
    # Create comprehensive data structure with metadata
    complete_data = {
        "metadata": {
            "total_simulations": NUM_SIMULATIONS,
            "description": "Random variables for all simulations in the 3D model",
            "sampling_method": sampling_method_name,
            "sampling_seed": sampling_seed,
            "distribution_type": "lognormal (except vertical_load, rayleigh_k, rayleigh_m which are Gaussian)",
            "coefficient_of_variation": "5%",
            "variables_per_simulation": 9,
            "variable_descriptions": {
                "clay_density": "Clay layer density (kg/m³), mean=1500",
                "clay_young_modulus": "Clay layer Young's modulus (Pa), mean=50e6",
                "sand_density": "Sand layer density (kg/m³), mean=1900", 
                "sand_young_modulus": "Sand layer Young's modulus (Pa), mean=350e6",
                "embankment_density": "Embankment layer density (kg/m³), mean=1900",
                "embankment_young_modulus": "Embankment layer Young's modulus (Pa), mean=100e6",
                "vertical_load": "Vertical load magnitude (N), mean=-10000, std=200 (Gaussian)",
                "rayleigh_k": "Rayleigh damping parameter k, mean=0.0002, COV=5% (Gaussian)",
                "rayleigh_m": "Rayleigh damping parameter m, mean=0.6, COV=5% (Gaussian)"
            }
        },
        "simulations": all_simulation_variables
    }
    
    with open(final_variables_file, 'w') as f:
        json.dump(complete_data, f, indent=2)
    
    print(f"All simulation variables saved to: {final_variables_file}")

    if not check_model_only and show_plots:
        if show_detailed_plots:
            # Use plotting utilities class
            PlottingUtilities.plot_response_output(responses_dict=responses_var, disp_coord='Y', 
                                                   name_of_the_model='test', NUM_SIMS=NUM_SIMULATIONS)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # Plot displacement histories with load annotations
        for i in range(NUM_SIMULATIONS):
            ax.plot(responses_var['TIME'], responses_var['DISPLACEMENT_X'][i], color='darkgray', alpha=0.3, label='Dis_X')
            # Annotate with vertical load value
            x_last = responses_var['TIME'][-10]
            y_last = responses_var['DISPLACEMENT_X'][i][-10]
            ax.text(x_last, y_last, f'Load={VerticalLoad[i]:.1f}', fontsize=8, ha='left', va='center', color='blue')
            ax.plot(responses_var['TIME'], responses_var['DISPLACEMENT_Y'][i], color='red', alpha=0.3, label='Dis_Y')

        
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper right')
        ax.set_ylabel('Displacement [m]')  
        plt.show()
    elif check_model_only and show_plots:
        print("Model check completed - creating plots...")
        
        # Create simple displacement plots for model check
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        # Plot X and Y displacements
        for i in range(NUM_SIMULATIONS):
            ax.plot(responses_var['TIME'], responses_var['DISPLACEMENT_X'][i], color='darkgray', alpha=0.7, label='Displacement X' if i == 0 else "")
            ax.plot(responses_var['TIME'], responses_var['DISPLACEMENT_Y'][i], color='red', alpha=0.7, label='Displacement Y' if i == 0 else "")
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Displacement [m]')
        ax.legend()
        ax.set_title('Model Check - Displacement vs Time')
        plt.show()
        
        # Use plotting utilities class for detailed analysis
        if show_detailed_plots:
            print("Creating detailed response analysis...")
            PlottingUtilities.plot_response_output(responses_dict=responses_var, disp_coord='Y', 
                                                   name_of_the_model='Model Check', NUM_SIMS=NUM_SIMULATIONS)
        
        print("Model check plots completed!")
    else:
        print("Model check completed - plotting skipped")

    pass

if __name__ == "__main__":
    # Choose what to run
    run_model_check = False        # Set to True to check model first
    run_sampling_comparison = False  # Set to False to skip
    run_sensitivity_flag = True     # Set to False to skip
    
    # Plotting control
    show_plots = True               # Set to True to show plots
    show_detailed_plots = True      # Set to True to show detailed analysis plots
    
    if run_model_check:
        print("\n" + "="*80)
        print("MODEL CHECK")
        print("="*80)
        
        # Run a single model to check setup and geometry
        print("\nRunning single 3D model to check setup...")
        run_3d_model(use_lhs=False, num_simulations=1, check_model_only=True, 
                    show_plots=show_plots, show_detailed_plots=show_detailed_plots)
        print("\nModel check completed!")
    
    if run_sampling_comparison:
        # Run comparison of sampling methods
        print("Running comparison of sampling methods...")
        all_samples = compare_sampling_methods(num_simulations=160, lhs_seed=42, 
                                                show_plots=show_plots, show_detailed_plots=show_detailed_plots)
    
    if run_sensitivity_flag:
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS")
        print("="*80)
        
        # Run Morris sensitivity analysis
        print("\nRunning Morris Sensitivity Analysis...")
        sensitivity_results = run_sensitivity_analysis(
            num_trajectories=1,  # Number of Morris trajectories
            num_levels=2,         # Number of levels
            seed=42,             # Random seed
            output_variable='DISPLACEMENT_Y',  # Variable to analyze
            extract_method='max', # Method to extract single value from time series
            use_fast_runner=True  # Use fast runner with caching
        )
        
        print("\nSensitivity analysis completed!")
        print("Results saved to: morris_sensitivity_results.json")
        print("Plots saved to: sensitivity_plots/ directory")
