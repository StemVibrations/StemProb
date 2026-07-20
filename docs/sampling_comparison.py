"""
Sampling comparison module for geotechnical model analysis.

This module contains classes and functions for comparing different sampling methods
used in Monte Carlo simulations, including Random, LHS, Sobol, Halton, and Optimized LHS.
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import qmc
from typing import Dict, Callable, Optional

# Import plotting and post-processing utilities
from plotting import PlottingUtilities
from postprocessing import PostProcessingUtilities


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


def generate_random_samples(num_simulations, seed=None, get_variable_definitions_func: Optional[Callable] = None):
    """
    Generate Random Sampling samples (for compatibility with other methods).
    
    Parameters:
    -----------
    num_simulations : int
        Number of simulations to generate
    seed : int, optional
        Random seed for reproducibility (each variable uses different seed offset)
    get_variable_definitions_func : callable, optional
        Function to get variable definitions. If None, imports from example_models_3d
        
    Returns:
    --------
    dict : Dictionary containing random samples for all variables
    """
    if get_variable_definitions_func is None:
        from example_models_3d import get_variable_definitions
        get_variable_definitions_func = get_variable_definitions
    
    variables = get_variable_definitions_func()
    n_dim = len(variables)
    
    # For random sampling, we generate independent random samples for each variable
    # using different seeds for each variable to ensure independence
    results = {}
    
    for i, (var_name, params) in enumerate(variables.items()):
        # Use different seed offset for each variable
        var_seed = (seed + i * 1000) if seed is not None else None
        rng = np.random.default_rng(var_seed)
        
        if params['dist'] == 'normal':
            mean_val = params['mean']
            if 'std' in params:
                std_val = params['std']
            elif 'cov' in params:
                std_val = mean_val * params['cov']
            else:
                raise ValueError(f"Normal distribution for {var_name} requires either 'std' or 'cov'")
            
            samples = rng.normal(loc=mean_val, scale=std_val, size=num_simulations)
            results[var_name] = samples
            
        elif params['dist'] == 'lognormal':
            mean_val = params['mean']
            cov = params['cov']
            
            # Calculate lognormal parameters
            sigma_ln = np.sqrt(np.log(1 + cov**2))
            mu_ln = np.log(mean_val) - 0.5 * sigma_ln**2
            
            # Generate lognormal samples
            samples = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=num_simulations)
        
        results[var_name] = samples
    
    return results


def generate_lhs_samples(num_simulations, seed=None, get_variable_definitions_func: Optional[Callable] = None):
    """
    Generate Latin Hypercube Sampling (LHS) samples.
    
    Parameters:
    -----------
    num_simulations : int
        Number of simulations to generate
    seed : int, optional
        Random seed for reproducibility
    get_variable_definitions_func : callable, optional
        Function to get variable definitions. If None, imports from example_models_3d
        
    Returns:
    --------
    dict : Dictionary containing LHS samples for all variables
    """
    if get_variable_definitions_func is None:
        from example_models_3d import get_variable_definitions
        get_variable_definitions_func = get_variable_definitions
    
    variables = get_variable_definitions_func()
    n_dim = len(variables)
    
    # Initialize LHS sampler
    sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
    uniform_samples = sampler.random(n=num_simulations)
    
    # Convert uniform samples to distributions
    results = convert_uniform_samples_to_distributions(uniform_samples, variables)
    
    return results


def generate_sobol_samples(num_simulations, seed=None, get_variable_definitions_func: Optional[Callable] = None):
    """
    Generate Sobol sequence (Quasi-Monte Carlo) samples.
    
    Parameters:
    -----------
    num_simulations : int
        Number of simulations to generate (must be power of 2 for Sobol)
    seed : int, optional
        Random seed for reproducibility (Sobol uses scramble parameter)
    get_variable_definitions_func : callable, optional
        Function to get variable definitions. If None, imports from example_models_3d
        
    Returns:
    --------
    dict : Dictionary containing Sobol samples for all variables
    """
    if get_variable_definitions_func is None:
        from example_models_3d import get_variable_definitions
        get_variable_definitions_func = get_variable_definitions
    
    variables = get_variable_definitions_func()
    n_dim = len(variables)
    
    # Sobol requires n to be power of 2, so we round up if needed
    n_sobol = 2 ** int(np.ceil(np.log2(num_simulations)))
    
    # Initialize Sobol sampler with scrambling for better distribution
    sampler = qmc.Sobol(d=n_dim, scramble=True, seed=seed)
    uniform_samples = sampler.random(n=n_sobol)
    
    # Take only the requested number of samples
    uniform_samples = uniform_samples[:num_simulations]
    
    # Convert uniform samples to distributions
    results = convert_uniform_samples_to_distributions(uniform_samples, variables)
    
    return results


def generate_hammersley_samples(num_simulations, seed=None, get_variable_definitions_func: Optional[Callable] = None):
    """
    Generate Halton sequence samples (using Halton as alternative to Hammersley).
    
    Parameters:
    -----------
    num_simulations : int
        Number of simulations to generate
    seed : int, optional
        Random seed for reproducibility (Halton uses scramble parameter)
    get_variable_definitions_func : callable, optional
        Function to get variable definitions. If None, imports from example_models_3d
        
    Returns:
    --------
    dict : Dictionary containing Halton samples for all variables
    """
    if get_variable_definitions_func is None:
        from example_models_3d import get_variable_definitions
        get_variable_definitions_func = get_variable_definitions
    
    variables = get_variable_definitions_func()
    n_dim = len(variables)
    
    # Check if Halton is available, otherwise use a simple implementation
    try:
        # Try to use Halton from scipy
        sampler = qmc.Halton(d=n_dim, scramble=True, seed=seed)
        uniform_samples = sampler.random(n=num_simulations)
    except AttributeError:
        # If Halton is not available, use LHS as fallback
        print(f"Warning: Halton not available in scipy version. Using LHS instead.")
        sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
        uniform_samples = sampler.random(n=num_simulations)
    
    # Convert uniform samples to distributions
    results = convert_uniform_samples_to_distributions(uniform_samples, variables)
    
    return results


def generate_optimized_lhs_samples(num_simulations, seed=None, n_optimize=10, get_variable_definitions_func: Optional[Callable] = None):
    """
    Generate optimized Latin Hypercube Sampling (LHS) with optimization.
    Uses multiple random starts and selects the one with lowest discrepancy.
    
    Parameters:
    -----------
    num_simulations : int
        Number of simulations to generate
    seed : int, optional
        Random seed for reproducibility
    n_optimize : int, optional
        Number of random LHS designs to try. Default is 10.
    get_variable_definitions_func : callable, optional
        Function to get variable definitions. If None, imports from example_models_3d
        
    Returns:
    --------
    dict : Dictionary containing optimized LHS samples for all variables
    """
    if get_variable_definitions_func is None:
        from example_models_3d import get_variable_definitions
        get_variable_definitions_func = get_variable_definitions
    
    variables = get_variable_definitions_func()
    n_dim = len(variables)
    
    # Generate multiple LHS designs and select the best one
    best_samples = None
    best_discrepancy = float('inf')
    
    # Use seed for reproducibility
    if seed is not None:
        rng = np.random.default_rng(seed)
        seeds_list = rng.integers(0, 2**31, size=n_optimize)
    else:
        seeds_list = [None] * n_optimize
    
    for opt_seed in seeds_list:
        sampler = qmc.LatinHypercube(d=n_dim, seed=opt_seed)
        uniform_samples = sampler.random(n=num_simulations)
        
        # Calculate discrepancy (lower is better for space-filling)
        disc = qmc.discrepancy(uniform_samples)
        
        if disc < best_discrepancy:
            best_discrepancy = disc
            best_samples = uniform_samples.copy()
    
    # Convert uniform samples to distributions
    results = convert_uniform_samples_to_distributions(best_samples, variables)
    
    return results


class SamplingComparisonRunner:
    """
    Class for comparing different sampling methods in Monte Carlo simulations.
    
    Supports comparison of: Random, LHS, Sobol, Halton, and Optimized LHS sampling methods.
    """
    
    def __init__(self, get_variable_definitions_func: Optional[Callable] = None):
        """
        Initialize the sampling comparison runner.
        
        Parameters:
        -----------
        get_variable_definitions_func : callable, optional
            Function to get variable definitions. If None, imports from example_models_3d
        """
        if get_variable_definitions_func is None:
            from example_models_3d import get_variable_definitions
            self.get_variable_definitions = get_variable_definitions
        else:
            self.get_variable_definitions = get_variable_definitions_func
        
        # Define available sampling methods
        self.method_map = {
            'random': ('Random Sampling', generate_random_samples),
            'lhs': ('Latin Hypercube Sampling', generate_lhs_samples),
            'sobol': ('Sobol Sequence', generate_sobol_samples),
            'halton': ('Halton Sequence', generate_hammersley_samples),
            'optimized_lhs': ('Optimized LHS', generate_optimized_lhs_samples)
        }
    
    def compare_sampling_methods(self, num_simulations=10, seed=42, 
                                 show_plots=True, show_detailed_plots=True,
                                 methods=None):
        """
        Compare multiple sampling methods: Random, LHS, Sobol, Halton, and Optimized LHS.
        
        Parameters:
        -----------
        num_simulations : int
            Number of simulations to run for each method
        seed : int
            Seed for sampling reproducibility
        show_plots : bool, optional
            If True, show basic displacement plots. Default is True.
        show_detailed_plots : bool, optional
            If True, show detailed statistical analysis plots. Default is True.
        methods : list, optional
            List of method names to compare. If None, compares all methods.
            Available: 'random', 'lhs', 'sobol', 'halton', 'optimized_lhs'
            
        Returns:
        --------
        dict : Dictionary containing samples from all methods
        """
        print(f"Comparing sampling methods with {num_simulations} simulations each...")
        print("="*80)
        
        # Use all methods if not specified
        if methods is None:
            methods_to_compare = list(self.method_map.keys())
        else:
            # Validate methods
            methods_to_compare = []
            for method in methods:
                method_lower = method.lower()
                if method_lower in self.method_map:
                    methods_to_compare.append(method_lower)
                else:
                    print(f"Warning: Unknown sampling method '{method}'. Skipping.")
        
        all_samples = {}
        
        # Generate samples for each method
        for idx, method_key in enumerate(methods_to_compare, 1):
            method_name, method_func = self.method_map[method_key]
            print(f"\n{idx}. Generating {method_name} samples...")
            
            # Generate samples using the appropriate function
            samples = method_func(num_simulations, seed=seed, 
                                 get_variable_definitions_func=self.get_variable_definitions)
            all_samples[method_name] = samples
        
        print("\nAll sampling methods completed!")
        
        # Print statistics table using post-processing utilities
        PostProcessingUtilities.print_statistics_table(all_samples, self.get_variable_definitions)
        
        # Create comparative plots using plotting utilities
        if show_plots and show_detailed_plots:
            # Only show histogram plots with PDF overlays
            PlottingUtilities.create_histogram_comparison_multiple(all_samples, self.get_variable_definitions)
            # Add pair plot showing relationships between all variables
            PlottingUtilities.create_pair_plot(all_samples, self.get_variable_definitions)
        
        return all_samples
    
    def get_available_methods(self):
        """
        Get list of available sampling methods.
        
        Returns:
        --------
        dict : Dictionary mapping method keys to their display names
        """
        return {key: name for key, (name, _) in self.method_map.items()}
    
    def generate_samples(self, method: str, num_simulations: int, seed: int = None):
        """
        Generate samples using a specific method.
        
        Parameters:
        -----------
        method : str
            Sampling method: 'random', 'lhs', 'sobol', 'halton', 'optimized_lhs'
        num_simulations : int
            Number of simulations to generate
        seed : int, optional
            Seed for sampling reproducibility
            
        Returns:
        --------
        dict : Dictionary containing samples for the specified method
        """
        method_lower = method.lower()
        if method_lower not in self.method_map:
            raise ValueError(f"Unknown sampling method: {method}. "
                           f"Choose from: {list(self.method_map.keys())}")
        
        method_name, method_func = self.method_map[method_lower]
        print(f"Using {method_name} with seed {seed}")
        
        # Generate samples
        samples = method_func(num_simulations, seed=seed, 
                             get_variable_definitions_func=self.get_variable_definitions)
        
        return samples

