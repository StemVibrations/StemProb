"""
Sensitivity Analysis module for geotechnical models using SALib.

This module provides classes for performing sensitivity analysis on the 3D model
using various methods from the SALib library, including Morris screening.
"""

import numpy as np
from SALib.sample import morris as morris_sampling
from SALib.analyze import morris as morris_analyze
import json
import os
from typing import Dict, List, Tuple, Optional

# Import plotting and post-processing utilities
from plotting import MorrisPlotting
from postprocessing import PostProcessingUtilities


class MorrisSensitivityAnalysis:
    """
    Class for performing Morris sensitivity analysis on the 3D geotechnical model.
    
    The Morris method is an efficient screening method that provides qualitative
    information about the relative importance of input parameters.
    """
    
    def __init__(self, num_trajectories: int = 10, num_levels: int = 4, seed: int = 42):
        """
        Initialize the Morris sensitivity analysis.
        
        Parameters:
        -----------
        num_trajectories : int
            Number of Morris trajectories (default: 10)
        num_levels : int
            Number of levels for Morris sampling (default: 4)
        seed : int
            Random seed for reproducibility (default: 42)
        """
        self.num_trajectories = num_trajectories
        self.num_levels = num_levels
        self.seed = seed
        
        # Define the problem structure for SALib
        self.problem = {
            'num_vars': 9,
            'names': [
                'clay_density',
                'clay_young_modulus', 
                'sand_density',
                'sand_young_modulus',
                'embankment_density',
                'embankment_young_modulus',
                'vertical_load',
                'rayleigh_k',
                'rayleigh_m'
            ],
            'bounds': [
                [1000, 3000],      # clay_density (kg/m³)
                [20e6, 100e6],      # clay_young_modulus (Pa)
                [1000, 3000],      # sand_density (kg/m³)
                [100e6, 400e6],    # sand_young_modulus (Pa)
                [1000, 3000],      # embankment_density (kg/m³)
                [50e6, 150e6],     # embankment_young_modulus (Pa)
                [-40000, -20000],     # vertical_load (N)      ###set as a constant value   needs to be checked with WIM --> 80 kn
                [1e-6, 1e-3],      # rayleigh_k
                [0.1, 0.9]         # rayleigh_m
            ],
            'groups': None  # No groups for this analysis
        }
        
        self.samples = None
        self.results = None
        self.sensitivity_indices = None
        
    def generate_samples(self) -> np.ndarray:
        """
        Generate Morris samples using SALib.
        
        Returns:
        --------
        np.ndarray : Array of Morris samples
        """
        print(f"Generating Morris samples with {self.num_trajectories} trajectories...")
        
        # Generate Morris samples
        self.samples = morris_sampling.sample(
            self.problem, 
            N=self.num_trajectories,
            num_levels=self.num_levels,
            seed=self.seed
        )
        
        print(f"Generated {len(self.samples)} Morris samples")
        return self.samples
    
    def analyze_sensitivity(self, model_outputs: np.ndarray) -> Dict:
        """
        Analyze sensitivity using Morris method.
        
        Parameters:
        -----------
        model_outputs : np.ndarray
            Array of model outputs (displacement values) corresponding to the samples
            
        Returns:
        --------
        Dict : Dictionary containing sensitivity indices
        """
        print("Analyzing sensitivity using Morris method...")
        
        if self.samples is None:
            raise ValueError("Samples must be generated first using generate_samples()")
            
        if len(model_outputs) != len(self.samples):
            raise ValueError(f"Number of outputs ({len(model_outputs)}) must match number of samples ({len(self.samples)})")
        
        # Perform Morris analysis
        self.sensitivity_indices = morris_analyze.analyze(
            self.problem,
            self.samples,
            model_outputs,
            num_levels=self.num_levels,
            num_resamples=50,  # For confidence intervals
            conf_level=0.95,
            print_to_console=False
        )

        for key, value in list(self.sensitivity_indices.items()):
            try:
                if np.ma.isMaskedArray(value):
                    value = value.filled(np.nan)
                array_value = np.asarray(value, dtype=float)
                array_value = np.nan_to_num(array_value, nan=0.0)
                self.sensitivity_indices[key] = array_value
            except (TypeError, ValueError):
                # Leave non-numeric entries (like 'names') untouched
                continue
        
        return self.sensitivity_indices
    
    def get_sensitivity_summary(self) -> Dict:
        """
        Get a summary of sensitivity results.
        
        Returns:
        --------
        Dict : Summary of sensitivity analysis results
        """
        if self.sensitivity_indices is None:
            raise ValueError("Sensitivity analysis must be performed first")
        
        summary = {
            'variable_names': list(self.problem['names']),
            'mu_star': self.sensitivity_indices['mu_star'].tolist(),
            'mu': self.sensitivity_indices['mu'].tolist(),
            'sigma': self.sensitivity_indices['sigma'].tolist(),
            'mu_star_conf': self.sensitivity_indices['mu_star_conf'].tolist(),
            'ranking': self._rank_variables(),
            'interpretation': self._interpret_results()
        }
        
        return summary
    
    def _rank_variables(self) -> List[Tuple[str, float]]:
        """
        Rank variables by their mu_star values.
        
        Returns:
        --------
        List[Tuple[str, float]] : List of (variable_name, mu_star) tuples ranked by importance
        """
        mu_star_values = self.sensitivity_indices['mu_star']
        if np.ma.isMaskedArray(mu_star_values):
            mu_star_values = mu_star_values.filled(np.nan)
        else:
            mu_star_values = np.asarray(mu_star_values, dtype=float)
        mu_star_values = np.nan_to_num(mu_star_values, nan=0.0)
        variable_names = self.problem['names']
        
        # Create list of tuples and sort by mu_star (descending)
        ranking = [(variable_names[i], float(mu_star_values[i])) for i in range(len(variable_names))]
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def _interpret_results(self) -> Dict[str, str]:
        """
        Interpret the Morris sensitivity results.
        
        Returns:
        --------
        Dict[str, str] : Interpretation of results for each variable
        """
        mu_star = self.sensitivity_indices['mu_star']
        mu = self.sensitivity_indices['mu']
        sigma = self.sensitivity_indices['sigma']

        def _prepare(arr):
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)
            else:
                arr = np.asarray(arr, dtype=float)
            return np.nan_to_num(arr, nan=0.0)

        mu_star = _prepare(mu_star)
        mu = _prepare(mu)
        sigma = _prepare(sigma)
        
        interpretations = {}
        
        for i, var_name in enumerate(self.problem['names']):
            mu_star_val = mu_star[i]
            mu_val = mu[i]
            sigma_val = sigma[i]
            
            # Determine sensitivity level
            if mu_star_val > 0.5:
                sensitivity_level = "High"
            elif mu_star_val > 0.2:
                sensitivity_level = "Medium"
            else:
                sensitivity_level = "Low"
            
            # Determine linearity
            if sigma_val > mu_star_val:
                linearity = "Non-linear effects present"
            else:
                linearity = "Linear effects dominate"
            
            # Determine interaction
            if sigma_val > 0.5:
                interaction = "High interaction with other variables"
            elif sigma_val > 0.2:
                interaction = "Moderate interaction with other variables"
            else:
                interaction = "Low interaction with other variables"
            
            interpretations[var_name] = {
                'sensitivity_level': sensitivity_level,
                'linearity': linearity,
                'interaction': interaction,
                'mu_star': float(mu_star_val),
                'mu': float(mu_val),
                'sigma': float(sigma_val)
            }
        
        return interpretations
    
    def plot_sensitivity_analysis(self, save_plots: bool = True, output_dir: str = "sensitivity_plots"):
        """
        Create comprehensive plots for Morris sensitivity analysis using MorrisPlotting class.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to files
        output_dir : str
            Directory to save plots
        """
        if self.sensitivity_indices is None:
            raise ValueError("Sensitivity analysis must be performed first")
        
        # Use MorrisPlotting class for all plotting
        morris_plotter = MorrisPlotting(self.sensitivity_indices, self.problem)
        ranking = self._rank_variables()
        
        # Create all plots
        try:
            morris_plotter.plot_mu_star_sigma(save_plots, output_dir)
        except Exception as e:
            print(f"Warning: Could not create mu_star_sigma plot: {e}")

        try:
            morris_plotter.plot_mu_sigma(save_plots, output_dir)
        except Exception as e:
            print(f"Warning: Could not create mu_sigma plot: {e}")
        
        try:
            morris_plotter.plot_sensitivity_ranking(save_plots, output_dir, ranking)
        except Exception as e:
            print(f"Warning: Could not create sensitivity ranking plot: {e}")
        
        try:
            morris_plotter.plot_variance_decomposition(save_plots, output_dir)
        except Exception as e:
            print(f"Warning: Could not create variance decomposition plot: {e}")
    
    # Note: Plotting methods have been moved to MorrisPlotting class in plotting.py
    # Use plot_sensitivity_analysis() which delegates to MorrisPlotting
    
    def save_results(self, filename: str = "morris_sensitivity_results.json"):
        """
        Save sensitivity analysis results to JSON file.
        
        Parameters:
        -----------
        filename : str
            Name of the output file
        """
        if self.sensitivity_indices is None:
            raise ValueError("Sensitivity analysis must be performed first")
        
        # Prepare results for JSON serialization
        results_to_save = {
            'method': 'Morris',
            'parameters': {
                'num_trajectories': self.num_trajectories,
                'num_levels': self.num_levels,
                'seed': self.seed
            },
            'problem_definition': self.problem,
            'sensitivity_indices': {
                'mu_star': self.sensitivity_indices['mu_star'],
                'mu': self.sensitivity_indices['mu'],
                'sigma': self.sensitivity_indices['sigma'],
                'mu_star_conf': self.sensitivity_indices['mu_star_conf']
            },
            'summary': self.get_sensitivity_summary()
        }
 
        # Convert numpy arrays to lists for JSON serialization
        for key, value in results_to_save['sensitivity_indices'].items():
            array_value = np.asarray(value, dtype=float)
            array_value = np.nan_to_num(array_value, nan=0.0)
            results_to_save['sensitivity_indices'][key] = array_value.tolist()

        # Ensure ranking and interpretations are JSON serializable
        ranking_serializable = []
        for name, score in results_to_save['summary']['ranking']:
            ranking_serializable.append([name, float(score)])
        results_to_save['summary']['ranking'] = ranking_serializable

        interpretation_serializable = {}
        for name, info in results_to_save['summary']['interpretation'].items():
            interpretation_serializable[name] = {
                'sensitivity_level': info['sensitivity_level'],
                'linearity': info['linearity'],
                'interaction': info['interaction'],
                'mu_star': float(info['mu_star']),
                'mu': float(info['mu']),
                'sigma': float(info['sigma'])
            }
        results_to_save['summary']['interpretation'] = interpretation_serializable
 
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Sensitivity analysis results saved to: {filename}")


class SensitivityAnalysisRunner:
    """
    Main class to run sensitivity analysis on the 3D geotechnical model.
    """
    
    def __init__(self, model_runner_func, output_variable: str = 'DISPLACEMENT_Y'):
        """
        Initialize the sensitivity analysis runner.
        
        Parameters:
        -----------
        model_runner_func : callable
            Function that runs the model and returns outputs
        output_variable : str
            Name of the output variable to analyze (default: 'DISPLACEMENT_Y')
        """
        self.model_runner_func = model_runner_func
        self.output_variable = output_variable
        self.morris_analysis = None
        self.last_time_series = None
        self.last_response_series_y = None
        self.last_response_series_x = None
        
    def run_morris_analysis(self, num_trajectories: int = 10, num_levels: int = 4, seed: int = 42):
        """
        Run Morris sensitivity analysis.
        
        Parameters:
        -----------
        num_trajectories : int
            Number of Morris trajectories
        num_levels : int
            Number of levels for Morris sampling
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        Dict : Sensitivity analysis results
        """
        print("Starting Morris Sensitivity Analysis...")
        
        # Initialize Morris analysis
        self.morris_analysis = MorrisSensitivityAnalysis(
            num_trajectories=num_trajectories,
            num_levels=num_levels,
            seed=seed
        )
        
        # Generate samples
        samples = self.morris_analysis.generate_samples()
        
        # Run model for each sample
        print(f"Running model for {len(samples)} Morris samples...")
        model_outputs = []
        time_series_list = []
        response_series_list_y = []
        response_series_list_x = []
        
        for i, sample in enumerate(samples):
            print("\n" + "=" * 80)
            print(f"MORRIS SAMPLE {i+1} / {len(samples)}".center(80))
            print("=" * 80 + "\n")
            
            # Run model with current sample
            output = self.model_runner_func(sample)
            
            if isinstance(output, dict) and 'summary_value' in output:
                model_outputs.append(output['summary_value'])

                time_values = output.get('time')
                response_y = output.get('response_y')
                response_x = output.get('response_x')

                if time_values is not None and response_y is not None:
                    time_series_list.append(np.asarray(time_values, dtype=float))
                    response_series_list_y.append(np.asarray(response_y, dtype=float))
                    if response_x is not None:
                        response_series_list_x.append(np.asarray(response_x, dtype=float))
            elif isinstance(output, dict) and self.output_variable in output:
                model_outputs.append(output[self.output_variable])
            else:
                model_outputs.append(output)
        
        model_outputs = np.array(model_outputs)

        time_array = None
        response_array_y = None
        response_array_x = None

        if response_series_list_y:
            try:
                response_array_y = np.stack(response_series_list_y)
                time_array = np.stack(time_series_list)
                if response_series_list_x and len(response_series_list_x) == len(response_series_list_y):
                    response_array_x = np.stack(response_series_list_x)
            except ValueError:
                print("Warning: Inconsistent time series lengths; skipping response summary plot.")

        self.last_time_series = time_array
        self.last_response_series_y = response_array_y
        self.last_response_series_x = response_array_x
        
        # Analyze sensitivity
        sensitivity_indices = self.morris_analysis.analyze_sensitivity(model_outputs)

        # Generate plots and save results
        # Ensure output directory exists before plotting
        output_dir = "sensitivity_plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        self.morris_analysis.plot_sensitivity_analysis(save_plots=True, output_dir=output_dir)
        self.morris_analysis.save_results()

        if self.last_time_series is not None and self.last_response_series_y is not None:
            try:
                # Use MorrisPlotting class for response summary plot
                MorrisPlotting.plot_morris_response_summary(
                    self.last_time_series,
                    self.last_response_series_y,
                    response_x_series=self.last_response_series_x,
                    title=f"Morris Response Summary ({len(model_outputs)} samples)",
                    save_plots=True,
                    output_dir="sensitivity_plots"
                )
            except Exception as e:
                print(f"Warning: Could not create response summary plot: {e}")
        
        return sensitivity_indices
    
    def get_results_summary(self):
        """Get summary of sensitivity analysis results."""
        if self.morris_analysis is None:
            raise ValueError("Morris analysis must be run first")
        
        return self.morris_analysis.get_sensitivity_summary()

# Note: plot_morris_response_summary() has been moved to MorrisPlotting class in plotting.py
# Use MorrisPlotting.plot_morris_response_summary() instead