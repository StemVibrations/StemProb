"""
Sensitivity Analysis module for geotechnical models using SALib.

This module provides classes for performing sensitivity analysis on the 3D model
using various methods, including Morris screening and SRC (Standardised Regression Coefficients).
"""

import numpy as np
from SALib.sample import morris as morris_sampling
from SALib.analyze import morris as morris_analyze
from scipy.stats import qmc
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


class SRCSensitivityAnalysis:
    """
    SRC (Standardised Regression Coefficients) sensitivity analysis.

    Fits a single multiple OLS regression on N LHS samples after standardising
    X and Y (Saltelli et al. 2008, Ch.3). SRC_i = β̂_i — the partial slope for
    parameter i, controlling for all others. Includes R² linearity check.
    """

    def __init__(self, n_samples: int = 100, seed: int = 42):
        """
        Initialize the RBD sensitivity analysis.

        Parameters:
        -----------
        n_samples : int
            Number of samples for RBD analysis (default: 100)
        seed : int
            Random seed for reproducibility (default: 42)
        """
        self.n_samples = n_samples
        self.seed = seed

        # Define the problem structure for consistency with Morris
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
                [-40000, -20000],     # vertical_load (N)
                [1e-6, 1e-3],      # rayleigh_k
                [0.1, 0.9]         # rayleigh_m
            ],
            'groups': None
        }

        self.samples = None
        self.results = None
        self.sensitivity_indices = None

    def generate_samples(self) -> np.ndarray:
        """
        Generate balanced Latin hypercube samples for RBD.

        Returns:
        --------
        np.ndarray : Array of RBD samples normalized to [0, 1]
        """
        print(f"Generating {self.n_samples} LHS samples for SRC analysis...")

        # Generate balanced Latin hypercube samples
        sampler = qmc.LatinHypercube(d=self.problem['num_vars'], seed=self.seed, optimization='random-cd')
        samples_normalized = sampler.random(n=self.n_samples)

        self.samples = samples_normalized

        print(f"Generated {len(self.samples)} SRC (LHS) samples")
        return self.samples

    def analyze_sensitivity(self, model_outputs: np.ndarray) -> Dict:
        """
        Analyze sensitivity using regression-based RBD method.

        Parameters:
        -----------
        model_outputs : np.ndarray
            Array of model outputs corresponding to the samples

        Returns:
        --------
        Dict : Dictionary containing sensitivity indices
        """
        print("Analyzing sensitivity using SRC method (multiple linear regression)...")

        if self.samples is None:
            raise ValueError("Samples must be generated first using generate_samples()")

        if len(model_outputs) != len(self.samples):
            raise ValueError(f"Number of outputs ({len(model_outputs)}) must match number of samples ({len(self.samples)})")

        model_outputs = np.asarray(model_outputs, dtype=float)

        # ── Standardise X and Y ───────────────────────────────────────────────
        # Scale each input column to zero mean, unit variance
        X_mean = self.samples.mean(axis=0)
        X_sig  = self.samples.std(axis=0, ddof=1)
        X_sig[X_sig < 1e-12] = 1.0                        # guard against constant cols
        X_std  = (self.samples - X_mean) / X_sig

        Y_mean = model_outputs.mean()
        Y_sig  = model_outputs.std(ddof=1)
        if Y_sig < 1e-12:
            Y_sig = 1.0
        Y_std  = (model_outputs - Y_mean) / Y_sig

        # ── Single multiple OLS regression on standardised variables ─────────
        # Y_std ≈ β₀ + β₁X₁_std + … + β₉X₉_std
        # Because both X and Y are standardised, the slopes β̂ᵢ ARE the SRCs directly
        # (Saltelli 2008, Ch.3: SRCᵢ = β̂ᵢ × σXᵢ/σY — already baked in by standardisation)
        A    = np.column_stack([np.ones(len(Y_std)), X_std])
        beta, _, _, _ = np.linalg.lstsq(A, Y_std, rcond=None)
        src  = beta[1:]                                    # drop intercept

        # ── R² — model linearity check ────────────────────────────────────────
        Y_pred  = A @ beta
        ss_res  = float(np.sum((Y_std - Y_pred) ** 2))
        ss_tot  = float(np.sum((Y_std - Y_std.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r_squared < 0.7:
            print(f"  WARNING: R² = {r_squared:.3f} < 0.7 — model is nonlinear; "
                  "SRC values may not be reliable.")
        else:
            print(f"  R² = {r_squared:.3f} — linear approximation is acceptable.")

        # ── Store results ─────────────────────────────────────────────────────
        self.sensitivity_indices = {
            'src':        src,                  # signed SRC (β̂ᵢ on standardised vars)
            'mu_star':    np.abs(src),          # |SRC| — for ranking / plotting compat.
            'R2':         r_squared,
        }

        return self.sensitivity_indices

    def get_sensitivity_summary(self) -> Dict:
        """
        Get a summary of SRC sensitivity results.

        Returns:
        --------
        Dict : Summary of sensitivity analysis results
        """
        if self.sensitivity_indices is None:
            raise ValueError("Sensitivity analysis must be performed first")

        summary = {
            'variable_names': list(self.problem['names']),
            'src':     [float(v) for v in self.sensitivity_indices['src']],
            'mu_star': [float(v) for v in self.sensitivity_indices['mu_star']],
            'R2':      float(self.sensitivity_indices['R2']),
            'ranking': self._rank_variables(),
            'interpretation': self._interpret_results()
        }

        return summary

    def _rank_variables(self) -> List[Tuple[str, float]]:
        """
        Rank variables by their absolute correlation values.

        Returns:
        --------
        List[Tuple[str, float]] : List of (variable_name, abs_correlation) tuples ranked by importance
        """
        mu_star_values = self.sensitivity_indices['mu_star']
        variable_names = self.problem['names']

        # Create list of tuples and sort by mu_star (descending)
        ranking = [(variable_names[i], float(mu_star_values[i])) for i in range(len(variable_names))]
        ranking.sort(key=lambda x: x[1], reverse=True)

        return ranking

    def _interpret_results(self) -> Dict[str, str]:
        """
        Interpret the SRC sensitivity results.

        Returns:
        --------
        Dict[str, str] : Interpretation of results for each variable
        """
        src_vals = self.sensitivity_indices['src']
        r2       = float(self.sensitivity_indices['R2'])

        interpretations = {}

        for i, var_name in enumerate(self.problem['names']):
            src_i   = float(src_vals[i])
            abs_src = abs(src_i)

            if abs_src > 0.5:
                sensitivity_level = "High"
            elif abs_src > 0.2:
                sensitivity_level = "Medium"
            else:
                sensitivity_level = "Low"

            if src_i > 0:
                direction = "Positive effect on output"
            elif src_i < 0:
                direction = "Negative effect on output"
            else:
                direction = "No effect on output"

            interpretations[var_name] = {
                'sensitivity_level': sensitivity_level,
                'direction': direction,
                'src': src_i,
                'abs_src': abs_src,
                'R2': r2,
            }

        return interpretations

    def plot_sensitivity_analysis(self, save_plots: bool = True, output_dir: str = "sensitivity_plots"):
        """
        Create plots for RBD sensitivity analysis using MorrisPlotting class (compatible).

        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to files
        output_dir : str
            Directory to save plots
        """
        if self.sensitivity_indices is None:
            raise ValueError("Sensitivity analysis must be performed first")

        # Use MorrisPlotting class for compatibility (mu_star is set to abs(correlation))
        morris_plotter = MorrisPlotting(self.sensitivity_indices, self.problem)
        ranking = self._rank_variables()

        # Create compatible plots
        try:
            morris_plotter.plot_sensitivity_ranking(save_plots, output_dir, ranking)
        except Exception as e:
            print(f"Warning: Could not create sensitivity ranking plot: {e}")

        try:
            morris_plotter.plot_variance_decomposition(save_plots, output_dir)
        except Exception as e:
            print(f"Warning: Could not create variance decomposition plot: {e}")

    def save_results(self, filename: str = "src_sensitivity_results.json"):
        """
        Save RBD sensitivity analysis results to JSON file.

        Parameters:
        -----------
        filename : str
            Name of the output file
        """
        if self.sensitivity_indices is None:
            raise ValueError("Sensitivity analysis must be performed first")

        results_to_save = {
            'method': 'SRC',
            'parameters': {
                'n_samples': self.n_samples,
                'seed': self.seed
            },
            'problem_definition': self.problem,
            'sensitivity_indices': {
                'src':     [float(v) for v in self.sensitivity_indices['src']],
                'mu_star': [float(v) for v in self.sensitivity_indices['mu_star']],
                'R2':      float(self.sensitivity_indices['R2']),
            },
            'summary': self.get_sensitivity_summary()
        }

        ranking_serializable = []
        for name, score in results_to_save['summary']['ranking']:
            ranking_serializable.append([name, float(score)])
        results_to_save['summary']['ranking'] = ranking_serializable

        interpretation_serializable = {}
        for name, info in results_to_save['summary']['interpretation'].items():
            interpretation_serializable[name] = {
                'sensitivity_level': info['sensitivity_level'],
                'direction': info['direction'],
                'src': float(info['src']),
                'abs_src': float(info['abs_src']),
                'R2': float(info['R2']),
            }
        results_to_save['summary']['interpretation'] = interpretation_serializable

        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"SRC sensitivity analysis results saved to: {filename}")


class SensitivityAnalysisRunner:
    """
    Main class to run sensitivity analysis on the 3D geotechnical model.
    """
    
    def __init__(self, model_runner_func, output_variable: str = 'DISPLACEMENT_Y',
                 output_mode: str = 'velocity'):
        """
        Initialize the sensitivity analysis runner.

        Parameters:
        -----------
        model_runner_func : callable
            Function that runs the model and returns outputs
        output_variable : str
            Name of the output variable to analyze (default: 'DISPLACEMENT_Y')
        output_mode : str
            Which response metric to use as the sensitivity target:
              'velocity'     -> V_eff_max  (default)
              'psd'          -> PSD_max
              'displacement' -> max |DISPLACEMENT_Y|
        """
        self.model_runner_func = model_runner_func
        self.output_variable = output_variable
        self.output_mode = output_mode
        self.morris_analysis = None
        self.src_analysis = None
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
            output = self.model_runner_func(sample, output_mode=self.output_mode)

            if isinstance(output, dict):
                key = self._MODE_KEY.get(self.output_mode)
                val = output.get(key, np.nan) if key else np.nan
                model_outputs.append(float(val) if val is not None else np.nan)

                time_values = output.get('time')
                response_y = output.get('velocity_y')
                if time_values is not None and response_y is not None:
                    time_series_list.append(np.asarray(time_values, dtype=float))
                    response_series_list_y.append(np.asarray(response_y, dtype=float))
            else:
                model_outputs.append(np.nan)

        model_outputs = np.array(model_outputs, dtype=float)

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
        output_dir = "sensitivity_plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.morris_analysis.plot_sensitivity_analysis(save_plots=True, output_dir=output_dir)
        self.morris_analysis.save_results(f"morris_sensitivity_results_{self.output_mode}.json")

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

    def run_src_analysis(self, n_samples: int = 100, seed: int = 42):
        """
        Run RBD (Random Balance Design) sensitivity analysis.

        Parameters:
        -----------
        n_samples : int
            Number of RBD samples
        seed : int
            Random seed for reproducibility

        Returns:
        --------
        Dict : Sensitivity analysis results
        """
        print("Starting SRC Sensitivity Analysis...")

        # Initialize SRC analysis
        self.src_analysis = SRCSensitivityAnalysis(
            n_samples=n_samples,
            seed=seed
        )

        # Generate samples
        samples = self.src_analysis.generate_samples()

        # Run model for each sample
        print(f"Running model for {len(samples)} SRC samples...")
        model_outputs = []
        time_series_list = []
        response_series_list_y = []
        response_series_list_x = []

        # Scale [0,1] LHS samples to actual parameter bounds
        bounds = np.array(self.src_analysis.problem['bounds'])
        lo, hi = bounds[:, 0], bounds[:, 1]

        for i, sample in enumerate(samples):
            print("\n" + "=" * 80)
            print(f"RBD SAMPLE {i+1} / {len(samples)}".center(80))
            print("=" * 80 + "\n")

            scaled_sample = lo + sample * (hi - lo)

            # Run model with current sample
            output = self.model_runner_func(scaled_sample, output_mode=self.output_mode)

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
        sensitivity_indices = self.src_analysis.analyze_sensitivity(model_outputs)

        # Generate plots and save results
        output_dir = "sensitivity_plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.src_analysis.plot_sensitivity_analysis(save_plots=True, output_dir=output_dir)
        self.src_analysis.save_results(f"src_sensitivity_results_{self.output_mode}.json")

        if self.last_time_series is not None and self.last_response_series_y is not None:
            try:
                MorrisPlotting.plot_morris_response_summary(
                    self.last_time_series,
                    self.last_response_series_y,
                    response_x_series=self.last_response_series_x,
                    title=f"RBD Response Summary ({len(model_outputs)} samples)",
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

    # ── All-modes runners (single model loop, multiple SA outputs) ────────────

    # Maps output_mode name -> key in the model output dict
    _MODE_KEY = {
        'velocity':     'V_eff_max',
        'v_y_max':      'V_y_max',
        'psd':          'PSD_max',
        'displacement': 'disp_y_max',
    }

    def run_morris_all_modes(self, modes=('velocity', 'psd', 'displacement'),
                             num_trajectories: int = 10, num_levels: int = 4, seed: int = 42):
        """
        Run Morris sensitivity analysis once, producing results for every requested
        output mode without re-running the model.

        Parameters
        ----------
        modes : iterable of str
            Any subset of ('velocity', 'v_y_max', 'psd', 'displacement').
        num_trajectories, num_levels, seed : Morris sampling parameters.

        Returns
        -------
        dict  {mode: sensitivity_indices}
        """
        print("Starting Morris Sensitivity Analysis (all modes in one pass)...")

        self.morris_analysis = MorrisSensitivityAnalysis(
            num_trajectories=num_trajectories,
            num_levels=num_levels,
            seed=seed
        )
        samples = self.morris_analysis.generate_samples()

        # Accumulators: one list per mode + time-series lists
        outputs_per_mode = {m: [] for m in modes}
        time_series_list, response_y_list, response_x_list = [], [], []
        fail_count = 0

        for i, sample in enumerate(samples):
            print("\n" + "=" * 80)
            print(f"MORRIS SAMPLE {i+1} / {len(samples)}".center(80))
            print("=" * 80 + "\n")

            # Single model call — all metrics are returned regardless of output_mode
            output = self.model_runner_func(sample, output_mode='velocity')

            if isinstance(output, dict):
                # Count failures: a run is failed if its core outputs are all NaN
                if np.isnan(output.get('disp_y_max', np.nan)) and np.isnan(output.get('V_eff_max', np.nan)):
                    fail_count += 1

                for m in modes:
                    key = self._MODE_KEY.get(m)
                    val = output.get(key, np.nan) if key else np.nan
                    outputs_per_mode[m].append(float(val) if val is not None else np.nan)

                t = output.get('time')
                ry = output.get('response_y')
                rx = output.get('response_x')
                if t is not None and ry is not None:
                    time_series_list.append(np.asarray(t, dtype=float))
                    response_y_list.append(np.asarray(ry, dtype=float))
                    if rx is not None:
                        response_x_list.append(np.asarray(rx, dtype=float))
            else:
                for m in modes:
                    outputs_per_mode[m].append(np.nan)

        print(f"\nMorris model runs: {len(samples) - fail_count}/{len(samples)} succeeded, "
              f"{fail_count} failed (returned NaN).")
        if fail_count == len(samples):
            print("  WARNING: ALL model runs failed. SA indices will be zero. "
                  "Check STEM output above for error messages.")

        # Store time-series for response summary plot
        try:
            if response_y_list:
                self.last_time_series = np.stack(time_series_list)
                self.last_response_series_y = np.stack(response_y_list)
                self.last_response_series_x = (
                    np.stack(response_x_list)
                    if len(response_x_list) == len(response_y_list) else None
                )
        except ValueError:
            print("Warning: Inconsistent time series lengths; skipping response plot.")
            self.last_time_series = self.last_response_series_y = self.last_response_series_x = None

        output_dir = "sensitivity_plots"
        os.makedirs(output_dir, exist_ok=True)

        all_results = {}
        for m in modes:
            print(f"\nAnalyzing Morris sensitivity for output mode: {m} ...")
            model_outputs = np.array(outputs_per_mode[m])
            self.morris_analysis.analyze_sensitivity(model_outputs)
            self.morris_analysis.plot_sensitivity_analysis(save_plots=True, output_dir=output_dir)
            self.morris_analysis.save_results(f"morris_sensitivity_results_{m}.json")
            all_results[m] = self.morris_analysis.sensitivity_indices
            print(f"  Saved: morris_sensitivity_results_{m}.json")

        if self.last_time_series is not None and self.last_response_series_y is not None:
            try:
                MorrisPlotting.plot_morris_response_summary(
                    self.last_time_series, self.last_response_series_y,
                    response_x_series=self.last_response_series_x,
                    title=f"Morris Response Summary ({len(samples)} samples)",
                    save_plots=True, output_dir=output_dir
                )
            except Exception as e:
                print(f"Warning: Could not create response summary plot: {e}")

        return all_results

    def run_src_all_modes(self, modes=('velocity', 'psd', 'displacement'),
                          n_samples: int = 100, seed: int = 42):
        """
        Run RBD sensitivity analysis once, producing results for every requested
        output mode without re-running the model.

        Parameters
        ----------
        modes : iterable of str
            Any subset of ('velocity', 'v_y_max', 'psd', 'displacement').
        n_samples, seed : RBD sampling parameters.

        Returns
        -------
        dict  {mode: sensitivity_indices}
        """
        print("Starting SRC Sensitivity Analysis (all modes in one pass)...")

        self.src_analysis = SRCSensitivityAnalysis(n_samples=n_samples, seed=seed)
        samples = self.src_analysis.generate_samples()

        outputs_per_mode = {m: [] for m in modes}
        time_series_list, response_y_list, response_x_list = [], [], []
        fail_count = 0

        # Scale [0,1] LHS samples to actual parameter bounds
        bounds = np.array(self.src_analysis.problem['bounds'])
        lo, hi = bounds[:, 0], bounds[:, 1]

        for i, sample in enumerate(samples):
            print("\n" + "=" * 80)
            print(f"RBD SAMPLE {i+1} / {len(samples)}".center(80))
            print("=" * 80 + "\n")

            scaled_sample = lo + sample * (hi - lo)
            output = self.model_runner_func(scaled_sample, output_mode='velocity')

            if isinstance(output, dict):
                if np.isnan(output.get('disp_y_max', np.nan)) and np.isnan(output.get('V_eff_max', np.nan)):
                    fail_count += 1

                for m in modes:
                    key = self._MODE_KEY.get(m)
                    val = output.get(key, np.nan) if key else np.nan
                    outputs_per_mode[m].append(float(val) if val is not None else np.nan)

                t = output.get('time')
                ry = output.get('response_y')
                rx = output.get('response_x')
                if t is not None and ry is not None:
                    time_series_list.append(np.asarray(t, dtype=float))
                    response_y_list.append(np.asarray(ry, dtype=float))
                    if rx is not None:
                        response_x_list.append(np.asarray(rx, dtype=float))
            else:
                for m in modes:
                    outputs_per_mode[m].append(np.nan)

        print(f"\nSRC model runs: {len(samples) - fail_count}/{len(samples)} succeeded, "
              f"{fail_count} failed (returned NaN).")
        if fail_count == len(samples):
            print("  WARNING: ALL model runs failed. SA indices will be zero. "
                  "Check STEM output above for error messages.")

        try:
            if response_y_list:
                self.last_time_series = np.stack(time_series_list)
                self.last_response_series_y = np.stack(response_y_list)
                self.last_response_series_x = (
                    np.stack(response_x_list)
                    if len(response_x_list) == len(response_y_list) else None
                )
        except ValueError:
            print("Warning: Inconsistent time series lengths; skipping response plot.")
            self.last_time_series = self.last_response_series_y = self.last_response_series_x = None

        output_dir = "sensitivity_plots"
        os.makedirs(output_dir, exist_ok=True)

        all_results = {}
        for m in modes:
            print(f"\nAnalyzing SRC sensitivity for output mode: {m} ...")
            model_outputs = np.array(outputs_per_mode[m])
            self.src_analysis.analyze_sensitivity(model_outputs)
            self.src_analysis.plot_sensitivity_analysis(save_plots=True, output_dir=output_dir)
            self.src_analysis.save_results(f"src_sensitivity_results_{m}.json")
            all_results[m] = self.src_analysis.sensitivity_indices
            print(f"  Saved: src_sensitivity_results_{m}.json")

        if self.last_time_series is not None and self.last_response_series_y is not None:
            try:
                MorrisPlotting.plot_morris_response_summary(
                    self.last_time_series, self.last_response_series_y,
                    response_x_series=self.last_response_series_x,
                    title=f"SRC Response Summary ({len(samples)} samples)",
                    save_plots=True, output_dir=output_dir
                )
            except Exception as e:
                print(f"Warning: Could not create response summary plot: {e}")

        return all_results

# Note: plot_morris_response_summary() has been moved to MorrisPlotting class in plotting.py
# Use MorrisPlotting.plot_morris_response_summary() instead