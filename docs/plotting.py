"""
Plotting utilities for geotechnical model analysis and sensitivity analysis.

This module contains classes for creating various plots including:
- Response output plots (time histories, statistics, histograms)
- Sampling method comparison plots
- Morris sensitivity analysis plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from typing import Dict, Optional, Any
import scipy.stats as stats


class PlottingUtilities:
    """
    Class for general plotting utilities for model response and sampling comparison.
    """
    
    @staticmethod
    def plot_response_output(responses_dict: dict,
                            disp_coord: str,
                            name_of_the_model: str | None = None,
                            NUM_SIMS: int | None = None):
        """
        Plot response output showing time histories, statistics, and histograms.
        
        Parameters:
        -----------
        responses_dict : dict
            Dictionary containing 'TIME' and 'DISPLACEMENT_{disp_coord}' keys
        disp_coord : str
            Displacement coordinate ('X', 'Y', or 'Z')
        name_of_the_model : str, optional
            Model name for plot title
        NUM_SIMS : int, optional
            Number of simulations
        """
        # Set up figure with subplots: time series (top), histogram (bottom-left), statistics (bottom-right)
        fig = plt.figure(layout='constrained')
        gs = GridSpec(2, 2, figure=fig)
        response_time_ax = fig.add_subplot(gs[0, :])  # Full-width time series plot
        hist_ax = fig.add_subplot(gs[1, 0])          # Histogram of mean values
        ax_statplot = fig.add_subplot(gs[1, 1])      # Mean and confidence intervals

        # Extract response data for the specified coordinate
        responses = responses_dict[f'DISPLACEMENT_{disp_coord}']
        time = responses_dict['TIME']
        responses_np = np.array(responses)

        # Compute statistical measures: mean and 95% confidence intervals
        mean = np.mean(responses_np, axis=0)  # Mean over all simulations at each time step
        ci_lower = np.percentile(responses_np, 2.5, axis=0)  # 2.5th percentile
        ci_upper = np.percentile(responses_np, 97.5, axis=0)  # 97.5th percentile

        # Plot individual time histories (transparent lines)
        for i in range(NUM_SIMS):
            response_time_ax.plot(time, responses_np[i, :], c='k', alpha=0.05)
        response_time_ax.set_xlabel('Time [s]')
        response_time_ax.set_ylabel(f'Displacement {disp_coord} [m]')

        # Plot mean and 95% confidence interval
        ax_statplot.plot(time, mean, c='k', label='Mean')
        ax_statplot.fill_between(time, ci_lower, ci_upper, alpha=0.4, color='darkgray', label='95% CI')
        ax_statplot.set_xlabel('Time [s]')
        ax_statplot.set_ylabel(f'Displacement {disp_coord} [m]')

        # Plot histogram of mean displacement values across simulations
        means_calculations = np.mean(responses_np, axis=1)  # Mean for each simulation
        mean_means = np.mean(means_calculations)  # Overall mean
        hist_ax.hist(means_calculations, bins=50, color='blue', alpha=0.5)
        hist_ax.axvline(mean_means, color='k', linestyle='dashed', linewidth=1, label='Mean of means')
        hist_ax.set_xlabel(f'Mean displacement {disp_coord} [m]')
        hist_ax.set_ylabel('Frequency')

        if name_of_the_model is not None and NUM_SIMS is not None:
            fig.suptitle(f'{name_of_the_model} - {NUM_SIMS} simulations')
        plt.show()
    
    @staticmethod
    def create_comparison_plots_multiple(all_samples: Dict, num_simulations: int, 
                                        get_variable_definitions_func=None):
        """
        Create comparative plots for multiple sampling methods.
        
        Parameters:
        -----------
        all_samples : dict
            Dictionary with method names as keys and sample dictionaries as values
        num_simulations : int
            Number of simulations
        get_variable_definitions_func : callable, optional
            Function to get variable definitions (needed for histogram plots)
        """
        # Set up the plotting
        variables = list(next(iter(all_samples.values())).keys())
        n_vars = len(variables)
        methods = list(all_samples.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        markers = ['o', 's', '^', 'v', 'D', 'p', '*']
        
        # Create figure with subplots for scatter plots (3x3 grid for up to 9 variables)
        n_cols = 3
        n_rows = int(np.ceil(n_vars / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        # Ensure axes is always a flat array
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot each variable
        for i, var_name in enumerate(variables):
            ax = axes[i]
            
            # Plot each method
            for j, method in enumerate(methods):
                data = all_samples[method][var_name]
                ax.scatter(range(num_simulations), data, alpha=0.6, 
                          label=method, color=colors[j], s=40, marker=markers[j % len(markers)])
                
                # Add mean line
                mean_val = np.mean(data)
                ax.axhline(y=mean_val, color=colors[j], linestyle='--', alpha=0.4, linewidth=1)
            
            ax.set_title(f'{var_name.replace("_", " ").title()}')
            ax.set_xlabel('Simulation Number')
            ax.set_ylabel('Value')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        if n_vars < len(axes):
            for i in range(n_vars, len(axes)):
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.suptitle(f'Comparison of Sampling Methods ({num_simulations} simulations)', 
                     y=1.02, fontsize=14)
        plt.show()
        
        # Create histogram comparison if variable definitions function is provided
        if get_variable_definitions_func is not None:
            PlottingUtilities.create_histogram_comparison_multiple(all_samples, get_variable_definitions_func)
    
    @staticmethod
    def create_histogram_comparison_multiple(all_samples: Dict, get_variable_definitions_func):
        """
        Create histogram comparison plots for multiple sampling methods.
        
        Parameters:
        -----------
        all_samples : dict
            Dictionary with method names as keys and sample dictionaries as values
        get_variable_definitions_func : callable
            Function to get variable definitions for theoretical PDF calculation
        """
        variables = list(next(iter(all_samples.values())).keys())
        n_vars = len(variables)
        methods = list(all_samples.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        # Create figure with subplots (3x3 grid for up to 9 variables)
        n_cols = 3
        n_rows = int(np.ceil(n_vars / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        # Ensure axes is always a flat array
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Get variable definitions to calculate theoretical PDFs
        var_definitions = get_variable_definitions_func()
        
        for i, var_name in enumerate(variables):
            ax = axes[i]
            
            # Get data range for all methods to determine PDF plot range
            all_data = []
            for method in methods:
                all_data.extend(all_samples[method][var_name])
            data_min = np.min(all_data)
            data_max = np.max(all_data)
            x_range = np.linspace(data_min, data_max, 200)
            
            # Plot histograms for each method
            for j, method in enumerate(methods):
                data = all_samples[method][var_name]
                ax.hist(data, alpha=0.5, bins=20, label=method, color=colors[j], density=True)
            
            # Plot theoretical PDF
            if var_name in var_definitions:
                params = var_definitions[var_name]
                
                if params['dist'] == 'lognormal':
                    # Lognormal PDF
                    mean_val = params['mean']
                    cov = params['cov']
                    sigma_ln = np.sqrt(np.log(1 + cov**2))
                    mu_ln = np.log(mean_val) - 0.5 * sigma_ln**2
                    pdf_values = stats.lognorm.pdf(x_range, s=sigma_ln, scale=np.exp(mu_ln))
                    ax.plot(x_range, pdf_values, 'k--', linewidth=2, label='Theoretical PDF', alpha=0.8)
                    
                elif params['dist'] == 'normal':
                    # Normal PDF
                    mean_val = params['mean']
                    if 'std' in params:
                        std_val = params['std']
                    elif 'cov' in params:
                        std_val = mean_val * params['cov']
                    else:
                        std_val = None
                    
                    if std_val is not None:
                        pdf_values = stats.norm.pdf(x_range, loc=mean_val, scale=std_val)
                        ax.plot(x_range, pdf_values, 'k--', linewidth=2, label='Theoretical PDF', alpha=0.8)
            
            ax.set_title(f'{var_name.replace("_", " ").title()}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        if n_vars < len(axes):
            for i in range(n_vars, len(axes)):
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.suptitle('Distribution Comparison of Sampling Methods', y=1.02, fontsize=14)
        plt.show()
    
    @staticmethod
    def create_pair_plot(all_samples: Dict, get_variable_definitions_func):
        """
        Create a pair plot showing only diagonal (histograms) and lower triangle (scatter plots)
        for Halton and Random sampling methods.
        
        Parameters:
        -----------
        all_samples : dict
            Dictionary with method names as keys and sample dictionaries as values
        get_variable_definitions_func : callable
            Function to get variable definitions for theoretical PDF calculation
        """
        # Select only Halton and Random methods
        methods_to_plot = []
        if 'Halton' in all_samples:
            methods_to_plot.append('Halton')
        elif 'Hammersley' in all_samples:  # Fallback if Halton was named differently
            methods_to_plot.append('Hammersley')
        
        if 'Random' in all_samples:
            methods_to_plot.append('Random')
        
        if not methods_to_plot:
            print("Warning: Halton or Random methods not found. Using available methods.")
            methods_to_plot = list(all_samples.keys())[:2]
        
        # Get variables from first method
        variables = list(all_samples[methods_to_plot[0]].keys())
        n_vars = len(variables)
        
        # Create figure with subplots for pair plot (only lower triangle + diagonal)
        fig, axes = plt.subplots(n_vars, n_vars, figsize=(3*n_vars, 3*n_vars))
        
        # Handle case where we have only one variable
        if n_vars == 1:
            axes = np.array([[axes]])
        elif not isinstance(axes, np.ndarray):
            axes = np.array(axes)
        
        # Colors for each method
        method_colors = {'Halton': 'blue', 'Hammersley': 'blue', 'Random': 'red'}
        method_alphas = {'Halton': 0.6, 'Hammersley': 0.6, 'Random': 0.6}
        
        var_definitions = get_variable_definitions_func()
        
        # Plot each pair
        for i in range(n_vars):
            for j in range(n_vars):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram for both methods
                    for method in methods_to_plot:
                        if method in all_samples:
                            data = all_samples[method][variables[i]]
                            color = method_colors.get(method, 'gray')
                            alpha = method_alphas.get(method, 0.6)
                            ax.hist(data, bins=20, alpha=alpha*0.7, color=color, 
                                   edgecolor='black', density=True, label=method)
                    
                    ax.set_ylabel('Density')
                    
                    # Add theoretical PDF on diagonal
                    if variables[i] in var_definitions:
                        params = var_definitions[variables[i]]
                        # Get data range from all methods
                        all_data = []
                        for method in methods_to_plot:
                            if method in all_samples:
                                all_data.extend(all_samples[method][variables[i]])
                        x_range = np.linspace(np.min(all_data), np.max(all_data), 200)
                        
                        if params['dist'] == 'lognormal':
                            mean_val = params['mean']
                            cov = params['cov']
                            sigma_ln = np.sqrt(np.log(1 + cov**2))
                            mu_ln = np.log(mean_val) - 0.5 * sigma_ln**2
                            pdf_values = stats.lognorm.pdf(x_range, s=sigma_ln, scale=np.exp(mu_ln))
                            ax.plot(x_range, pdf_values, 'k--', linewidth=2, alpha=0.8, label='Theoretical PDF')
                        elif params['dist'] == 'normal':
                            mean_val = params['mean']
                            if 'std' in params:
                                std_val = params['std']
                            elif 'cov' in params:
                                std_val = mean_val * params['cov']
                            if std_val is not None:
                                pdf_values = stats.norm.pdf(x_range, loc=mean_val, scale=std_val)
                                ax.plot(x_range, pdf_values, 'k--', linewidth=2, alpha=0.8, label='Theoretical PDF')
                    
                    if i == 0:  # Only show legend on first diagonal plot
                        ax.legend(fontsize=7)
                
                elif i > j:
                    # Lower triangle: scatter plot for both methods
                    for method in methods_to_plot:
                        if method in all_samples:
                            x_data = all_samples[method][variables[j]]
                            y_data = all_samples[method][variables[i]]
                            color = method_colors.get(method, 'gray')
                            alpha = method_alphas.get(method, 0.6)
                            ax.scatter(x_data, y_data, alpha=alpha, s=15, color=color, 
                                     edgecolors='black', linewidths=0.3, label=method)
                    
                    # Add correlation coefficient (using first method)
                    if methods_to_plot and methods_to_plot[0] in all_samples:
                        x_data = all_samples[methods_to_plot[0]][variables[j]]
                        y_data = all_samples[methods_to_plot[0]][variables[i]]
                        if len(x_data) > 1:
                            corr = np.corrcoef(x_data, y_data)[0, 1]
                            ax.text(0.05, 0.95, f'ρ={corr:.3f}', transform=ax.transAxes,
                                   fontsize=8, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    if i == n_vars - 1 and j == 0:  # Only show legend on bottom-left plot
                        ax.legend(fontsize=7)
                
                else:
                    # Upper triangle: hide these subplots
                    ax.axis('off')
                
                # Set labels
                if i == n_vars - 1 and i > j:
                    ax.set_xlabel(variables[j].replace('_', '\n'), fontsize=8)
                if j == 0 and i >= j:
                    ax.set_ylabel(variables[i].replace('_', '\n'), fontsize=8)
                
                # Remove ticks for cleaner look
                if i >= j:  # Only for diagonal and lower triangle
                    ax.tick_params(labelsize=6)
        
        methods_str = " & ".join(methods_to_plot)
        plt.suptitle(f'Pair Plot of Input Variables - {methods_str} Sampling (Lower Triangle)', 
                     y=0.995, fontsize=14)
        plt.tight_layout()
        plt.show()


class MorrisPlotting:
    """
    Class for Morris sensitivity analysis plotting utilities.
    """
    
    def __init__(self, sensitivity_indices: Dict, problem: Dict):
        """
        Initialize Morris plotting utilities.
        
        Parameters:
        -----------
        sensitivity_indices : dict
            Dictionary containing Morris sensitivity indices
        problem : dict
            Problem definition from SALib
        """
        self.sensitivity_indices = sensitivity_indices
        self.problem = problem
    
    def plot_all(self, save_plots: bool = True, output_dir: str = "sensitivity_plots", ranking: list = None):
        """
        Create all Morris sensitivity analysis plots.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to files
        output_dir : str
            Directory to save plots
        ranking : list, optional
            List of (variable_name, mu_star) tuples for ranking plot
        """
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create comprehensive sensitivity plots
        try:
            self.plot_mu_star_sigma(save_plots, output_dir)
        except Exception as e:
            print(f"Warning: Could not create mu_star_sigma plot: {e}")

        try:
            self.plot_mu_sigma(save_plots, output_dir)
        except Exception as e:
            print(f"Warning: Could not create mu_sigma plot: {e}")
        
        # Ranking plot requires ranking list
        if ranking is not None:
            try:
                self.plot_sensitivity_ranking(save_plots, output_dir, ranking)
            except Exception as e:
                print(f"Warning: Could not create sensitivity ranking plot: {e}")
        
        try:
            self.plot_variance_decomposition(save_plots, output_dir)
        except Exception as e:
            print(f"Warning: Could not create variance decomposition plot: {e}")
    
    def plot_mu_star_sigma(self, save_plots: bool, output_dir: str):
        """
        Plot mu_star vs sigma (Morris plot).
        
        This plot helps identify:
        - High μ* (top-right): Important variables with nonlinear/interaction effects
        - High μ*, low σ (right side): Important linear effects
        - Low μ* (left side): Unimportant variables
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract sensitivity indices
        mu_star = self.sensitivity_indices['mu_star']  # Mean absolute elementary effect (importance)
        sigma = self.sensitivity_indices['sigma']      # Standard deviation (nonlinearity/interactions)

        # Handle masked arrays (from SALib) and convert to regular numpy arrays
        if np.ma.isMaskedArray(mu_star):
            mu_star = mu_star.filled(np.nan)
        else:
            mu_star = np.asarray(mu_star, dtype=float)

        if np.ma.isMaskedArray(sigma):
            sigma = sigma.filled(np.nan)
        else:
            sigma = np.asarray(sigma, dtype=float)
        
        # Create scatter plot with color coding by variable index
        scatter = ax.scatter(mu_star, sigma, s=100, alpha=0.7, c=range(len(self.problem['names'])), 
                           cmap='viridis')

        # Dynamically adjust axis limits based on data range (handle edge cases)
        finite_mu = mu_star[np.isfinite(mu_star)]      # Filter out NaN/Inf values
        finite_sigma = sigma[np.isfinite(sigma)]

        # Handle empty arrays (fallback to zero)
        if finite_mu.size == 0:
            finite_mu = np.array([0.0])
        if finite_sigma.size == 0:
            finite_sigma = np.array([0.0])

        # Calculate data range and add padding for better visualization
        mu_star_min, mu_star_max = finite_mu.min(), finite_mu.max()
        sigma_min, sigma_max = finite_sigma.min(), finite_sigma.max()

        mu_star_range = mu_star_max - mu_star_min
        sigma_range = sigma_max - sigma_min

        # Add 10% padding (or minimum padding if range is zero)
        if mu_star_range <= 0:
            mu_star_padding = 0.1 * max(abs(mu_star_max), 1e-6)
        else:
            mu_star_padding = 0.1 * mu_star_range

        if sigma_range <= 0:
            sigma_padding = 0.1 * max(abs(sigma_max), 1e-6)
        else:
            sigma_padding = 0.1 * sigma_range

        # Set axis limits (ensure x-axis starts at 0 or above)
        ax.set_xlim(max(0.0, mu_star_min - mu_star_padding), mu_star_max + mu_star_padding)
        lower_sigma_bound = sigma_min - sigma_padding
        ax.set_ylim(lower_sigma_bound, sigma_max + sigma_padding)
        
        # Add variable labels near each point
        for i, var_name in enumerate(self.problem['names']):
            ax.annotate(var_name.replace('_', '\n'),  # Replace underscores with newlines
                       (mu_star[i], sigma[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, ha='left')
        
        # Add reference lines at μ*=0.5 and σ=0.5 (typical thresholds)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('μ* (Mean Absolute Elementary Effect)')
        ax.set_ylabel('σ (Standard Deviation of Elementary Effect)')
        ax.set_title('Morris Sensitivity Analysis: μ* vs σ')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar to show variable index
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Variable Index')
        
        try:
            plt.tight_layout()
        except:
            pass  # Ignore tight_layout warnings
        
        if save_plots:
            try:
                # Ensure output directory exists
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'morris_mu_star_sigma.png'), dpi=300, bbox_inches='tight')
                print(f"Saved plot: {os.path.join(output_dir, 'morris_mu_star_sigma.png')}")
            except Exception as e:
                print(f"Could not save mu_star_sigma plot: {e}")
        plt.close(fig)

    def plot_mu_sigma(self, save_plots: bool, output_dir: str):
        """Plot signed mu vs sigma (directional effects)."""
        fig, ax = plt.subplots(figsize=(10, 8))

        mu = self.sensitivity_indices['mu']
        sigma = self.sensitivity_indices['sigma']

        # Create scatter
        scatter = ax.scatter(mu, sigma, s=100, alpha=0.7, c=range(len(self.problem['names'])), cmap='coolwarm')

        # Add labels
        for i, var_name in enumerate(self.problem['names']):
            ax.annotate(var_name.replace('_', '\n'),
                        (mu[i], sigma[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, ha='left')

        # Axis limits
        finite_mu = mu[np.isfinite(mu)]
        finite_sigma = sigma[np.isfinite(sigma)]

        if finite_mu.size == 0:
            finite_mu = np.array([0.0])
        if finite_sigma.size == 0:
            finite_sigma = np.array([0.0])

        mu_min, mu_max = finite_mu.min(), finite_mu.max()
        sigma_min, sigma_max = finite_sigma.min(), finite_sigma.max()

        mu_range = mu_max - mu_min
        sigma_range = sigma_max - sigma_min

        mu_padding = 0.1 * mu_range if mu_range > 0 else 0.1 * max(abs(mu_max), 1e-6)
        sigma_padding = 0.1 * sigma_range if sigma_range > 0 else 0.1 * max(abs(sigma_max), 1e-6)

        ax.set_xlim(mu_min - mu_padding, mu_max + mu_padding)
        ax.set_ylim(sigma_min - sigma_padding, sigma_max + sigma_padding)

        # Reference axes
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('μ (Mean Elementary Effect)')
        ax.set_ylabel('σ (Standard Deviation of Elementary Effect)')
        ax.set_title('Morris Sensitivity Analysis: μ vs σ')
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Variable Index')

        try:
            plt.tight_layout()
        except:
            pass

        if save_plots:
            try:
                # Ensure output directory exists
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'morris_mu_sigma.png'), dpi=300, bbox_inches='tight')
                print(f"Saved plot: {os.path.join(output_dir, 'morris_mu_sigma.png')}")
            except Exception as e:
                print(f"Could not save mu_sigma plot: {e}")
        plt.close(fig)
    
    def plot_sensitivity_ranking(self, save_plots: bool, output_dir: str, ranking: list):
        """Plot sensitivity ranking bar chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        variables = [item[0].replace('_', ' ').title() for item in ranking]
        mu_star_values = np.asarray([item[1] for item in ranking], dtype=float)
        if np.ma.isMaskedArray(mu_star_values):
            mu_star_values = mu_star_values.filled(np.nan)
        mu_star_values = np.nan_to_num(mu_star_values, nan=0.0)
        
        bars = ax.barh(variables, mu_star_values, color='skyblue', alpha=0.7)

        # Pick an x-limit that stays close to the largest value to preserve readability
        max_mu = float(mu_star_values.max()) if mu_star_values.size else 0.0
        if max_mu <= 0:
            x_max = 1.0
        else:
            # Expand just enough to provide breathing room without dwarfing small bars
            x_max = max_mu * 1.1
        ax.set_xlim(0, x_max)

        text_offset = 0.02 * x_max

        # Add value labels on bars with dynamic positioning/formatting
        for bar in bars:
            width = bar.get_width()
            if width + text_offset <= x_max:
                label_x = width + text_offset
                ha = 'left'
            else:
                label_x = max(width - text_offset, 0)
                ha = 'right'

            if width == 0:
                value_text = '0'
            elif abs(width) < 1e-2 or abs(width) >= 1e3:
                value_text = f'{width:.3e}'
            else:
                value_text = f'{width:.3f}'

            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                    value_text, ha=ha, va='center', fontweight='bold')
        
        ax.set_xlabel('μ* (Mean Absolute Elementary Effect)')
        ax.set_title('Variable Sensitivity Ranking (Morris Method)')
        ax.grid(True, alpha=0.3, axis='x')
        
        try:
            plt.tight_layout()
        except:
            pass  # Ignore tight_layout warnings
        
        if save_plots:
            try:
                # Ensure output directory exists
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'sensitivity_ranking.png'), dpi=300, bbox_inches='tight')
                print(f"Saved plot: {os.path.join(output_dir, 'sensitivity_ranking.png')}")
            except Exception as e:
                print(f"Could not save sensitivity ranking plot: {e}")
        plt.close(fig)
    
    def plot_variance_decomposition(self, save_plots: bool, output_dir: str):
        """Plot variance decomposition pie chart."""
        mu_star = self.sensitivity_indices['mu_star']
        if np.ma.isMaskedArray(mu_star):
            mu_star = mu_star.filled(np.nan)
        else:
            mu_star = np.asarray(mu_star, dtype=float)
        mu_star = np.nan_to_num(mu_star, nan=0.0)
        
        # Normalize mu_star values to get relative importance
        total_mu_star = np.sum(mu_star)
        relative_importance = mu_star / total_mu_star * 100
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(relative_importance, 
                                         labels=[name.replace('_', ' ').title() for name in self.problem['names']],
                                         autopct='%1.1f%%',
                                         startangle=90,
                                         colors=plt.cm.Set3(np.linspace(0, 1, len(self.problem['names']))))
        
        # Improve text formatting
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Relative Importance of Variables (Morris Method)')
        
        try:
            plt.tight_layout()
        except:
            pass  # Ignore tight_layout warnings
        
        if save_plots:
            try:
                # Ensure output directory exists
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'variance_decomposition.png'), dpi=300, bbox_inches='tight')
                print(f"Saved plot: {os.path.join(output_dir, 'variance_decomposition.png')}")
            except Exception as e:
                print(f"Could not save variance decomposition plot: {e}")
        plt.close(fig)
    
    @staticmethod
    def plot_morris_response_summary(time_series: np.ndarray,
                                    response_y_series: np.ndarray,
                                    response_x_series: Optional[np.ndarray] = None,
                                    title: str = "Morris Response Summary",
                                    save_plots: bool = False,
                                    output_dir: str = "sensitivity_plots",
                                    filename: str = "morris_response_summary.png"):
        """
        Plot response summary (time histories, mean & 95% CI, histogram) for Morris outputs.

        Parameters
        ----------
        time_series : np.ndarray
            Array of shape (n_samples, n_timesteps) containing the time values for each sample.
        response_y_series : np.ndarray
            Array of shape (n_samples, n_timesteps) containing displacement Y responses for each sample.
        response_x_series : np.ndarray, optional
            Array of shape (n_samples, n_timesteps) containing displacement X responses for each sample.
        title : str
            Title for the figure.
        save_plots : bool
            Whether to save the plot
        output_dir : str
            Directory to save plots
        filename : str
            Filename for saved plot
        """
        if response_y_series.ndim != 2:
            raise ValueError("response_y_series must be a 2D array with shape (n_samples, n_timesteps)")

        n_samples, n_timesteps = response_y_series.shape

        if response_x_series is not None and response_x_series.ndim != 2:
            raise ValueError("response_x_series must be a 2D array with shape (n_samples, n_timesteps)")

        if time_series.ndim == 1:
            time = np.broadcast_to(time_series, (n_samples, n_timesteps))
        elif time_series.shape != response_y_series.shape:
            raise ValueError("time_series must either be 1D or have the same shape as response_y_series")
        else:
            time = time_series

        fig = plt.figure(layout='constrained', figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig)
        ax_time_y = fig.add_subplot(gs[0, :])
        ax_hist = fig.add_subplot(gs[1, 0])
        ax_time_x = fig.add_subplot(gs[1, 1])

        # Plot displacement Y histories (individual trajectories)
        for i in range(n_samples):
            ax_time_y.plot(time[i], response_y_series[i], color='red', alpha=0.3)

        ax_time_y.set_xlabel('Time [s]')
        ax_time_y.set_ylabel('Displacement Y [m]')
        ax_time_y.set_title('Displacement Y Histories')

        # Plot displacement X histories if available
        if response_x_series is not None:
            for i in range(n_samples):
                ax_time_x.plot(time[i], response_x_series[i], color='dimgray', alpha=0.3)
            ax_time_x.set_ylabel('Displacement X [m]')
        else:
            # Show message if X displacement data is not available
            ax_time_x.text(0.5, 0.5, 'Displacement X data not available', ha='center', va='center', transform=ax_time_x.transAxes)
            ax_time_x.set_ylabel('Displacement X [m]')

        ax_time_x.set_xlabel('Time [s]')
        ax_time_x.set_title('Displacement X Histories')

        # Calculate and plot mean and 95% confidence interval for Y (on Y panel)
        mean_y = np.nanmean(response_y_series, axis=0)  # Mean over samples at each time step
        ci_lower = np.nanpercentile(response_y_series, 2.5, axis=0)  # 2.5th percentile
        ci_upper = np.nanpercentile(response_y_series, 97.5, axis=0)  # 97.5th percentile
        ax_time_y.plot(time[0], mean_y, color='red', linewidth=2, label='Mean')
        ax_time_y.fill_between(time[0], ci_lower, ci_upper, color='lightcoral', alpha=0.3, label='95% CI')
        ax_time_y.legend()

        # Histogram of mean displacement Y values (one mean per sample)
        mean_values_y = np.nanmean(response_y_series, axis=1)  # Mean over time for each sample
        bins = min(50, max(5, n_samples // 2))  # Adaptive bin count based on sample size
        ax_hist.hist(mean_values_y, bins=bins, color='red', alpha=0.6)
        ax_hist.axvline(mean_values_y.mean(), color='k', linestyle='dashed', linewidth=1, label='Mean')
        ax_hist.set_xlabel('Mean Displacement Y [m]')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Distribution of Mean Displacement Y')
        ax_hist.legend()

        fig.suptitle(title, fontsize=14)
        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            try:
                fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                print(f"Saved plot: {os.path.join(output_dir, filename)}")
            except Exception as e:
                print(f"Warning: Could not save response summary plot: {e}")

        plt.close(fig)


class SensitivityComparisonPlotting:
    """
    Class for comparing results from different sensitivity analysis methods (Morris vs SRC).
    """

    @staticmethod
    def plot_method_comparison(morris_results: Dict, src_results: Dict,
                              save_plots: bool = True, output_dir: str = "sensitivity_plots"):
        """
        Create side-by-side comparison plots of Morris and RBD sensitivity results.

        Parameters:
        -----------
        morris_results : Dict
            Dictionary from Morris sensitivity analysis (from JSON or get_sensitivity_summary())
        src_results : Dict
            Dictionary from RBD sensitivity analysis (from JSON or get_sensitivity_summary())
        save_plots : bool
            Whether to save plots to files
        output_dir : str
            Directory to save plots
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Extract rankings from both methods
        morris_ranking = morris_results.get('ranking', [])
        src_ranking = src_results.get('ranking', [])

        # Create side-by-side bar chart comparison
        SensitivityComparisonPlotting._plot_ranking_comparison(
            morris_ranking, src_ranking, save_plots, output_dir
        )

        # Create scatter plot showing correlation between rankings
        SensitivityComparisonPlotting._plot_ranking_scatter(
            morris_ranking, src_ranking, save_plots, output_dir
        )

    @staticmethod
    def _plot_ranking_comparison(morris_ranking: list, src_ranking: list,
                                 save_plots: bool, output_dir: str):
        """
        Create side-by-side bar charts comparing parameter rankings from both methods.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Extract names and values
        morris_names = [r[0] for r in morris_ranking]
        morris_values = [r[1] for r in morris_ranking]
        src_names = [r[0] for r in src_ranking]
        src_values = [r[1] for r in src_ranking]

        # Morris plot
        colors_morris = plt.cm.Blues(np.linspace(0.5, 0.9, len(morris_names)))
        ax1.barh(range(len(morris_names)), morris_values, color=colors_morris)
        ax1.set_yticks(range(len(morris_names)))
        ax1.set_yticklabels(morris_names, fontsize=10)
        ax1.set_xlabel('Sensitivity Index (mu*)', fontsize=11, fontweight='bold')
        ax1.set_title('Morris Sensitivity Ranking', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        # RBD plot
        colors_rbd = plt.cm.Oranges(np.linspace(0.5, 0.9, len(src_names)))
        ax2.barh(range(len(src_names)), src_values, color=colors_rbd)
        ax2.set_yticks(range(len(src_names)))
        ax2.set_yticklabels(src_names, fontsize=10)
        ax2.set_xlabel('Sensitivity Index (|correlation|)', fontsize=11, fontweight='bold')
        ax2.set_title('SRC Sensitivity Ranking', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_plots:
            filename = os.path.join(output_dir, 'sensitivity_methods_comparison.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved comparison plot: {filename}")

        plt.close(fig)

    @staticmethod
    def _plot_ranking_scatter(morris_ranking: list, src_ranking: list,
                              save_plots: bool, output_dir: str):
        """
        Create scatter plot showing how parameters rank differently between methods.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create mapping of parameter names to rankings for both methods
        morris_dict = {r[0]: i + 1 for i, r in enumerate(morris_ranking)}
        src_dict = {r[0]: i + 1 for i, r in enumerate(src_ranking)}

        # Get all parameter names
        all_params = set(morris_dict.keys()) | set(src_dict.keys())

        # Create arrays for scatter plot
        morris_ranks = []
        src_ranks = []
        labels = []

        for param in sorted(all_params):
            morris_rank = morris_dict.get(param, len(morris_dict) + 1)
            rbd_rank = src_dict.get(param, len(src_dict) + 1)
            morris_ranks.append(morris_rank)
            src_ranks.append(rbd_rank)
            labels.append(param)

        # Scatter plot
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        scatter = ax.scatter(morris_ranks, src_ranks, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=2)

        # Add labels for each point
        for i, label in enumerate(labels):
            ax.annotate(label, (morris_ranks[i], src_ranks[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

        # Add diagonal line (perfect agreement)
        max_rank = max(len(morris_dict), len(src_dict))
        ax.plot([0.5, max_rank + 0.5], [0.5, max_rank + 0.5], 'k--', alpha=0.3, linewidth=2, label='Perfect agreement')

        ax.set_xlabel('Morris Ranking (1 = most sensitive)', fontsize=11, fontweight='bold')
        ax.set_ylabel('SRC Ranking (1 = most sensitive)', fontsize=11, fontweight='bold')
        ax.set_title('Sensitivity Method Ranking Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, max_rank + 0.5)
        ax.set_ylim(0.5, max_rank + 0.5)

        plt.tight_layout()

        if save_plots:
            filename = os.path.join(output_dir, 'sensitivity_methods_ranking_scatter.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved scatter plot: {filename}")

        plt.close(fig)

    @staticmethod
    def load_and_compare(morris_json: str = 'morris_sensitivity_results.json',
                        src_json: str = 'rbd_sensitivity_results.json',
                        save_plots: bool = True, output_dir: str = 'sensitivity_plots'):
        """
        Load results from JSON files and create comparison plots.

        Parameters:
        -----------
        morris_json : str
            Path to Morris results JSON file
        src_json : str
            Path to RBD results JSON file
        save_plots : bool
            Whether to save plots
        output_dir : str
            Output directory for plots
        """
        import json

        # Load Morris results
        with open(morris_json, 'r') as f:
            morris_data = json.load(f)

        # Load RBD results
        with open(src_json, 'r') as f:
            rbd_data = json.load(f)

        # Create comparison plots
        SensitivityComparisonPlotting.plot_method_comparison(
            morris_data['summary'],
            rbd_data['summary'],
            save_plots=save_plots,
            output_dir=output_dir
        )


class VelocityResponsePlotting:
    """
    Class for plotting velocity response data and processed metrics.
    """
    
    @staticmethod
    def plot_velocity_response(processed_data: Dict[str, Any],
                              title: str = "Velocity Response Analysis",
                              save_plots: bool = False,
                              output_dir: str = "output_plots",
                              filename: str = "velocity_response.png"):
        """
        Plot velocity response data including velocity Y, effective velocity, and PSD.
        
        This function creates a 3-panel plot showing:
        - Velocity Y over time (mm/s)
        - Effective velocity over time (mm/s)
        - Power Spectral Density (PSD) vs frequency ((mm/s)^2/Hz)
        
        Parameters:
        -----------
        processed_data : dict
            Dictionary from process_response_data containing:
            - 'time': Time array
            - 'velocity_y': Velocity Y array (mm/s)
            - 'v_eff': Effective velocity array (mm/s)
            - 'frequency_Pxx': Frequency array for PSD (Hz)
            - 'Pxx': PSD array ((mm/s)^2/Hz)
        title : str
            Title for the figure
        save_plots : bool
            Whether to save the plot to file
        output_dir : str
            Directory to save plots
        filename : str
            Filename for saved plot
        """
        # Extract data
        time = processed_data.get('time')
        velocity_y = processed_data.get('velocity_y')
        v_eff = processed_data.get('v_eff')
        frequency_Pxx = processed_data.get('frequency_Pxx')
        Pxx = processed_data.get('Pxx')
        
        if time is None or velocity_y is None:
            raise ValueError("processed_data must contain 'time' and 'velocity_y' keys")
        
        # Create figure with 3 subplots
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 4))
        
        # Plot 1: Velocity Y over time
        ax[0].plot(time, velocity_y, label=r"v$_{y}$", color="blue")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Velocity Y (mm/s)")
        ax[0].set_xlim(left=0)
        ax[0].grid()
        ax[0].legend()
        
        # Plot 2: Effective velocity over time
        if v_eff is not None:
            # Ensure v_eff has the same length as time
            v_eff_plot = v_eff[:len(time)] if len(v_eff) >= len(time) else v_eff
            time_v_eff = time[:len(v_eff_plot)] if len(v_eff) < len(time) else time
            ax[1].plot(time_v_eff, v_eff_plot, label=r"v$_{eff}$", color="orange")
            ax[1].set_xlabel("Time (s)")
            ax[1].set_ylabel("V$_eff$ (mm/s)")
            ax[1].set_xlim(left=0)
            ax[1].set_ylim(bottom=0)
            ax[1].grid()
            ax[1].legend()
        else:
            ax[1].text(0.5, 0.5, 'Effective velocity data not available', 
                      ha='center', va='center', transform=ax[1].transAxes)
            ax[1].set_xlabel("Time (s)")
            ax[1].set_ylabel("V$_eff$ (mm/s)")
        
        # Plot 3: Power Spectral Density
        if frequency_Pxx is not None and Pxx is not None:
            ax[2].plot(frequency_Pxx, Pxx, label=r"v$_{y}$", color="blue")
            ax[2].set_xlabel("Frequency (Hz)")
            ax[2].set_ylabel("PSD ([mm/s]$^2$/Hz)")
            ax[2].set_xlim(0, 100)
            ax[2].set_ylim(bottom=0)
            ax[2].grid()
            ax[2].legend()
        else:
            ax[2].text(0.5, 0.5, 'PSD data not available', 
                      ha='center', va='center', transform=ax[2].transAxes)
            ax[2].set_xlabel("Frequency (Hz)")
            ax[2].set_ylabel("PSD ([mm/s]$^2$/Hz)")
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            try:
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                print(f"Saved plot: {os.path.join(output_dir, filename)}")
            except Exception as e:
                print(f"Warning: Could not save velocity response plot: {e}")
        
        plt.close(fig)
    
    @staticmethod
    def plot_velocity_metrics_summary(processed_data_list: list,
                                     save_plots: bool = False,
                                     output_dir: str = "output_plots",
                                     filename: str = "velocity_metrics_summary.png"):
        """
        Plot summary of velocity metrics (V_y_max, V_eff_max, PSD_max, Freq_PSD_max) 
        across multiple processed outputs.
        
        Parameters:
        -----------
        processed_data_list : list
            List of dictionaries from process_response_data, each containing:
            - 'V_y_max': Maximum absolute velocity Y (mm/s)
            - 'V_eff_max': Maximum effective velocity (mm/s)
            - 'PSD_max': Maximum PSD ((mm/s)^2/Hz)
            - 'Freq_PSD_max': Frequency at max PSD (Hz)
        save_plots : bool
            Whether to save the plot to file
        output_dir : str
            Directory to save plots
        filename : str
            Filename for saved plot
        """
        if not processed_data_list:
            raise ValueError("processed_data_list cannot be empty")
        
        # Extract metrics
        v_y_max_values = [data.get('V_y_max', 0) for data in processed_data_list]
        v_eff_max_values = [data.get('V_eff_max', 0) for data in processed_data_list]
        psd_max_values = [data.get('PSD_max', 0) for data in processed_data_list]
        freq_psd_max_values = [data.get('Freq_PSD_max', 0) for data in processed_data_list]
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics = [
            ('V_y_max', v_y_max_values, 'Maximum Velocity Y (mm/s)'),
            ('V_eff_max', v_eff_max_values, 'Maximum Effective Velocity (mm/s)'),
            ('PSD_max', psd_max_values, 'Maximum PSD ((mm/s)²/Hz)'),
            ('Freq_PSD_max', freq_psd_max_values, 'Frequency at Max PSD (Hz)')
        ]
        
        for i, (metric_name, values, ylabel) in enumerate(metrics):
            ax = axes[i]
            ax.hist(values, bins=min(30, len(values)), alpha=0.7, color=plt.cm.Set1(i))
            ax.axvline(np.mean(values), color='k', linestyle='dashed', linewidth=2, label='Mean')
            ax.set_xlabel(ylabel)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric_name.replace("_", " ").title()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Velocity Response Metrics Summary', fontsize=14)
        plt.tight_layout()
        
        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            try:
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                print(f"Saved plot: {os.path.join(output_dir, filename)}")
            except Exception as e:
                print(f"Warning: Could not save metrics summary plot: {e}")
        
        plt.close(fig)
