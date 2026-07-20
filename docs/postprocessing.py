"""
Post-processing utilities for geotechnical model analysis.

This module contains classes for statistical post-processing and summary reporting.
"""

import numpy as np
from typing import Dict, Any, Callable


class PostProcessingUtilities:
    """
    Class for post-processing utilities including statistics tables and summaries.
    """
    
    @staticmethod
    def print_statistics_table(all_samples: Dict, get_variable_definitions_func: Callable):
        """
        Print a table showing mean, std, and COV for each variable and each sampling method.
        Includes percentage of absolute difference from theoretical values.
        
        Parameters:
        -----------
        all_samples : dict
            Dictionary with method names as keys and sample dictionaries as values
        get_variable_definitions_func : callable
            Function to get variable definitions for theoretical values
        """
        variables = list(next(iter(all_samples.values())).keys())
        methods = list(all_samples.keys())
        
        # Get theoretical values from variable definitions
        var_definitions = get_variable_definitions_func()
        
        print("\n" + "="*150)
        print("STATISTICS TABLE: Mean, Standard Deviation, and Coefficient of Variation")
        print("Percentage values show absolute difference from theoretical values")
        print("="*150)
        
        # Print header
        header = f"{'Variable':<25}"
        for method in methods:
            header += f"{method:>40}"
        print(header)
        print("-" * 150)
        
        # Print statistics for each variable
        for var_name in variables:
            print(f"\n{var_name.replace('_', ' ').title()}:")
            
            # Get theoretical values
            if var_name in var_definitions:
                params = var_definitions[var_name]
                theoretical_mean = params['mean']
                
                # Calculate theoretical std
                if params['dist'] == 'lognormal':
                    cov = params['cov']
                    # For lognormal: std = mean * COV
                    theoretical_std = theoretical_mean * cov
                    theoretical_cov = cov * 100  # Convert to percentage
                elif params['dist'] == 'normal':
                    if 'std' in params:
                        theoretical_std = params['std']
                    elif 'cov' in params:
                        theoretical_std = theoretical_mean * params['cov']
                    else:
                        theoretical_std = None
                    if theoretical_std is not None and theoretical_mean != 0:
                        theoretical_cov = (theoretical_std / abs(theoretical_mean)) * 100
                    else:
                        theoretical_cov = None
            else:
                theoretical_mean = None
                theoretical_std = None
                theoretical_cov = None
            
            # Mean row
            mean_row = f"{'  Mean':<25}"
            for method in methods:
                data = all_samples[method][var_name]
                mean_val = np.mean(data)
                if abs(mean_val) < 1e-3 or abs(mean_val) > 1e6:
                    mean_str = f"{mean_val:.4e}"
                else:
                    mean_str = f"{mean_val:.6f}"
                
                # Calculate percentage difference
                if theoretical_mean is not None and theoretical_mean != 0:
                    pct_diff = abs(mean_val - theoretical_mean) / abs(theoretical_mean) * 100
                    mean_str += f" ({pct_diff:.2f}%)"
                else:
                    mean_str += " (N/A)"
                
                mean_row += f"{mean_str:>40}"
            print(mean_row)
            
            # Std row
            std_row = f"{'  Std':<25}"
            for method in methods:
                data = all_samples[method][var_name]
                std_val = np.std(data)
                if abs(std_val) < 1e-3 or abs(std_val) > 1e6:
                    std_str = f"{std_val:.4e}"
                else:
                    std_str = f"{std_val:.6f}"
                
                # Calculate percentage difference
                if theoretical_std is not None and theoretical_std != 0:
                    pct_diff = abs(std_val - theoretical_std) / abs(theoretical_std) * 100
                    std_str += f" ({pct_diff:.2f}%)"
                else:
                    std_str += " (N/A)"
                
                std_row += f"{std_str:>40}"
            print(std_row)
            
            # COV row
            cov_row = f"{'  COV (%)':<25}"
            for method in methods:
                data = all_samples[method][var_name]
                mean_val = np.mean(data)
                std_val = np.std(data)
                if mean_val != 0:
                    cov_val = (std_val / abs(mean_val)) * 100
                    cov_str = f"{cov_val:.4f}"
                else:
                    cov_str = "N/A"
                
                # Calculate percentage difference
                if theoretical_cov is not None and cov_str != "N/A":
                    pct_diff = abs(cov_val - theoretical_cov) / abs(theoretical_cov) * 100
                    cov_str += f" ({pct_diff:.2f}%)"
                else:
                    cov_str += " (N/A)"
                
                cov_row += f"{cov_str:>40}"
            print(cov_row)
        
        print("\n" + "="*150 + "\n")
    
    @staticmethod
    def print_summary_statistics(lhs_samples: Dict, random_samples: Dict):
        """
        Print summary statistics comparing both sampling methods.
        
        Parameters:
        -----------
        lhs_samples : dict
            Dictionary containing LHS samples
        random_samples : dict
            Dictionary containing Random samples
        """
        print("\n" + "="*80)
        print("SUMMARY STATISTICS COMPARISON")
        print("="*80)
        
        for var_name in lhs_samples.keys():
            lhs_data = lhs_samples[var_name]
            random_data = random_samples[var_name]
            
            print(f"\n{var_name.replace('_', ' ').title()}:")
            print(f"  LHS    - Mean: {np.mean(lhs_data):.6e}, Std: {np.std(lhs_data):.6e}, CV: {np.std(lhs_data)/np.mean(lhs_data)*100:.2f}%")
            print(f"  Random - Mean: {np.mean(random_data):.6e}, Std: {np.std(random_data):.6e}, CV: {np.std(random_data)/np.mean(random_data)*100:.2f}%")
            
            # Calculate coefficient of variation for comparison
            lhs_cv = np.std(lhs_data) / np.mean(lhs_data) * 100
            random_cv = np.std(random_data) / np.mean(random_data) * 100
            cv_diff = abs(lhs_cv - random_cv)
            
            print(f"  CV Difference: {cv_diff:.2f}% (target CV: 10.00%)")
    
    @staticmethod
    def print_sensitivity_summary(summary: Dict):
        """
        Print a formatted summary of sensitivity analysis results.
        
        Parameters:
        -----------
        summary : Dict
            Summary of sensitivity analysis results
        """
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nVariable Ranking (by μ* - Mean Absolute Elementary Effect):")
        print("-" * 60)
        for i, (var_name, mu_star) in enumerate(summary['ranking'], 1):
            print(f"{i:2d}. {var_name.replace('_', ' ').title():25s} μ* = {mu_star:.4f}")
        
        # Detailed Interpretation section - commented out for now
        # print(f"\nDetailed Interpretation:")
        # print("-" * 60)
        # for var_name, interpretation in summary['interpretation'].items():
        #     print(f"\n{var_name.replace('_', ' ').title()}:")
        #     print(f"  Sensitivity Level: {interpretation['sensitivity_level']}")
        #     print(f"  Linearity:         {interpretation['linearity']}")
        #     print(f"  Interaction:       {interpretation['interaction']}")
        #     print(f"  μ* (importance):   {interpretation['mu_star']:.4f}")
        #     print(f"  μ (linear effect): {interpretation['mu']:.4f}")
        #     print(f"  σ (nonlinear):     {interpretation['sigma']:.4f}")

