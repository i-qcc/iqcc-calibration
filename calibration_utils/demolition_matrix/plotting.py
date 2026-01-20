from typing import List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibrate import QualibrationNode
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_confusion_and_demolition_matrices(node: QualibrationNode):
    """
    Plot confusion matrices and demolition error matrices for all qubits.
    If depletion_time optimization was performed, also plot the optimization results.
    
    Parameters:
    -----------
    node : QualibrationNode
        The calibration node containing the fit results.
        
    Returns:
    --------
    fig_confusion : matplotlib.figure.Figure
        Figure containing confusion matrix plots.
    fig_demolition : matplotlib.figure.Figure
        Figure containing demolition error matrix plots.
    fig_optimization : matplotlib.figure.Figure or None
        Figure containing depletion_time optimization plots, or None if not optimizing.
    """
    qubits = node.namespace["qubits"]
    fit_results = node.results["fit_results"]
    ds_fit = node.results.get("ds_fit", node.results["ds_raw"])
    
    # Check if we have depletion_time optimization data
    has_optimization = "depletion_time" in ds_fit.dims
    
    # Create grid based on qubit locations (same as IQ blobs)
    grid_confusion = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    grid_demolition = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    
    # Plot confusion matrices
    for ax, qubit in grid_iter(grid_confusion):
        q_name = qubit["qubit"]
        fit_result = fit_results[q_name]
        plot_individual_confusion_matrix(ax, qubit, fit_result)
    
    grid_confusion.fig.suptitle("Readout Confusion Matrix")
    
    # Get number of measurements from parameters
    num_measurements = getattr(node.parameters, 'num_of_measurement', 1)
    grid_confusion.fig.set_size_inches(15, 9)
    grid_confusion.fig.tight_layout()
    
    # Plot demolition error matrices
    for ax, qubit in grid_iter(grid_demolition):
        q_name = qubit["qubit"]
        fit_result = fit_results[q_name]
        plot_individual_demolition_matrix(ax, qubit, fit_result, num_measurements)
    
    grid_demolition.fig.suptitle(f"Measurement Demolition Error Matrix (N={num_measurements + 1} measurements)")
    grid_demolition.fig.set_size_inches(15, 9)
    grid_demolition.fig.tight_layout()
    
    # Plot depletion_time optimization if available
    fig_optimization = None
    if has_optimization:
        fig_optimization = plot_depletion_time_optimization(node)
    
    plt.show()
    
    return grid_confusion.fig, grid_demolition.fig, fig_optimization


def plot_individual_confusion_matrix(ax: Axes, qubit: dict[str, str], fit_result: dict):
    """
    Plot individual confusion matrix on a given axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the matrix.
    qubit : dict[str, str]
        Mapping to the qubit to plot.
    fit_result : dict
        Dictionary containing the fit results for this qubit.
    """
    confusion = np.array(fit_result["confusion_matrix"])
    
    # Use same style as IQ blobs confusion matrix
    ax.imshow(confusion, vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels=["|g>", "|e>"])
    ax.set_yticklabels(labels=["|g>", "|e>"])
    ax.set_ylabel("Prepared")
    ax.set_xlabel("Measured")
    
    # Add text annotations with appropriate colors (same as IQ blobs)
    # Use "k" for dark backgrounds and "w" for light backgrounds
    ax.text(0, 0, f"{100 * confusion[0][0]:.1f}%", ha="center", va="center", 
            color="k" if confusion[0][0] > 0.5 else "w", fontweight='bold')
    ax.text(1, 0, f"{100 * confusion[0][1]:.1f}%", ha="center", va="center", 
            color="k" if confusion[0][1] > 0.5 else "w", fontweight='bold')
    ax.text(0, 1, f"{100 * confusion[1][0]:.1f}%", ha="center", va="center", 
            color="k" if confusion[1][0] > 0.5 else "w", fontweight='bold')
    ax.text(1, 1, f"{100 * confusion[1][1]:.1f}%", ha="center", va="center", 
            color="k" if confusion[1][1] > 0.5 else "w", fontweight='bold')
    ax.set_title(qubit["qubit"])


def plot_individual_demolition_matrix(ax: Axes, qubit: dict[str, str], fit_result: dict, num_measurements: int = 1):
    """
    Plot individual demolition error matrix on a given axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the matrix.
    qubit : dict[str, str]
        Mapping to the qubit to plot.
    fit_result : dict
        Dictionary containing the fit results for this qubit.
    num_measurements : int
        Number of consecutive measurements between first and last (default 1).
        The last measurement is measurement #(num_measurements + 1).
    """
    demolition = np.array(fit_result["demolition_error_matrix"])
    
    # Use same colormap style as confusion matrix
    ax.imshow(demolition, vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels=["0", "1"])
    ax.set_yticklabels(labels=["0", "1"])
    ax.set_ylabel("First Measurement")
    last_measurement_num = num_measurements + 1
    ax.set_xlabel(f"Measurement #{last_measurement_num}")
    
    # Add text annotations with appropriate colors
    for i in range(2):
        for j in range(2):
            value = demolition[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{100 * value:.1f}%", ha="center", va="center", 
                       color="k" if value > 0.5 else "w", fontweight='bold')
            else:
                ax.text(j, i, "N/A", ha="center", va="center", 
                       color="k", fontweight='bold')
    
    # Calculate total fidelity (probability of preserving state): (P(0|0) + P(1|1)) / 2
    total_fidelity = (demolition[0, 0] + demolition[1, 1]) / 2
    
    # Calculate per-measurement demolition using geometric model:
    # If each measurement has fidelity f, after n measurements: total_fidelity = f^n
    # So per-measurement fidelity = total_fidelity^(1/n)
    # And per-measurement demolition = 1 - per-measurement fidelity
    if num_measurements > 0 and total_fidelity > 0:
        per_measurement_fidelity = total_fidelity ** (1.0 / num_measurements)
        per_measurement_demolition = 1 - per_measurement_fidelity
    else:
        per_measurement_demolition = 1 - total_fidelity
    
    # Add title with per-measurement demolition and optimized depletion_time if available
    title = f"{qubit['qubit']}\nAvg demolition/meas: {100 * per_measurement_demolition:.2f}%"
    if fit_result.get("optimal_depletion_time") is not None:
        title += f" (dt={fit_result['optimal_depletion_time']} ns)"
    ax.set_title(title)


def plot_depletion_time_optimization(node: QualibrationNode) -> Figure:
    """
    Plot depletion_time optimization results showing diagonal sum vs depletion_time.
    
    Parameters:
    -----------
    node : QualibrationNode
        The calibration node containing the fit results.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing depletion_time optimization plots.
    """
    qubits = node.namespace["qubits"]
    fit_results = node.results["fit_results"]
    ds_fit = node.results.get("ds_fit", node.results["ds_raw"])
    num_measurements = getattr(node.parameters, 'num_of_measurement', 1)
    
    # Create grid based on qubit locations
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    
    # Plot optimization curves
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        fit_result = fit_results[q_name]
        
        # Get P(0|0), P(1|1), and P(0|g) data for this qubit
        p00_var = f"demolition_p00_{q_name}"
        p11_var = f"demolition_p11_{q_name}"
        p0g_var = f"confusion_p0g_{q_name}"
        
        if p00_var in ds_fit.data_vars and p11_var in ds_fit.data_vars:
            p00_data = ds_fit[p00_var]
            p11_data = ds_fit[p11_var]
            # Convert depletion times from clock cycles to nanoseconds (*4)
            depletion_times_ns = p00_data.depletion_time.values * 4
            
            # Plot both curves
            ax.plot(depletion_times_ns, p00_data.values, 'o-', linewidth=2, markersize=6, label='P(0|0)')
            ax.plot(depletion_times_ns, p11_data.values, 's-', linewidth=2, markersize=6, label='P(1|1)')
            
            # Plot P(0|g) baseline and threshold if available
            if p0g_var in ds_fit.data_vars:
                p0g_data = ds_fit[p0g_var]
                ax.plot(depletion_times_ns, p0g_data.values, '--', linewidth=1, color='gray', label='P(0|g) baseline')
                ax.plot(depletion_times_ns, p0g_data.values - 0.1, ':', linewidth=1, color='gray', label='P(0|g) - 10%')
            
            # Mark the optimal point (optimal_depletion_time is already in ns from analysis)
            if fit_result.get("optimal_depletion_time") is not None:
                optimal_dt = fit_result["optimal_depletion_time"]
                optimal_idx = np.argmin(np.abs(depletion_times_ns - optimal_dt))
                optimal_p11 = p11_data.values[optimal_idx]
                ax.plot(optimal_dt, optimal_p11, 'r*', markersize=15, label='Optimal', zorder=10)
                ax.axvline(optimal_dt, color='r', linestyle='--', alpha=0.5)
            
            ax.legend(fontsize=8)
        
        ax.set_xlabel("Depletion Time (ns)")
        ax.set_ylabel("Probability")
        ax.set_title(f"{q_name}\nOptimal: {fit_result.get('optimal_depletion_time', 'N/A')} ns")
        ax.grid(True, alpha=0.3)
    
    grid.fig.suptitle(f"Depletion Time Optimization (N={num_measurements + 1} measurements)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    
    return grid.fig

