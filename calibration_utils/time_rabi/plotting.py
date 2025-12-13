from typing import List, Dict, Any
import xarray as xr
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

def plot_raw_data_with_fit(
    ds_raw: xr.Dataset, 
    qubits: List[AnyTransmon], 
    ds_fit: xr.Dataset,
    fit_results: Dict[str, Any] # <-- Pass in fit results (as dicts)
) -> plt.Figure:
    """
    Plot the raw Time Rabi data and the fitted cosine function.
    
    Also displays the results from both Duration and Amplitude analysis
    in the title of each subplot.
    """
    
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    
    # Check if we're using state discrimination or I/Q
    if "state" in ds_raw:
        signal_name = "state"
        y_label = "State"
    else:
        signal_name = "V" if "V" in ds_raw else "I"
        y_label = f"{signal_name} (V)"
        
    for ax, qubit_info in grid_iter(grid):
        q_name = qubit_info["qubit"]
        
        # Plot raw data
        ds_raw.sel(qubit=q_name)[signal_name].plot(
            ax=ax, 
            marker=".", 
            linestyle="None", 
            label="Raw data"
        )
        
        # Plot fit
        ds_fit.sel(qubit=q_name).plot(ax=ax, label="Fit")
        
        # --- Create the title string with fit results ---
        fit_result = fit_results[q_name]
        
        if fit_result["success"]:
            # Format results for display
            f_mhz = fit_result['f'] * 1e3
            pi_dur = fit_result['opt_dur_pi']
            pi_half_dur = fit_result['opt_dur_pi_half']
            
            target_f = fit_result['target_freq']
            amp_old = fit_result['drive_amp_scale']
            amp_new = fit_result['amp_fit']
            
            # Create a multi-line title string
            title_str = (
                f"{q_name}\n"
                f"Freq: {f_mhz:.2f} MHz | π-pulse: {pi_dur:.1f} ns\n"
                f"New Amp: {amp_new:.3f} (Old: {amp_old:.3f})"
            )
        else:
            title_str = f"{q_name}\n(Fit Failed)"
        
        # --- End title logic ---
        
        ax.set_ylabel(y_label)
        ax.set_xlabel("Pulse duration [ns]")
        # Set the multi-line title, slightly smaller
        ax.set_title(title_str, fontsize=10) 
        
        # Keep legend in a non-overlapping spot
        ax.legend(loc='lower right')

    grid.fig.suptitle("Time Rabi (Duration & Amplitude Analysis)")
    plt.tight_layout()
    # Do not call plt.show() here, let the node script handle it
    
    return grid.fig

def plot_raw_data(
    ds_raw: xr.Dataset, 
    qubits: List[AnyTransmon]
) -> plt.Figure:
    """
    Plot only the raw Time Rabi data.
    """
    
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    
    # Check if we're using state discrimination or I/Q
    if "state" in ds_raw:
        signal_name = "state"
        y_label = "State"
    else:
        signal_name = "V" if "V" in ds_raw else "I"
        y_label = f"{signal_name} (V)"
        
    for ax, qubit_info in grid_iter(grid):
        q_name = qubit_info["qubit"]
        
        # Plot raw data
        ds_raw.sel(qubit=q_name)[signal_name].plot(
            ax=ax, 
            marker=".", 
            linestyle="None",
            label="Raw data"
        )
        
        ax.set_ylabel(y_label)
        ax.set_xlabel("Pulse duration [ns]")
        ax.set_title(q_name) # Simple title
        ax.legend(loc='lower right')

    grid.fig.suptitle("Time Rabi (Raw Data Only)")
    plt.tight_layout()
    # Do not call plt.show()
    
    return grid.fig

