from typing import List, Dict, Any
import xarray as xr
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

def plot_rabi_freq_vs_amplitude(
    rabi_freqs: Dict[str, xr.Dataset],
    qubits: List[AnyTransmon]
) -> plt.Figure:
    """
    Plot rabi frequency vs amplitude for each qubit.
    X-axis: amplitude factor
    Y-axis: rabi frequency (MHz)
    """
    # Create a minimal dataset with qubit dimension for QubitGrid
    qubit_names = [q.name for q in qubits]
    dummy_ds = xr.Dataset({
        "dummy": (["qubit"], [0] * len(qubit_names))
    }, coords={"qubit": qubit_names})
    
    grid = QubitGrid(dummy_ds, [q.grid_location for q in qubits])
    
    for ax, qubit_info in grid_iter(grid):
        q_name = qubit_info["qubit"]
        
        if q_name in rabi_freqs:
            ds = rabi_freqs[q_name]
            
            # Plot successful fits only
            success_mask = ds.success.values
            amp_success = ds.amp_prefactor.values[success_mask]
            freq_success = ds.rabi_frequency.values[success_mask]
            
            if len(amp_success) > 0:
                ax.plot(amp_success, freq_success, 'o-', label="Rabi frequency", markersize=4)
            
            # Plot failed fits as red x's
            failed_mask = ~success_mask
            if np.any(failed_mask):
                amp_failed = ds.amp_prefactor.values[failed_mask]
                freq_failed = ds.rabi_frequency.values[failed_mask]
                ax.plot(amp_failed, freq_failed, 'rx', label="Failed fit", markersize=6)
            
            ax.set_xlabel("Amplitude factor")
            ax.set_ylabel("Rabi frequency [MHz]")
            ax.set_title(q_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{q_name}\n(No data)")
    
    grid.fig.suptitle("Time Rabi: Rabi Frequency vs Amplitude")
    plt.tight_layout()
    
    return grid.fig


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset, 
    qubits: List[AnyTransmon], 
    ds_fit: xr.Dataset,
    fit_results: Dict[str, Dict[float, Any]],
    selected_amplitudes: List[float] = None
) -> plt.Figure:
    """
    Plot the raw Time Rabi data and the fitted cosine function for selected amplitudes.
    If selected_amplitudes is None, plots all amplitudes (may be crowded).
    """
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    
    # Check if we're using state discrimination or I/Q
    if "state" in ds_raw:
        signal_name = "state"
        y_label = "State"
    else:
        signal_name = "V" if "V" in ds_raw else "I"
        y_label = f"{signal_name} (V)"
    
    # If no amplitudes specified, use all
    if selected_amplitudes is None:
        selected_amplitudes = sorted(ds_raw.amp_prefactor.values) if "amp_prefactor" in ds_raw.dims else []
    
    for ax, qubit_info in grid_iter(grid):
        q_name = qubit_info["qubit"]
        
        # Plot raw data for selected amplitudes
        for amp in selected_amplitudes:
            try:
                ds_raw.sel(qubit=q_name, amp_prefactor=amp)[signal_name].plot(
                    ax=ax, 
                    marker=".", 
                    linestyle="None", 
                    label=f"Raw (amp={amp:.3f})"
                )
                
                # Plot fit if available
                if q_name in fit_results and amp in fit_results[q_name]:
                    fit_result = fit_results[q_name][amp]
                    if fit_result.get("success", False):
                        # Get fit curve from ds_fit
                        if "amp_prefactor" in ds_fit.dims:
                            try:
                                ds_fit.sel(qubit=q_name, amp_prefactor=amp).plot(
                                    ax=ax, 
                                    label=f"Fit (amp={amp:.3f})",
                                    alpha=0.7
                                )
                            except:
                                pass
            except KeyError:
                continue
        
        ax.set_ylabel(y_label)
        ax.set_xlabel("Pulse duration [ns]")
        ax.set_title(q_name)
        ax.legend(loc='best', fontsize=8)
    
    grid.fig.suptitle(f"Time Rabi: Raw Data & Fits (Selected Amplitudes)")
    plt.tight_layout()
    
    return grid.fig
