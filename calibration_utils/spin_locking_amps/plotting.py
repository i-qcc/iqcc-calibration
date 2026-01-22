from typing import List, Dict
import xarray as xr
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_t2_sl_vs_amplitude(
    t2_sl_data: Dict[str, xr.Dataset],
    qubits: List[AnyTransmon]
) -> plt.Figure:
    """
    Plot T2_SL vs amplitude for each qubit.
    X-axis: voltage (V) calculated from full_scale_power_dbm and amp_prefactor
    Y-axis: T2_SL (µs)
    Shows dots for values and error bars for errors.
    """
    # Create a minimal dataset with qubit dimension for QubitGrid
    qubit_names = [q.name for q in qubits]
    dummy_ds = xr.Dataset({
        "dummy": (["qubit"], [0] * len(qubit_names))
    }, coords={"qubit": qubit_names})
    
    grid = QubitGrid(dummy_ds, [q.grid_location for q in qubits])
    
    for ax, qubit_info in grid_iter(grid):
        q_name = qubit_info["qubit"]
        
        if q_name in t2_sl_data:
            ds = t2_sl_data[q_name]
            
            # Find the qubit object to get full_scale_power_dbm and base pulse amplitude
            qubit_obj = next((q for q in qubits if q.name == q_name), None)
            if qubit_obj is None or not hasattr(qubit_obj, 'xy_sl') or not hasattr(qubit_obj.xy_sl, 'opx_output'):
                # Fallback to amplitude factor if we can't get power info
                amp_success = ds.amp_prefactor.values[ds.success.values]
                x_label = "Amplitude factor"
            else:
                # Get full_scale_power_dbm from qubit's xy_sl output (used for spin locking)
                full_scale_power_dbm = qubit_obj.xy_sl.opx_output.full_scale_power_dbm
                
                # Get base amplitude of the x180_Square pulse
                base_amplitude = qubit_obj.xy_sl.operations["x180_Square"].amplitude
                
                # Convert power (dBm) to milliwatts: x_mw = 10^(x_dbm / 10)
                x_mw = 10 ** (full_scale_power_dbm / 10)
                
                # Convert milliwatts to voltage for 50 Ω load: x_v = sqrt((2 * 50 * x_mw) / 1000)
                # This is the voltage for full-scale waveform (amplitude = 1.0)
                full_scale_voltage = np.sqrt((2 * 50 * x_mw) / 1000)
                
                # Convert amplitude factors to voltages: amp_prefactor * base_amplitude * full_scale_voltage
                # This accounts for both the sweep (amp_prefactor) and the base pulse amplitude
                amp_success = ds.amp_prefactor.values[ds.success.values] * base_amplitude * full_scale_voltage
                x_label = "Voltage [V]"
            
            # Convert from ns to µs (divide by 1000)
            t2_sl_success = ds.T2_SL.values[ds.success.values] / 1000.0
            t2_sl_error_success = ds.T2_SL_error.values[ds.success.values] / 1000.0
            
            if len(amp_success) > 0:
                # Plot dots for values
                ax.plot(amp_success, t2_sl_success, 'o', label="T2_SL", markersize=6, color='blue')
                # Plot error bars
                ax.errorbar(amp_success, t2_sl_success, yerr=t2_sl_error_success, 
                           fmt='none', capsize=3, capthick=1, color='blue', alpha=0.6)
            
            # Plot failed fits as red x's
            failed_mask = ~ds.success.values
            if np.any(failed_mask):
                if qubit_obj is not None and hasattr(qubit_obj, 'xy_sl') and hasattr(qubit_obj.xy_sl, 'opx_output'):
                    full_scale_power_dbm = qubit_obj.xy_sl.opx_output.full_scale_power_dbm
                    base_amplitude = qubit_obj.xy_sl.operations["x180_Square"].amplitude
                    x_mw = 10 ** (full_scale_power_dbm / 10)
                    full_scale_voltage = np.sqrt((2 * 50 * x_mw) / 1000)
                    amp_failed = ds.amp_prefactor.values[failed_mask] * base_amplitude * full_scale_voltage
                else:
                    amp_failed = ds.amp_prefactor.values[failed_mask]
                t2_sl_failed = ds.T2_SL.values[failed_mask] / 1000.0
                ax.plot(amp_failed, t2_sl_failed, 'rx', label="Failed fit", markersize=6)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel("T2_SL [µs]")
            ax.set_title(q_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{q_name}\n(No data)")
    
    grid.fig.suptitle("Spin Locking: T2_SL vs Amplitude")
    plt.tight_layout()
    
    return grid.fig


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset, 
    qubits: List[AnyTransmon], 
    ds_fit: xr.Dataset = None,
    selected_amplitudes: List[float] = None
):
    """
    Plot the raw Spin Locking data and the fitted exponential decay for selected amplitudes.
    If selected_amplitudes is None, plots all amplitudes (may be crowded).
    Returns a single figure for both state discrimination and I, Q mode (only I is shown for I/Q mode).
    """
    # Check if we're using state discrimination or I/Q
    if "state" in ds_raw:
        signal_name = "state"
        y_label = "State"
        
        grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
        for ax, qubit_info in grid_iter(grid):
            q_name = qubit_info["qubit"]
            
            # If no amplitudes specified, use all
            if selected_amplitudes is None:
                selected_amplitudes = sorted(ds_raw.amp_prefactor.values) if "amp_prefactor" in ds_raw.dims else []
            
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
                    if ds_fit is not None and q_name in ds_fit.qubit.values:
                        try:
                            fit_data = ds_fit.sel(qubit=q_name, amp_prefactor=amp)
                            if "fit_data" in fit_data.data_vars:
                                from qualibration_libs.analysis import decay_exp
                                fit_curve = decay_exp(
                                    ds_raw.spin_locking_time,
                                    fit_data.fit_data.sel(fit_vals="a"),
                                    fit_data.fit_data.sel(fit_vals="offset"),
                                    fit_data.fit_data.sel(fit_vals="decay"),
                                )
                                if fit_curve is not None:
                                    ax.plot(ds_raw.spin_locking_time, fit_curve, "--", 
                                           label=f"Fit (amp={amp:.3f})", alpha=0.7)
                        except:
                            pass
                except KeyError:
                    continue
            
            ax.set_ylabel(y_label)
            ax.set_xlabel("Spin locking time [ns]")
            ax.set_title(q_name)
            ax.legend(loc='best', fontsize=8)
        
        grid.fig.suptitle(f"Spin Locking: Raw Data & Fits (Selected Amplitudes)")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        return grid.fig
        
    else:
        # I, Q mode - show only I (not Q) for amplitude sweeps
        # Figure for I only
        grid_I = QubitGrid(ds_raw, [q.grid_location for q in qubits])
        for ax, qubit_info in grid_iter(grid_I):
            q_name = qubit_info["qubit"]
            
            if selected_amplitudes is None:
                selected_amplitudes = sorted(ds_raw.amp_prefactor.values) if "amp_prefactor" in ds_raw.dims else []
            
            for amp in selected_amplitudes:
                try:
                    (ds_raw.sel(qubit=q_name, amp_prefactor=amp).I * 1e3).plot(
                        ax=ax, 
                        marker=".", 
                        linestyle="None", 
                        label=f"Raw (amp={amp:.3f})"
                    )
                    
                    if ds_fit is not None and q_name in ds_fit.qubit.values:
                        try:
                            fit_data = ds_fit.sel(qubit=q_name, amp_prefactor=amp)
                            if "fit_data_I" in fit_data.data_vars:
                                from qualibration_libs.analysis import decay_exp
                                fit_curve = decay_exp(
                                    ds_raw.spin_locking_time,
                                    fit_data.fit_data_I.sel(fit_vals="a"),
                                    fit_data.fit_data_I.sel(fit_vals="offset"),
                                    fit_data.fit_data_I.sel(fit_vals="decay"),
                                )
                                if fit_curve is not None:
                                    ax.plot(ds_raw.spin_locking_time, fit_curve * 1e3, "--", 
                                           label=f"Fit (amp={amp:.3f})", alpha=0.7)
                        except:
                            pass
                except KeyError:
                    continue
            
            ax.set_ylabel("Trans. amp. I [mV]")
            ax.set_xlabel("Spin locking time [ns]")
            ax.set_title(q_name)
            ax.legend(loc='best', fontsize=8)
        
        grid_I.fig.suptitle(f"Spin Locking: Raw Data & Fits - I (Selected Amplitudes)")
        grid_I.fig.set_size_inches(15, 9)
        grid_I.fig.tight_layout()
        
        return grid_I.fig
