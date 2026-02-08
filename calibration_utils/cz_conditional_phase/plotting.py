from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from calibration_utils.cz_conditional_phase.analysis import FitResults
from qualibration_libs.core import BatchableList
import xarray as xr


def plot_raw_data_with_fit(
    fit_results: xr.Dataset,
    qubit_pairs,
) -> plt.Figure:
    """
    Plot the CZ phase calibration data showing phase difference vs amplitude with fit.

    Parameters:
    -----------
    fit_results : xr.Dataset
        Fit results for each qubit pair
        Optimal amplitudes for each qubit pair
    qubit_pairs : BatchableList
        List of qubit pairs

    Returns:
    --------
    plt.Figure
        The generated figure
    """
    n_pairs = len(qubit_pairs)
    # Two subplots per pair: phase (top) and state_control (bottom)
    rows = 2
    cols = max(1, n_pairs)
    fig, axes_grid = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    # Column-major order so axes[2*i] = phase, axes[2*i+1] = state_control for pair i
    axes = axes_grid.T.flatten()

    for i, qp in enumerate(qubit_pairs):
        ax = axes[2 * i]
        qp_name = qp.name
        fit_result = fit_results.sel(qubit_pair=qp_name)

        # Plot phase difference data
        fit_result.phase_diff.plot.line(ax=ax, x="amp_full")

        # Extract scalar values from xarray DataArrays
        success_value = bool(fit_result.success.values) if hasattr(fit_result.success, 'values') else bool(fit_result.success)
        optimal_amp_value = float(fit_result.optimal_amplitude.values) if hasattr(fit_result.optimal_amplitude, 'values') else float(fit_result.optimal_amplitude)

        # Plot fitted curve if available and valid
        if success_value and not np.all(np.isnan(fit_result.fitted_curve)):
            ax.plot(fit_result.phase_diff.amp_full, fit_result.fitted_curve)

        # Mark optimal point only if fit was successful and amplitude is valid
        if success_value and not (np.isnan(optimal_amp_value) or np.isinf(optimal_amp_value)):
            ax.plot([optimal_amp_value], [0.5], marker="o", color="red")
            ax.axhline(y=0.5, color="red", linestyle="--", lw=0.5)
            ax.axvline(x=optimal_amp_value, color="red", linestyle="--", lw=0.5)

        # Add red X marker for failed fits (similar to node 09)
        if not success_value or np.isnan(optimal_amp_value) or np.isinf(optimal_amp_value):
            ax.scatter([1.0], [1.0], marker='x', s=100, c='red', linewidths=2, 
                      transform=ax.transAxes, clip_on=False, zorder=10)

        # Add secondary x-axis for detuning in MHz
        def amp_to_detuning_MHz(amp):
            return -(amp**2) * qp.qubit_control.freq_vs_flux_01_quad_term / 1e6  # Convert Hz to MHz

        def detuning_MHz_to_amp(detuning_MHz):
            return np.sqrt(-detuning_MHz * 1e6 / qp.qubit_control.freq_vs_flux_01_quad_term)

        secax = ax.secondary_xaxis("top", functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
        secax.set_xlabel("Detuning (MHz)")

        ax.set_title(qp_name)
        ax.set_xlabel("Amplitude (V)")
        ax.set_ylabel("Phase difference")

        # Add secondary plot below: state_control for control_axis=1 averaged over frame (if available)
        ax_sub = axes[2 * i + 1]
        has_state_control = False
        if "g_state_control" in fit_results.data_vars and "e_state_control" in fit_results.data_vars and "f_state_control" in fit_results.data_vars:
            data_g = fit_results.g_state_control.sel(qubit_pair=qp_name, control_axis=1).mean(dim="frame")
            data_e = fit_results.e_state_control.sel(qubit_pair=qp_name, control_axis=1).mean(dim="frame")
            data_f = fit_results.f_state_control.sel(qubit_pair=qp_name, control_axis=1).mean(dim="frame")
            has_state_control = True
        elif "state_control" in fit_results.data_vars:
            # Derive g/e/f populations from state_control (values 0=g, 1=e, 2=f)
            sc = fit_results.state_control.sel(qubit_pair=qp_name, control_axis=1)
            data_g = (sc == 0).mean(dim="frame").astype(float)
            data_e = (sc == 1).mean(dim="frame").astype(float)
            data_f = (sc == 2).mean(dim="frame").astype(float)
            has_state_control = True
        else:
            # Per-pair stream names: state_control1, state_control2, ...
            var_name = f"state_control{i + 1}"
            if var_name in fit_results.data_vars:
                sc = fit_results[var_name].sel(control_axis=1)
                data_g = (sc == 0).mean(dim="frame").astype(float)
                data_e = (sc == 1).mean(dim="frame").astype(float)
                data_f = (sc == 2).mean(dim="frame").astype(float)
                has_state_control = True
        if has_state_control:
            amps = fit_result.amp_full.values if "amp_full" in fit_result.coords else fit_result.amp.values
            ax_sub.plot(amps, data_g, label="g", color="blue")
            ax_sub.plot(amps, data_e, label="e", color="red")
            ax_sub.plot(amps, data_f, label="f", color="green")
            opt_amp = fit_result.optimal_amplitude.item() if fit_result.optimal_amplitude.size == 1 else float(fit_result.optimal_amplitude.values)
            if not (np.isnan(opt_amp) or np.isinf(opt_amp)):
                ax_sub.axvline(opt_amp, color="red", linestyle="--", lw=0.5, label="optimal")
            ax_sub.axhline(0.0, color="red", linestyle="--", lw=0.5)
            ax_sub.axhline(1.0, color="red", linestyle="--", lw=0.5)
            ax_sub.set_ylabel("Control qubit population")
            ax_sub.set_xlabel("Amplitude (V)")
            secax2 = ax_sub.secondary_xaxis("top", functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
            secax2.set_xlabel("Detuning (MHz)")
            ax_sub.legend()
        else:
            ax_sub.axis("off")

    # Hide unused axes (we use 2*n_pairs; grid may have more if cols > n_pairs)
    for i in range(2 * n_pairs, len(axes)):
        axes[i].axis("off")

    fig.suptitle("CZ phase calibration (phase difference + fit)")
    fig.tight_layout()

    return fig
