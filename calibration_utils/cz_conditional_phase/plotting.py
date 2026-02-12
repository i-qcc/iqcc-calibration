from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from calibration_utils.cz_conditional_phase.analysis import FitResults
from qualibration_libs.core import BatchableList
import xarray as xr


def plot_raw_data_with_fit(
    fit_results: xr.Dataset,
    qubit_pairs,
) -> tuple[plt.Figure, plt.Figure]:
    """
    Plot the CZ phase calibration data showing phase difference vs amplitude with fit,
    and a separate figure with control qubit population vs amplitude.

    Parameters:
    -----------
    fit_results : xr.Dataset
        Fit results for each qubit pair
        Optimal amplitudes for each qubit pair
    qubit_pairs : BatchableList
        List of qubit pairs

    Returns:
    --------
    tuple[plt.Figure, plt.Figure | None]
        (phase_figure, control_population_figure or None if state_control data not in fit_results)
    """
    n_pairs = len(qubit_pairs)
    cols = min(4, n_pairs)  # Max 4 columns
    rows = (n_pairs + cols - 1) // cols  # Ceiling division
    figsize = (3 * cols, 3 * rows)

    # --- Figure 1: Phase difference + fit ---
    fig_phase, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, qp in enumerate(qubit_pairs):
        ax = axes[i]
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

    # Hide unused axes
    for j in range(n_pairs, len(axes)):
        axes[j].axis("off")

    fig_phase.suptitle("CZ phase calibration (phase difference + fit)")
    fig_phase.tight_layout()

    # --- Figure 2: Control qubit population (same layout and styling), if data available ---
    has_state_control = all(
        v in fit_results.data_vars for v in ("g_state_control", "e_state_control", "f_state_control")
    )
    fig_control = None
    if has_state_control:
        fig_control, axes_control = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes_control = axes_control.flatten()

        for i, qp in enumerate(qubit_pairs):
            ax = axes_control[i]
            qp_name = qp.name
            fit_result = fit_results.sel(qubit_pair=qp_name)

            def amp_to_detuning_MHz(amp):
                return -(amp**2) * qp.qubit_control.freq_vs_flux_01_quad_term / 1e6  # Convert Hz to MHz

            def detuning_MHz_to_amp(detuning_MHz):
                return np.sqrt(-detuning_MHz * 1e6 / qp.qubit_control.freq_vs_flux_01_quad_term)

            data_g = fit_results.g_state_control.sel(qubit_pair=qp_name, control_axis=1).mean(dim="frame")
            data_e = fit_results.e_state_control.sel(qubit_pair=qp_name, control_axis=1).mean(dim="frame")
            data_f = fit_results.f_state_control.sel(qubit_pair=qp_name, control_axis=1).mean(dim="frame")
            amps = fit_result.amp_full.values if "amp_full" in fit_result.coords else fit_result.amp.values

            ax.plot(amps, data_g, label="g", color="blue")
            ax.plot(amps, data_e, label="e", color="red")
            ax.plot(amps, data_f, label="f", color="green")
            ax.axhline(0.0, color="gray", linestyle="--", lw=0.5)
            ax.axhline(1.0, color="gray", linestyle="--", lw=0.5)

            success_value = bool(fit_result.success.values) if hasattr(fit_result.success, 'values') else bool(fit_result.success)
            optimal_amp_value = float(fit_result.optimal_amplitude.values) if hasattr(fit_result.optimal_amplitude, 'values') else float(fit_result.optimal_amplitude)
            if success_value and not (np.isnan(optimal_amp_value) or np.isinf(optimal_amp_value)):
                ax.axvline(optimal_amp_value, color="red", linestyle="--", lw=0.5, label="optimal")

            ax.set_title(qp_name)
            ax.set_xlabel("Amplitude (V)")
            ax.set_ylabel("Control qubit population")
            secax = ax.secondary_xaxis("top", functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
            secax.set_xlabel("Detuning (MHz)")
            ax.legend()

        for j in range(n_pairs, len(axes_control)):
            axes_control[j].axis("off")

        fig_control.suptitle("CZ phase calibration (control qubit population)")
        fig_control.tight_layout()

    return fig_phase, fig_control