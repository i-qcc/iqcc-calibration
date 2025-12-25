from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def _is_fit_successful(fit: xr.Dataset) -> bool:
    """Check if the fit is successful from the fit dataset."""
    return bool(fit.success.values) if "success" in fit.data_vars else True


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Ramsey vs flux")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_parabolas_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, outcomes: dict = None):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.
    outcomes : dict, optional
        Dictionary containing outcomes ("successful" or "failed") for each qubit.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    
    # Create a mapping from qubit name to qubit object for easy access
    qubit_dict = {q.name: q for q in qubits}
    for ax, qubit in grid_iter(grid):
        qubit_obj = qubit_dict.get(qubit["qubit"])
        fit_qubit = fits.sel(qubit=qubit["qubit"])
        
        # Calculate x-axis limits for this specific qubit
        frequency = fit_qubit.sel(fit_vals="f").fit_results
        flux_bias = frequency.flux_bias
        flux_min = float(flux_bias.min().values)
        flux_max = float(flux_bias.max().values)
        
        # Extend limits to include target_offset and flux_offset for this qubit
        # Only if fit is successful
        if _is_fit_successful(fit_qubit):
            for var_name in ["target_offset", "flux_offset"]:
                if var_name in fit_qubit.data_vars:
                    try:
                        val = float(fit_qubit[var_name].values)
                        if not np.isnan(val):
                            flux_min = min(flux_min, val)
                            flux_max = max(flux_max, val)
                    except (KeyError, ValueError):
                        pass
        
        # Add 5% padding
        if flux_max > flux_min:
            padding = 0.05 * (flux_max - flux_min)
            flux_min -= padding
            flux_max += padding
        
        # Plot with the calculated xlim for this qubit
        plot_individual_parabolas_with_fit(ax, ds, qubit, fit_qubit, outcomes, qubit_obj, flux_min, flux_max)
        # Set x-axis range for this subplot
        ax.set_xlim(flux_min, flux_max)

    grid.fig.suptitle("Ramsey vs flux frequency ")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    
    # Create legend entries manually
    legend_handles = [
        Line2D([0], [0], color='blue', marker='.', linestyle='', markersize=8, label='Data'),
        Line2D([0], [0], color='orange', linestyle='-', linewidth=2, label='Parabolic fit'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='Flux offset'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='Detuning from SweetSpot'),
        Line2D([0], [0], color='orange', linestyle='-.', linewidth=4, label='Target offset'),
        Line2D([0], [0], color='red', marker='x', linestyle='', markersize=6, markeredgewidth=2, label='Fit failed'),
    ]
    grid.fig.legend(legend_handles, [h.get_label() for h in legend_handles], 
                    loc='upper right', bbox_to_anchor=(0.98, 0.98),
                    frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Idle_time (uS)")
    ax.set_ylabel(" Flux (V)")

    # Only plot flux offset if fit is successful
    if _is_fit_successful(fit):
        flux_offset = fit.flux_offset
        ax.axhline(flux_offset, color="red", linestyle="--", label="Flux offset")


def plot_individual_parabolas_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None, outcomes: dict = None, qubit_obj: AnyTransmon = None, xlim_min: float = None, xlim_max: float = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).
    outcomes : dict, optional
        Dictionary containing outcomes ("successful" or "failed") for each qubit.
    qubit_obj : AnyTransmon, optional
        The qubit object for accessing target_detuning_from_sweet_spot.
    xlim_min : float, optional
        Minimum x-axis limit (for extending parabolic fit).
    xlim_max : float, optional
        Maximum x-axis limit (for extending parabolic fit).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    - The parabolic fit will extend to xlim_min/xlim_max if provided, otherwise uses data range.
    """
    detuning = float(fit.artifitial_detuning.values)
    frequency = fit.sel(fit_vals="f").fit_results  # frequency is in GHz
    flux_bias = frequency.flux_bias
    
    # Plot data points (convert GHz to MHz and subtract detuning)
    (frequency * 1e3 - detuning).plot(ax=ax, linestyle="", marker=".", label="Data", color="blue")
    
    # Only plot parabolic fit and calculate offsets if fit is successful
    if _is_fit_successful(fit):
        # Cache offset values to avoid repeated access
        flux_offset_val = float(fit.flux_offset.values)  # V
        freq_offset_val = float(fit.freq_offset.values)  # Hz
        
        # Reconstruct polynomial coefficients from stored fit parameters
        quad_term = float(fit.quad_term.values)  # MHz/V^2
        c2 = -quad_term / 1e6  # GHz/V^2
        c1 = -2 * flux_offset_val * c2  # GHz/V
        c0 = freq_offset_val / 1e9 - c2 * flux_offset_val**2 - c1 * flux_offset_val  # GHz
        
        # Determine flux range for plotting parabola
        data_flux_min = float(flux_bias.min().values)
        data_flux_max = float(flux_bias.max().values)
        
        if xlim_min is not None and xlim_max is not None:
            # Extend parabola to cover full xlim range
            plot_flux_min = min(xlim_min, data_flux_min)
            plot_flux_max = max(xlim_max, data_flux_max)
        else:
            # Use data range only
            plot_flux_min = data_flux_min
            plot_flux_max = data_flux_max
        
        # Create smooth flux range for plotting
        flux_smooth = np.linspace(plot_flux_min, plot_flux_max, 200)
        
        # Plot smooth parabola
        freq_smooth = c2 * flux_smooth**2 + c1 * flux_smooth + c0
        freq_smooth_mhz = freq_smooth * 1e3 - detuning
        ax.plot(flux_smooth, freq_smooth_mhz, linestyle="-", color="orange", linewidth=2, label="Parabolic fit")
        
        # Plot flux offset and detuning lines
        ax.axvline(flux_offset_val, color="red", linestyle="--", linewidth=4, label="Flux offset")
        ax.axhline(freq_offset_val * 1e-6 - detuning, color="green", linestyle="--", label="Detuning from SweetSpot")
        
        # Plot target offset line if non-zero
        if (qubit_obj is not None and hasattr(qubit_obj, 'xy') and 
            hasattr(qubit_obj.xy, 'target_detuning_from_sweet_spot')):
            target_detuning = qubit_obj.xy.target_detuning_from_sweet_spot
            if abs(target_detuning) > 1e-6 and "target_offset" in fit.data_vars:
                try:
                    target_offset_val = float(fit.target_offset.values)
                    if not np.isnan(target_offset_val):
                        ax.axvline(target_offset_val, linestyle="-.", linewidth=4, color="orange",
                                 label=f"target offset ({target_detuning/1e6:.1f} MHz)", alpha=0.7)
                except (KeyError, ValueError, AttributeError):
                    pass
    
    # Set y-axis limits based on data range
    data_mhz = (frequency * 1e3 - detuning).values
    y_min = float(np.nanmin(data_mhz))
    y_max = float(np.nanmax(data_mhz))
    y_padding = max(0.1 * (y_max - y_min), 5.0)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Flux offset (V)")
    ax.set_ylabel("detuning (MHz)")
    
    # Mark failed fits
    if outcomes and outcomes.get(qubit["qubit"]) == "failed":
        ax.scatter([1.0], [1.0], marker='x', s=100, c='red', linewidths=2, 
                  transform=ax.transAxes, clip_on=False, zorder=10)
