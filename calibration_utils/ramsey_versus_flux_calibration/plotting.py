import re
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def _excluded_qubit_font_sizes(n_subplots: int):
    """Font sizes for excluded-qubit X and text, scaled by number of subplots."""
    scale = max(1, n_subplots ** 0.5)
    return max(12, int(96 / scale)), max(6, int(20 / scale))


def _mark_excluded_qubit(ax, qubit_name: str, fs_x: int, fs_text: int):
    """Mark a subplot as excluded with green X and explanatory text."""
    ax.text(0.5, 0.5, 'X', transform=ax.transAxes, fontsize=fs_x,
            color='green', fontweight='bold', ha='center', va='center')
    ax.text(0.5, 0.15, 'Excluded: qubit set outside sweetspot', transform=ax.transAxes, 
            fontsize=fs_text, color='black', ha='center', va='center')
    ax.set_title(qubit_name)
    ax.set_xticks([])
    ax.set_yticks([])


def _create_grid_with_all_locations(ds: xr.Dataset, grid_locations: List, qubit_names: List[str]):
    """Create a QubitGrid-like object that includes all specified grid locations."""
    clean_up = lambda s: re.sub("[^0-9]", "", s)
    
    grid_indices = [
        tuple(map(int, [clean_up(x) for x in (loc.split(",") if isinstance(loc, str) else [str(x) for x in loc])]))
        for loc in grid_locations
    ]
    grid_name_mapping = dict(zip(grid_indices, qubit_names))
    
    rows, cols = [idx[1] for idx in grid_indices], [idx[0] for idx in grid_indices]
    min_row, max_row, min_col = min(rows), max(rows), min(cols)
    shape = (max_row - min_row + 1, max(cols) - min_col + 1)
    
    figure, all_axes = plt.subplots(*shape, figsize=(shape[1] * 3, shape[0] * 3), squeeze=False)
    grid_axes, name_dicts = [], []
    
    for row, axis_row in enumerate(all_axes):
        for col, ax in enumerate(axis_row):
            grid_idx = (col + min_col, max_row - row)
            if grid_idx in grid_indices:
                grid_axes.append(ax)
                if (name := grid_name_mapping.get(grid_idx)) is not None:
                    name_dicts.append({"qubit": name})
            else:
                ax.axis("off")
    
    class GridResult:
        def __init__(self, fig, axes, name_dicts):
            self.fig, self.axes, self.name_dicts = fig, [axes], [name_dicts]
    
    return GridResult(figure, grid_axes, name_dicts)


def _is_fit_successful(fit: xr.Dataset) -> bool:
    """Check if the fit is successful from the fit dataset."""
    return bool(fit.success.values) if "success" in fit.data_vars else True


def _is_lo_limit_exceeded(fit: xr.Dataset) -> bool:
    """Check if the LO frequency limit is exceeded from the fit dataset."""
    return bool(fit.lo_limit_exceeded.values) if "lo_limit_exceeded" in fit.data_vars else False


def _should_show_fit(fit: xr.Dataset) -> bool:
    """Check if we should show the fitted line (fit succeeded, even if LO limit exceeded)."""
    # Show fit if it was successful OR if LO limit was exceeded (meaning fit succeeded but frequency is out of range)
    return _is_fit_successful(fit) or _is_lo_limit_exceeded(fit)


def _get_flux_values(ds: xr.Dataset, qubit_name: str, fallback_array: xr.DataArray = None):
    """Get flux values for a qubit, using per-qubit values if available."""
    if "flux_actual" in ds.data_vars:
        return ds.flux_actual.sel(qubit=qubit_name).values
    return fallback_array.values if fallback_array is not None else ds.flux_bias.values


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, 
                           excluded_qubits: Optional[List[AnyTransmon]] = None):
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
    excluded_qubits : list of AnyTransmon, optional
        Qubits excluded from the experiment (e.g., not at sweep spot).

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    # Include all qubits in grid creation so excluded qubits have subplot positions
    all_qubits = list(qubits) + (excluded_qubits or [])
    all_grid_locs = [q.grid_location for q in all_qubits]
    included_names = {q.name for q in qubits}
    
    grid = QubitGrid(ds, [q.grid_location for q in qubits]) if not excluded_qubits else \
           _create_grid_with_all_locations(ds, all_grid_locs, [q.name for q in all_qubits])
    
    n_subplots = len(grid.fig.axes)
    fs_x, fs_text = _excluded_qubit_font_sizes(n_subplots)
    
    for ax, qubit in grid_iter(grid):
        if qubit["qubit"] in included_names:
            plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
        else:
            _mark_excluded_qubit(ax, qubit["qubit"], fs_x, fs_text)
    
    grid.fig.suptitle("Ramsey vs flux")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_parabolas_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, 
                            outcomes: dict = None, excluded_qubits: Optional[List[AnyTransmon]] = None):
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
    excluded_qubits : list of AnyTransmon, optional
        Qubits excluded from the experiment (e.g., not at sweep spot).

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    # Include all qubits in grid creation so excluded qubits have subplot positions
    all_qubits = list(qubits) + (excluded_qubits or [])
    all_grid_locs = [q.grid_location for q in all_qubits]
    included_names = {q.name for q in qubits}
    
    grid = QubitGrid(ds, [q.grid_location for q in qubits]) if not excluded_qubits else \
           _create_grid_with_all_locations(ds, all_grid_locs, [q.name for q in all_qubits])
    
    n_subplots = len(grid.fig.axes)
    fs_x, fs_text = _excluded_qubit_font_sizes(n_subplots)
    
    qubit_dict = {q.name: q for q in qubits}
    for ax, qubit in grid_iter(grid):
        if qubit["qubit"] not in included_names:
            _mark_excluded_qubit(ax, qubit["qubit"], fs_x, fs_text)
            continue
        qubit_obj = qubit_dict.get(qubit["qubit"])
        fit_qubit = fits.sel(qubit=qubit["qubit"])
        
        # Calculate x-axis limits for this specific qubit
        flux_values = _get_flux_values(ds, qubit["qubit"], fit_qubit.sel(fit_vals="f").fit_results.flux_bias)
        flux_min, flux_max = float(np.min(flux_values)), float(np.max(flux_values))
        
        # Extend limits to include target_offset and flux_offset for this qubit
        # Show if fit succeeded (even if LO limit exceeded)
        if _should_show_fit(fit_qubit):
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
        Line2D([0], [0], color='magenta', marker='x', linestyle='', markersize=6, markeredgewidth=2, label='LO limit exceeded'),
    ]
    if excluded_qubits:
        legend_handles.append(Line2D([0], [0], color='green', marker='x', linestyle='', markersize=6, 
                                     markeredgewidth=2, label='Excluded: qubit set outside sweetspot'))
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
    state = ds.sel(qubit=qubit["qubit"]).state
    flux_values = _get_flux_values(ds, qubit["qubit"], state.flux_bias)
    
    # Plot using pcolormesh with correct flux values
    ax.pcolormesh(state.idle_times.values, flux_values, state.values, shading='auto')
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Idle_time (ns)")
    ax.set_ylabel("Flux (V)")

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
        The qubit object for accessing xy.extras["target_detuning_from_sweet_spot"].
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
    flux_values = _get_flux_values(ds, qubit["qubit"], frequency.flux_bias)
    
    # Plot data points (convert GHz to MHz and subtract detuning)
    freq_mhz = frequency.values * 1e3 - detuning
    ax.plot(flux_values, freq_mhz, linestyle="", marker=".", label="Data", color="blue")
    
    # Plot parabolic fit and calculate offsets if fit succeeded (even if LO limit exceeded)
    if _should_show_fit(fit):
        # Cache offset values to avoid repeated access
        flux_offset_val = float(fit.flux_offset.values)  # V
        freq_offset_val = float(fit.freq_offset.values)  # Hz
        
        # Reconstruct polynomial coefficients from stored fit parameters
        quad_term = float(fit.quad_term.values)  # GHz/V^2
        c2 = -quad_term  # GHz/V^2
        c1 = -2 * flux_offset_val * c2  # GHz/V
        c0 = freq_offset_val / 1e9 - c2 * flux_offset_val**2 - c1 * flux_offset_val  # GHz
        
        # Determine flux range for plotting parabola
        data_flux_min = float(np.min(flux_values))
        data_flux_max = float(np.max(flux_values))
        
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
            "target_detuning_from_sweet_spot" in qubit_obj.xy.extras):
            target_detuning = qubit_obj.xy.extras["target_detuning_from_sweet_spot"]
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
    
    # Mark failed fits or LO limit exceeded
    if outcomes and outcomes.get(qubit["qubit"]) == "failed":
        # Check if LO limit exceeded (fit succeeded but frequency out of range)
        if _is_lo_limit_exceeded(fit):
            ax.scatter([1.0], [1.0], marker='x', s=100, c='magenta', linewidths=2, 
                      transform=ax.transAxes, clip_on=False, zorder=10)
        else:
            ax.scatter([1.0], [1.0], marker='x', s=100, c='red', linewidths=2, 
                      transform=ax.transAxes, clip_on=False, zorder=10)
