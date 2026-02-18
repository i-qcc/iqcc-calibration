import re
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


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


def _create_grid_with_all_locations(ds: xr.Dataset, grid_locations: List, qubit_names: List[str], size: int = 3):
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
    
    figure, all_axes = plt.subplots(*shape, figsize=(shape[1] * size, shape[0] * size), squeeze=False)
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


def _get_qubit_data(fit: xr.Dataset, qubit_name: str, data_var: str) -> Optional[xr.DataArray]:
    """Helper function to get qubit data using sel or isel."""
    try:
        return fit[data_var].sel(qubit=qubit_name)
    except (KeyError, ValueError):
        try:
            qubit_idx = next(i for i, q in enumerate(fit.qubit.values) if q == qubit_name)
            return fit[data_var].isel(qubit=qubit_idx)
        except (StopIteration, KeyError, ValueError):
            return None


def _get_qubit_value(fit: xr.Dataset, qubit_name: str, coord_or_var: str) -> Optional[float]:
    """Helper function to get a scalar value for a qubit."""
    try:
        if coord_or_var in fit.coords:
            return float(fit.coords[coord_or_var].sel(qubit=qubit_name).values)
        elif coord_or_var in fit.data_vars:
            return float(fit.data_vars[coord_or_var].sel(qubit=qubit_name).values)
    except (KeyError, ValueError):
        try:
            qubit_idx = next(i for i, q in enumerate(fit.qubit.values) if q == qubit_name)
            if coord_or_var in fit.coords:
                return float(fit.coords[coord_or_var].isel(qubit=qubit_idx).values)
            elif coord_or_var in fit.data_vars:
                return float(fit.data_vars[coord_or_var].isel(qubit=qubit_idx).values)
        except (StopIteration, KeyError, ValueError):
            return None
    return None


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset,
                           excluded_qubits: Optional[List[AnyTransmon]] = None):
    """
    Plots the raw data with fitted curves for the given qubits.

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
           _create_grid_with_all_locations(ds, all_grid_locs, [q.name for q in all_qubits], size=6)
    
    n_subplots = len(grid.fig.axes)
    fs_x, fs_text = _excluded_qubit_font_sizes(n_subplots)
    
    qubit_dict = {q.name: q for q in qubits}
    primary_axes = []
    
    for ax, qubit in grid_iter(grid):
        if qubit["qubit"] not in included_names:
            _mark_excluded_qubit(ax, qubit["qubit"], fs_x, fs_text)
            continue
        primary_axes.append(ax)
        plot_individual_raw_data_with_fit(ax, ds, qubit, fits, qubit_dict.get(qubit["qubit"]))

    # Create a single global legend for the entire figure
    handles, labels = [], []
    seen_labels = set()
    
    if primary_axes:
        h, l = primary_axes[0].get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in seen_labels:
                handles.append(handle)
                labels.append(label)
                seen_labels.add(label)
    
    # Add color scale indicators and excluded qubit marker
    handles.append(Patch(facecolor='#fde725', edgecolor='#fde725'))
    labels.append('high')
    handles.append(Patch(facecolor='#440154', edgecolor='#440154'))
    labels.append('low')
    if excluded_qubits:
        handles.append(Line2D([0], [0], color='green', marker='x', linestyle='', markersize=6, markeredgewidth=2))
        labels.append('Excluded: qubit set outside sweetspot')
    
    grid.fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                    ncol=min(len(handles), 7), frameon=True, fontsize=12)

    grid.fig.suptitle("Qubit spectroscopy vs flux")
    grid.fig.set_size_inches(15, 18)
    grid.fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    return grid.fig


def plot_individual_raw_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None, qubit_obj: AnyTransmon = None) -> bool:
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
    qubit_obj : AnyTransmon, optional
        The qubit object for accessing additional properties.

    Returns
    -------
    bool
        True if target offset was plotted, False otherwise.

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    - Individual legends are not shown; a global legend is created at the figure level.
    """
    target_offset_plotted = False  # Initialize return value
    
    # Store original data range before plotting
    original_flux_min = float(ds.flux_bias.min().values)
    original_flux_max = float(ds.flux_bias.max().values)
    
    ax2 = ax.twiny()
    # Plot using the attenuated current x-axis
    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax2,
        add_colorbar=False,
        x="attenuated_current",
        y="freq_GHz",
        robust=True,
    )
    ax2.set_xlabel("Current (A)")
    ax2.set_ylabel("Freq (GHz)")
    ax2.set_title("")
    # Move ax2 behind ax
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    # Plot using the flux x-axis
    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax, add_colorbar=False, x="flux_bias", y="freq_GHz", robust=True
    )
    
    # Clip heatmap to original data range to prevent stretching when limits are extended
    ylim = ax.get_ylim()
    clip_bbox = Bbox([[original_flux_min, ylim[0]], [original_flux_max, ylim[1]]])
    clip_vertices = np.array([
        [original_flux_min, ylim[0]], [original_flux_max, ylim[0]],
        [original_flux_max, ylim[1]], [original_flux_min, ylim[1]],
        [original_flux_min, ylim[0]]
    ])
    clip_path = Path(clip_vertices, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    
    # Clip images and collections on both axes
    for img in list(ax.images) + list(ax2.images):
        img.set_clip_box(clip_bbox)
        img.set_clip_on(True)
        if img in ax.images:  # Only set extent for ax images (flux_bias axis)
            extent = img.get_extent()
            img.set_extent([original_flux_min, original_flux_max, extent[2], extent[3]])
    
    for collection in list(ax.collections) + list(ax2.collections):
        collection.set_clip_path(clip_path, transform=ax.transData)
        collection.set_clip_on(True)
    
    ax.set_xlim(original_flux_min, original_flux_max)
    
    if fit is not None:
        qubit_name = qubit["qubit"]
        freq_ref = ds.full_freq.sel(qubit=qubit_name).values[0] - ds.detuning.values[0]
        
        # Get fit parameters and offset values
        success = bool(_get_qubit_value(fit, qubit_name, "success") or False) if "success" in fit.coords else False
        idle_offset_val = _get_qubit_value(fit, qubit_name, "idle_offset")
        target_offset_val = _get_qubit_value(fit, qubit_name, "target_offset")
        
        # Get fit coefficients for plotting
        fit_coeffs = None
        if "polyfit_coefficients" in fit.data_vars:
            coeff_data = _get_qubit_data(fit, qubit_name, "polyfit_coefficients")
            if coeff_data is not None:
                try:
                    c0, c1, c2 = [float(coeff_data.sel(degree=d).values) for d in [0, 1, 2]]
                    if not (np.isnan(c0) or np.isnan(c1) or np.isnan(c2)):
                        fit_coeffs = (c0, c1, c2)
                except (KeyError, ValueError):
                    pass
        elif "fitted_parabola" in fit.data_vars:
            fitted_parabola = _get_qubit_data(fit, qubit_name, "fitted_parabola")
            if fitted_parabola is not None:
                fit_coeffs = ("parabola", fitted_parabola)
        
        # Plot peak positions if available
        if "peak_freq" in fit.data_vars:
            peak_freq = _get_qubit_data(fit, qubit_name, "peak_freq")
            if peak_freq is not None:
                peak_freq_filtered = peak_freq.where(~np.isnan(peak_freq))
                if np.any(~np.isnan(peak_freq_filtered.values)):
                    peak_freq_GHz = (peak_freq_filtered + freq_ref) / 1e9
                    peak_freq_GHz.plot(ax=ax, x="flux_bias", ls="", marker=".", color="magenta", ms=16, label="peaks (filtered)")
        
        # Plot idle offset line and sweet spot (only if within ±0.5 V to avoid messing up scale)
        idle_offset_plotted = False
        flux_limit = 0.5  # V - don't show vertical lines outside this range
        
        if idle_offset_val is not None and not np.isnan(idle_offset_val):
            flux_shift = float(idle_offset_val)
            
            # Only plot if within ±0.5 V
            if abs(flux_shift) <= flux_limit:
                # Plot sweet spot marker if freq_shift is available
                if success:
                    freq_shift_val = _get_qubit_value(fit, qubit_name, "freq_shift")
                    if freq_shift_val is not None and not np.isnan(freq_shift_val):
                        freq_shift_GHz = (freq_shift_val + freq_ref) / 1e9
                        ax.plot(flux_shift, freq_shift_GHz, "r*", markersize=10, label="sweet spot")
                
                # Plot idle offset vertical line
                ax.axvline(flux_shift, linestyle="-", linewidth=4, color="r", alpha=0.5, label="idle offset")
                idle_offset_plotted = True
            
            # Plot target offset line if applicable (only if within ±0.5 V)
            if (qubit_obj is not None and hasattr(qubit_obj, 'xy') and 
                "target_detuning_from_sweet_spot" in qubit_obj.xy.extras):
                target_detuning = qubit_obj.xy.extras.get("target_detuning_from_sweet_spot", 0)
                if abs(target_detuning) > 1e-6 and target_offset_val is not None and not np.isnan(target_offset_val):
                    if abs(float(target_offset_val)) <= flux_limit:
                        ax.axvline(float(target_offset_val), linestyle="-", linewidth=4, 
                                  color="orange", label="target offset", alpha=0.7)
                        target_offset_plotted = True
        
        # Extend plot limits if offset lines are outside original range (only for offsets within ±0.5 V)
        xmin, xmax = original_flux_min, original_flux_max
        offset_values = []
        if idle_offset_val is not None and not np.isnan(idle_offset_val) and abs(float(idle_offset_val)) <= flux_limit:
            offset_values.append(float(idle_offset_val))
        if target_offset_val is not None and not np.isnan(target_offset_val) and abs(float(target_offset_val)) <= flux_limit:
            offset_values.append(float(target_offset_val))
        
        if offset_values:
            margin = (xmax - xmin) * 0.15
            for offset_val in offset_values:
                if offset_val < xmin:
                    xmin = min(xmin, offset_val - margin)
                if offset_val > xmax:
                    xmax = max(xmax, offset_val + margin)
            
            if xmin < original_flux_min or xmax > original_flux_max:
                ax.set_xlim(xmin, xmax)
                # Re-apply clipping after extending limits
                ylim = ax.get_ylim()
                clip_bbox = Bbox([[original_flux_min, ylim[0]], [original_flux_max, ylim[1]]])
                clip_vertices = np.array([
                    [original_flux_min, ylim[0]], [original_flux_max, ylim[0]],
                    [original_flux_max, ylim[1]], [original_flux_min, ylim[1]],
                    [original_flux_min, ylim[0]]
                ])
                clip_path = Path(clip_vertices, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
                
                for img in list(ax.images) + list(ax2.images):
                    img.set_clip_box(clip_bbox)
                    img.set_clip_on(True)
                    if img in ax.images:
                        extent = img.get_extent()
                        img.set_extent([original_flux_min, original_flux_max, extent[2], extent[3]])
                
                for collection in list(ax.collections) + list(ax2.collections):
                    collection.set_clip_path(clip_path, transform=ax.transData)
                    collection.set_clip_on(True)
        
        # Plot fit line using final axis limits
        if fit_coeffs is not None:
            final_xmin, final_xmax = ax.get_xlim()
            flux_bias_fine = np.linspace(final_xmin, final_xmax, 100)
            
            if isinstance(fit_coeffs[0], str) and fit_coeffs[0] == "parabola":
                # Interpolated parabola - only plot within original data range
                fitted_parabola = fit_coeffs[1]
                fitted_freq_GHz = (fitted_parabola + freq_ref) / 1e9
                original_flux_array = fitted_freq_GHz.flux_bias.values
                flux_bias_fine_clipped = np.clip(flux_bias_fine, original_flux_min, original_flux_max)
                fitted_freq_GHz_fine = np.interp(flux_bias_fine_clipped, original_flux_array, fitted_freq_GHz.values)
                mask = (flux_bias_fine >= original_flux_min) & (flux_bias_fine <= original_flux_max)
                if np.any(mask):
                    ax.plot(flux_bias_fine[mask], fitted_freq_GHz_fine[mask], 
                           linewidth=2, ls="--", color="r", label="parabolic fit", zorder=5)
            else:
                # Polynomial coefficients - can extend to full range
                c0, c1, c2 = fit_coeffs
                fitted_freq_GHz_fine = (c2 * flux_bias_fine**2 + c1 * flux_bias_fine + c0 + freq_ref) / 1e9
                ax.plot(flux_bias_fine, fitted_freq_GHz_fine, linewidth=2, ls="--", 
                       color="r", label="parabolic fit", zorder=5)
    
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Flux (V)")
    ax.set_ylabel("Freq (GHz)")
    # Don't show individual legends - a global legend will be created at the figure level
    
    # Add red X marker in top-right corner if fit failed
    if fit is not None:
        qubit_name = qubit["qubit"]
        success = bool(_get_qubit_value(fit, qubit_name, "success") or False) if "success" in fit.coords else False
        if not success:
            ax.text(0.95, 0.95, "✗", transform=ax.transAxes, fontsize=24, 
                   color='red', fontweight='bold', ha='right', va='top', zorder=10)
    
    return target_offset_plotted
