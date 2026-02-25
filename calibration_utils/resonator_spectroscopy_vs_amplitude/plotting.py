from typing import List
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset,
    clip_left_mhz: float = None,
    clip_right_mhz: float = 1.0,
):
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
    clip_left_mhz : float, optional
        Left (negative detuning) clipping bound in MHz. If provided, a vertical line is drawn.
    clip_right_mhz : float, optional
        Right (positive detuning) clipping bound in MHz. If provided, a vertical line is drawn.

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
    legend_entries = {}
    for ax, qubit in grid_iter(grid):
        ax2 = plot_individual_raw_data_with_fit(
            ax, ds, qubit, fits.sel(qubit=qubit["qubit"]),
            clip_left_mhz=clip_left_mhz, clip_right_mhz=clip_right_mhz,
        )
        for handle, label in zip(*ax2.get_legend_handles_labels()):
            legend_entries.setdefault(label, handle)

    if legend_entries:
        grid.fig.legend(legend_entries.values(), legend_entries.keys(), loc="upper right", fontsize="small")

    grid.fig.suptitle("Resonator spectroscopy vs power")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    fit: xr.Dataset = None,
    clip_left_mhz: float = None,
    clip_right_mhz: float = 1.0,
) -> Axes:
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
    clip_left_mhz : float, optional
        Left (negative detuning) clipping bound in MHz.
    clip_right_mhz : float, optional
        Right (positive detuning) clipping bound in MHz.

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax,
        add_colorbar=False,
        x="freq_GHz",
        y="power",
        linewidth=0.5,
    )
    ax.set_ylabel("Power (dBm)")
    ax2 = ax.twiny()
    ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].IQ_abs_norm.plot(
        ax=ax2, add_colorbar=False, x="detuning_MHz", y="power", robust=True
    )
    ax2.set_xlabel("Detuning [MHz]")
    # Plot the raw resonance frequency tracking as points with a connecting line
    rr_vals = fit.rr_min_response * 1e-6
    valid = ~np.isnan(rr_vals)
    ax2.plot(
        rr_vals.where(valid),
        fit.power.where(valid),
        color="orange",
        linewidth=0.8,
        alpha=0.5,
    )
    ax2.scatter(
        rr_vals.where(valid),
        fit.power.where(valid),
        color="orange",
        s=4,
        zorder=5,
        label="Dip tracking",
    )
    # Overlay the arctan fit line
    if "rr_min_response_arctan_fit" in fit:
        arctan_vals = fit.rr_min_response_arctan_fit * 1e-6
        valid_arctan = ~np.isnan(arctan_vals)
        ax2.plot(
            arctan_vals.where(valid_arctan),
            fit.power.where(valid_arctan),
            color="cyan",
            linewidth=2,
            label="Arctan fit",
        )
    # Plot where the optimum readout power was found
    if fit.success:
        ax2.axhline(
            y=fit.optimal_power,
            color="g",
            linestyle="-",
            label="Optimal power",
        )
        ax2.axvline(
            x=fit.freq_shift * 1e-6,
            color="blue",
            linestyle="--",
            label="Freq. shift",
        )
    if clip_left_mhz is not None:
        ax2.axvline(x=-clip_left_mhz, color="red", linestyle=":", linewidth=3, alpha=0.7, label="Clip bounds")
    if clip_right_mhz is not None:
        ax2.axvline(x=clip_right_mhz, color="red", linestyle=":", linewidth=3, alpha=0.7)
    return ax2
