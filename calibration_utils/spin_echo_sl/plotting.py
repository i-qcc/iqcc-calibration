from typing import List
import xarray as xr
from matplotlib.axes import Axes

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, skip_Q: bool = False):
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
    skip_Q : bool, optional
        If True, skip plotting Q quadrature. Default is False.

    Returns
    -------
    Figure or tuple of Figures
        The matplotlib figure object(s) containing the plots.
        Returns a single figure for state discrimination, or a tuple of (fig_I, fig_Q) for I, Q mode.
        If skip_Q is True, returns only fig_I.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    - When using I, Q mode, separate figures are created for I and Q (unless skip_Q is True).
    """
    # Check if we're using state discrimination or I, Q mode
    if "state" in ds.data_vars:
        # State discrimination mode - single figure
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), quadrature="state")

        grid.fig.suptitle("T2 SL with fit")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        return grid.fig
    else:
        # I, Q mode - create separate figures for I and Q
        # Figure for I
        grid_I = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid_I):
            plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), quadrature="I")

        grid_I.fig.suptitle("T2 SL with fit - I")
        grid_I.fig.set_size_inches(15, 9)
        grid_I.fig.tight_layout()
        
        # Figure for Q (only if skip_Q is False)
        if skip_Q:
            return grid_I.fig
        else:
            grid_Q = QubitGrid(ds, [q.grid_location for q in qubits])
            for ax, qubit in grid_iter(grid_Q):
                plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), quadrature="Q")

            grid_Q.fig.suptitle("T2 SL - Q")
            grid_Q.fig.set_size_inches(15, 9)
            grid_Q.fig.tight_layout()
            
            return grid_I.fig, grid_Q.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None, quadrature: str = None):
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
    quadrature : str, optional
        Which quadrature to plot: "state", "I", or "Q". If None, auto-detects from fit.

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    # Auto-detect quadrature if not specified
    if quadrature is None:
        if "state" in ds.data_vars:
            quadrature = "state"
        elif "fit_data_Q" in fit.data_vars:
            # If we have Q fit data, default to I (caller should specify)
            quadrature = "I"
        else:
            quadrature = "I"
    
    # Select appropriate fit data
    if quadrature == "state":
        fit_data = fit.fit_data
        fitted = decay_exp(
            ds.idle_time,
            fit_data.sel(fit_vals="a"),
            fit_data.sel(fit_vals="offset"),
            fit_data.sel(fit_vals="decay"),
        )
        ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
        if fitted is not None:
            ax.plot(ds.idle_time, fitted, "r--")
        ax.set_ylabel("State")
    elif quadrature == "I":
        fit_data = fit.fit_data_I if "fit_data_I" in fit.data_vars else fit.fit_data
        fitted = decay_exp(
            ds.idle_time,
            fit_data.sel(fit_vals="a"),
            fit_data.sel(fit_vals="offset"),
            fit_data.sel(fit_vals="decay"),
        )
        (ds.sel(qubit=qubit["qubit"]).I * 1e3).plot(ax=ax)
        if fitted is not None:
            ax.plot(ds.idle_time, fitted * 1e3, "r--")
        ax.set_ylabel("Trans. amp. I [mV]")
    elif quadrature == "Q":
        # Plot Q without fitting
        (ds.sel(qubit=qubit["qubit"]).Q * 1e3).plot(ax=ax)
        ax.set_ylabel("Trans. amp. Q [mV]")
    else:
        raise RuntimeError(f"Unknown quadrature: {quadrature}. Must be 'state', 'I', or 'Q'.")

    ax.set_title(qubit["qubit"])
    ax.set_xlabel("spin locking time (ns)")
    
    # Only show fit text for state and I, not for Q
    if quadrature != "Q" and fit is not None:
        ax.text(
            0.1,
            0.9,
            f'T2SL = {fit["T2_SL"].values*1e-3:.1f} ± {fit["T2_SL_error"].values*1e-3:.1f} µs',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )
