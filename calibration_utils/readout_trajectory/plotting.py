from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

def plot_adc_trace(ds: xr.Dataset, qubits: List[AnyTransmon]):
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
        plot_individual_single_adc_stream(ax, ds, qubit)

    grid.fig.suptitle("Single run")
    grid.fig.set_size_inches(15, 9)
    grid.fig.legend(loc="upper right", ncols=4, bbox_to_anchor=(0.5, 1.35))
    grid.fig.tight_layout()
    return grid.fig

def plot_individual_single_adc_stream(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    # ds.loc[qubit].adc_single_runI.plot(ax=ax, x="readout_time", label="I", color="b")
    # ds.loc[qubit].adc_single_runQ.plot(ax=ax, x="readout_time", label="Q", color="r")
    ds.adc_single_runI.plot(ax=ax, x="readout_time", label="I", color="b")
    ds.adc_single_runQ.plot(ax=ax, x="readout_time", label="Q", color="r")
    #ax.axvline(fit.delay, color="k", linestyle="--", label="TOF")
    # ax.axhline(ds.loc[qubit].offsets_I, color="b", linestyle="--")
    # ax.axhline(ds.loc[qubit].offsets_Q, color="r", linestyle="--")
    ax.fill_between(
        range(ds.sizes["readout_time"]),
        -0.5,
        0.5,
        color="grey",
        alpha=0.2,
        label="ADC Range",
    )
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Readout amplitude [mV]")
    ax.set_title(qubit["qubit"])
    ax.legend()


def plot_single_run_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
        plot_individual_single_run_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Single run")
    grid.fig.set_size_inches(15, 9)
    grid.fig.legend(loc="upper right", ncols=4, bbox_to_anchor=(0.5, 1.35))
    grid.fig.tight_layout()
    return grid.fig





def plot_averaged_run_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
        plot_individual_averaged_run_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Averaged run")
    grid.fig.set_size_inches(15, 9)
    grid.fig.legend(loc="upper right", ncols=4, bbox_to_anchor=(0.5, 1.35))
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_single_run_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    ds.loc[qubit].adc_single_runI.plot(ax=ax, x="readout_time", label="I", color="b")
    ds.loc[qubit].adc_single_runQ.plot(ax=ax, x="readout_time", label="Q", color="r")
    ax.axvline(fit.delay, color="k", linestyle="--", label="TOF")
    # ax.axhline(ds.loc[qubit].offsets_I, color="b", linestyle="--")
    # ax.axhline(ds.loc[qubit].offsets_Q, color="r", linestyle="--")
    ax.fill_between(
        range(ds.sizes["readout_time"]),
        -0.5,
        0.5,
        color="grey",
        alpha=0.2,
        label="ADC Range",
    )
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Readout amplitude [mV]")
    ax.set_title(qubit["qubit"])


def plot_individual_averaged_run_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    ds.loc[qubit].adcI.plot(ax=ax, x="readout_time", label="I", color="b")
    ds.loc[qubit].adcQ.plot(ax=ax, x="readout_time", label="Q", color="r")
    ax.axvline(fit.delay, color="k", linestyle="--", label="TOF")
    # ax.axhline(ds.loc[qubit].offsets_I, color="b", linestyle="--")
    # ax.axhline(ds.loc[qubit].offsets_Q, color="r", linestyle="--")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Readout amplitude [mV]")
    ax.set_title(qubit["qubit"])


def plot_readout_trajectory(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    parameters,
    ds_fit: Optional[xr.Dataset] = None,
) -> Figure:
    """
    Plot the readout trajectory in the IQ plane for ground and excited states.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data with Ie, Qe, Ig, Qg.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    parameters
        Node parameters containing qubit index, square_length, zero_length, etc.
    ds_fit : xr.Dataset, optional
        The dataset containing fit results (default is None).

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    n_avg = parameters.num_shots
    square_length = parameters.square_length
    zero_length = parameters.zero_length
    segment_length = 10  # clock cycles
    W = segment_length * 4  # Convert to nanoseconds
    square_num = int(square_length / W)

    # Extract I and Q arrays from the dataset
    # XarrayDataFetcher creates datasets with qubit dimension
    # Data shape is typically (n_runs, readout_time, qubit) or (n_runs, readout_time)
    if "qubit" in ds["Ie"].dims and len(ds["qubit"]) > 0:
        # Multiple qubits - plot first qubit (can be extended to plot all in a grid)
        qubit_names = ds["qubit"].values
        qubit_idx = 0
        # Select first qubit
        Ie = ds["Ie"].isel(qubit=qubit_idx).values if "n_runs" in ds["Ie"].dims else ds["Ie"].values
        Qe = ds["Qe"].isel(qubit=qubit_idx).values if "n_runs" in ds["Qe"].dims else ds["Qe"].values
        Ig = ds["Ig"].isel(qubit=qubit_idx).values if "n_runs" in ds["Ig"].dims else ds["Ig"].values
        Qg = ds["Qg"].isel(qubit=qubit_idx).values if "n_runs" in ds["Qg"].dims else ds["Qg"].values
        qubit_name = qubit_names[qubit_idx] if len(qubit_names) > qubit_idx else f"Q{qubit_idx+1}"
    else:
        # Single qubit case or no qubit dimension
        Ie = ds["Ie"].values
        Qe = ds["Qe"].values
        Ig = ds["Ig"].values
        Qg = ds["Qg"].values
        qubit_name = qubits[0].name if qubits else "Q1"
    
    # Handle different data shapes - ensure we have (n_runs, readout_time) shape
    if len(Ie.shape) > 2:
        # If shape is (n_runs, readout_time, qubit), take first qubit
        Ie = Ie[:, :, 0] if Ie.shape[2] > 0 else Ie[:, :]
        Qe = Qe[:, :, 0] if Qe.shape[2] > 0 else Qe[:, :]
        Ig = Ig[:, :, 0] if Ig.shape[2] > 0 else Ig[:, :]
        Qg = Qg[:, :, 0] if Qg.shape[2] > 0 else Qg[:, :]
    elif len(Ie.shape) == 1:
        # If shape is (readout_time,), add n_runs dimension
        Ie = Ie[np.newaxis, :]
        Qe = Qe[np.newaxis, :]
        Ig = Ig[np.newaxis, :]
        Qg = Qg[np.newaxis, :]

    # Calculate statistics
    Ie_var = np.std(Ie, axis=0)
    Qe_var = np.std(Qe, axis=0)
    Ig_var = np.std(Ig, axis=0)
    Qg_var = np.std(Qg, axis=0)

    # Average over shots
    Ie = np.mean(Ie, axis=0)
    Qe = np.mean(Qe, axis=0)
    Ig = np.mean(Ig, axis=0)
    Qg = np.mean(Qg, axis=0)

    # Calculate separation
    diff = (Ie - Ig) ** 2 + (Qe - Qg) ** 2
    t = np.arange(0, len(diff) * W, W)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))

    # Plot 1: Separation (linear scale)
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(t, diff, label="Separation", color="#146BEE")
    plt.ylim(diff.min(), 1.2 * diff.max())
    plt.legend()
    plt.xlabel("Time [ns]")
    plt.ylabel("Separation")
    plt.title("State Separation")
    plt.grid(True)

    # Plot 2: Separation (log scale)
    ax2 = plt.subplot(2, 2, 2)
    eps = 1e-12
    diff_log = diff + eps
    plt.plot(t, diff_log, label="Separation", color="#146BEE")
    plt.yscale("log")
    plt.ylim(diff_log.min(), 1.2 * diff_log.max())
    plt.xlabel("Time [ns]")
    plt.ylabel("Separation (log scale)")
    plt.title("State Separation (log scale)")
    plt.legend()
    plt.grid(True, which="both")

    # Plot 3: IQ plane with error bars
    ax3 = plt.subplot(2, 2, 3)
    plt.errorbar(
        Ie,
        Qe,
        xerr=Ie_var**2,
        yerr=Qe_var**2,
        fmt="none",
        label="Excited",
        color="#EE3183",
        capsize=2,
        elinewidth=1,
    )
    plt.errorbar(
        Ig,
        Qg,
        xerr=Ig_var**2,
        yerr=Qg_var**2,
        fmt="none",
        label="Ground",
        color="#146BEE",
        capsize=2,
        elinewidth=1,
    )
    plt.scatter(Ie, Qe, s=8, label="Excited", color="#EE3183")
    plt.scatter(Ig, Qg, s=8, label="Ground", color="#146BEE")
    plt.scatter(Ie[0], Qe[0], color="green", s=80, zorder=3, label="Start")
    plt.scatter(Ig[0], Qg[0], color="green", s=80, zorder=3)
    plt.scatter(Ie[-1], Qe[-1], color="black", s=80, zorder=3, label="End")
    plt.scatter(Ig[-1], Qg[-1], color="black", s=80, zorder=3)
    plt.legend()
    plt.xlabel("I [V]")
    plt.ylabel("Q [V]")
    plt.title(f"IQ Plane - {qubit_name}, shots: {n_avg}, total: {square_length+zero_length}ns")
    plt.grid(True)
    plt.axis("equal")

    # Plot 4: IQ plane with trajectory lines
    ax4 = plt.subplot(2, 2, 4)
    # Apply drift correction for square section
    diff_Ig = Ig[-1] - Ig[0]
    diff_Qg = Qg[-1] - Qg[0]
    Ie_corrected = Ie.copy()
    Ig_corrected = Ig.copy()
    Qe_corrected = Qe.copy()
    Qg_corrected = Qg.copy()
    Ie_corrected[:square_num] = Ie[:square_num] + diff_Ig
    Ig_corrected[:square_num] = Ig[:square_num] + diff_Ig
    Qe_corrected[:square_num] = Qe[:square_num] + diff_Qg
    Qg_corrected[:square_num] = Qg[:square_num] + diff_Qg

    plt.errorbar(
        Ie_corrected,
        Qe_corrected,
        xerr=np.abs(Ie_var) ** 2,
        yerr=np.abs(Qe_var) ** 2,
        fmt="-o",
        label="Excited",
        color="#EE3183",
        capsize=2,
        elinewidth=1,
        markersize=4,
    )
    plt.errorbar(
        Ig_corrected,
        Qg_corrected,
        xerr=np.abs(Ig_var) ** 2,
        yerr=np.abs(Qg_var) ** 2,
        fmt="-o",
        label="Ground",
        color="#146BEE",
        capsize=2,
        elinewidth=1,
        markersize=4,
    )
    plt.scatter(Ie_corrected[0], Qe_corrected[0], color="green", s=80, zorder=3, label="Start")
    plt.scatter(Ig_corrected[0], Qg_corrected[0], color="green", s=80, zorder=3)
    plt.scatter(Ie_corrected[-1], Qe_corrected[-1], color="black", s=80, zorder=3, label="End")
    plt.scatter(Ig_corrected[-1], Qg_corrected[-1], color="black", s=80, zorder=3)
    plt.legend()
    plt.xlabel("I [V]")
    plt.ylabel("Q [V]")
    plt.title(
        f"IQ Trajectory - {qubit_name}, shots: {n_avg}, square: {square_length}ns, zero: {zero_length}ns, slice: {W}ns"
    )
    plt.grid(True)
    plt.axis("equal")

    plt.tight_layout()
    return fig
