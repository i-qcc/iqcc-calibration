from typing import List, Tuple
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import to_rgb
import colorsys

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def create_color_gradient(base_color: str, n_points: int, brightness_range: Tuple[float, float] = (0.3, 1.0)) -> np.ndarray:
    """
    Create a color gradient from dark to bright for a given base color.
    
    Parameters
    ----------
    base_color : str
        Base color in hex format (e.g., "#EE3183")
    n_points : int
        Number of points in the gradient
    brightness_range : Tuple[float, float]
        Range of brightness values (V in HSV) from dark to bright. Default (0.3, 1.0) provides
        good visibility for human eye while maintaining color distinction.
    
    Returns
    -------
    np.ndarray
        Array of RGB colors (n_points, 3) ranging from dark to bright
    """
    # Convert hex to RGB
    rgb = np.array(to_rgb(base_color))
    
    # Convert RGB to HSV using colorsys (values 0-1)
    hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    
    # Create brightness gradient from dark to bright
    brightness_values = np.linspace(brightness_range[0], brightness_range[1], n_points)
    
    # Create array of colors
    colors = np.zeros((n_points, 3))
    for i, brightness in enumerate(brightness_values):
        # Keep hue and saturation, vary brightness
        rgb_colored = colorsys.hsv_to_rgb(hsv[0], hsv[1], brightness)
        colors[i] = np.array(rgb_colored)
    
    return colors

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


def plot_readout_trajectories(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    square_length: int,
    zero_length: int,
    W: int,
    *,
    apply_correction: bool = False,
) -> Tuple[Figure, Figure, Figure]:
    """
    Plot all readout trajectory figures: difference plots and IQ trajectories.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : List[AnyTransmon]
        List of qubits to plot.
    square_length : int
        Square pulse length in nanoseconds.
    zero_length : int
        Zero pulse length in nanoseconds.
    W : int
        Slice width in nanoseconds.
    apply_correction : bool, optional
        Whether to apply offset correction to square pulse region (default is False).

    Returns
    -------
    Tuple[Figure, Figure, Figure]
        Tuple containing (fig_diff_log, fig_IQ_raw, fig_IQ_corrected).
    """
    # Filter qubits to only include those that have data in the dataset
    qubits_in_dataset = [q for q in qubits if q.name in ds.qubit.values]
    
    if not qubits_in_dataset:
        raise ValueError(f"No qubits from the provided list found in dataset. Dataset contains: {list(ds.qubit.values)}")
    
    # Plot difference (log scale) - using QubitGrid for multiple qubits
    fig_diff = plot_trajectory_difference(ds, qubits_in_dataset, square_length, zero_length, W, log_scale=True)
    
    # Plot IQ trajectories - using QubitGrid for multiple qubits
    fig_IQ_raw, fig_IQ = plot_iq_trajectories(
        ds, qubits_in_dataset, square_length, zero_length, W, apply_correction=apply_correction
    )
    
    return fig_diff, fig_IQ_raw, fig_IQ


def plot_trajectory_difference(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    square_length: int,
    zero_length: int,
    W: int,
    log_scale: bool = False,
) -> Figure:
    """
    Plot the difference between excited and ground state trajectories for multiple qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : List[AnyTransmon]
        List of qubits to plot.
    square_length : int
        Square pulse length in nanoseconds.
    zero_length : int
        Zero pulse length in nanoseconds.
    W : int
        Slice width in nanoseconds.
    log_scale : bool, optional
        Whether to use log scale for y-axis (default is False).

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_trajectory_difference(ax, ds, qubit, square_length, zero_length, W, log_scale=log_scale)
    
    title = "difference (log scale)" if log_scale else "difference"
    grid.fig.suptitle(title)
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_trajectory_difference(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    square_length: int,
    zero_length: int,
    W: int,
    log_scale: bool = False,
):
    """
    Plot the difference between excited and ground state trajectories for a single qubit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        Mapping to the qubit to plot.
    square_length : int
        Square pulse length in nanoseconds.
    zero_length : int
        Zero pulse length in nanoseconds.
    W : int
        Slice width in nanoseconds.
    log_scale : bool, optional
        Whether to use log scale for y-axis (default is False).
    """
    qubit_name = qubit["qubit"]
    
    # Check if qubit exists in dataset
    if qubit_name not in ds.qubit.values:
        ax.text(0.5, 0.5, f"No data for {qubit_name}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(qubit_name)
        return
    
    ds_qubit = ds.sel(qubit=qubit_name)
    
    # Extract I and Q arrays
    Ie = ds_qubit["Ie"].values
    Qe = ds_qubit["Qe"].values
    Ig = ds_qubit["Ig"].values
    Qg = ds_qubit["Qg"].values
    
    # Handle different data shapes
    if Ie.ndim == 3:
        Ie = Ie[:, :, 0]
        Qe = Qe[:, :, 0]
        Ig = Ig[:, :, 0]
        Qg = Qg[:, :, 0]
    
    # Calculate averages
    Ie = np.mean(Ie, axis=0)
    Qe = np.mean(Qe, axis=0)
    Ig = np.mean(Ig, axis=0)
    Qg = np.mean(Qg, axis=0)
    
    diff = (Ie - Ig) ** 2 + (Qe - Qg) ** 2

    if log_scale:
        # avoid log(0)
        eps = 1e-12
        diff = diff + eps

    t = np.arange(0, len(diff) * W, W)
    ax.plot(t, diff, label="diff")
    
    # Add vertical lines to indicate pulse boundaries
    y_min, y_max = ax.get_ylim()
    if log_scale:
        y_min = diff.min()
        y_max = diff.max() * 1.2
    
    # Vertical line at end of square pulse
    ax.axvline(x=square_length, color='r', linestyle='--', linewidth=1, alpha=0.7, label=f'square: {square_length}ns')
    # Vertical line at end of zero pulse (start of zero + square length)
    total_length = square_length + zero_length
    ax.axvline(x=total_length, color='g', linestyle='--', linewidth=1, alpha=0.7, label=f'zero: {zero_length}ns')
    
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("t [ns]")
    ax.set_ylabel("diff")
    duration = square_length + zero_length
    ax.set_title(f"{qubit_name}\nsquare length: {square_length}ns, zero length: {zero_length}ns, total: {duration}ns")
    ax.grid(True, which="both" if log_scale else "major")
    ax.legend(loc='best', fontsize=8)


def plot_iq_trajectories(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    square_length: int,
    zero_length: int,
    W: int,
    *,
    apply_correction: bool = False,
) -> Tuple[Figure, Figure]:
    """
    Plot IQ plane trajectories for ground and excited states for multiple qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : List[AnyTransmon]
        List of qubits to plot.
    square_length : int
        Square pulse length in nanoseconds.
    zero_length : int
        Zero pulse length in nanoseconds.
    W : int
        Slice width in nanoseconds.
    apply_correction : bool, optional
        Whether to apply offset correction to square pulse region (default is False).

    Returns
    -------
    Tuple[Figure, Figure]
        Tuple containing (raw_figure, corrected_figure) if apply_correction is True,
        otherwise (raw_figure, line_figure).
    """
    # Create grid for raw IQ trajectories
    grid_raw = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid_raw):
        plot_individual_iq_trajectories_raw(
            ax, ds, qubit, square_length, zero_length, W
        )
    
    grid_raw.fig.suptitle("IQ Plane (Raw)")
    grid_raw.fig.set_size_inches(15, 9)
    handles, labels = grid_raw.fig.axes[0].get_legend_handles_labels()
    if handles:
        grid_raw.fig.legend(handles, labels, loc="upper right", ncols=2)
    grid_raw.fig.tight_layout()
    
    if apply_correction:
        # Create grid for corrected IQ trajectories
        grid_corrected = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid_corrected):
            plot_individual_iq_trajectories_corrected(
                ax, ds, qubit, square_length, zero_length, W
            )
        
        grid_corrected.fig.suptitle("IQ Plane (Corrected)")
        grid_corrected.fig.set_size_inches(15, 9)
        handles, labels = grid_corrected.fig.axes[0].get_legend_handles_labels()
        if handles:
            grid_corrected.fig.legend(handles, labels, loc="upper right", ncols=2)
        grid_corrected.fig.tight_layout()
        
        return grid_raw.fig, grid_corrected.fig
    
    # Create grid for line IQ trajectories (when not applying correction)
    grid_line = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid_line):
        plot_individual_iq_trajectories_line(
            ax, ds, qubit, square_length, zero_length, W
        )
    
    grid_line.fig.suptitle("IQ Plane (Line)")
    grid_line.fig.set_size_inches(15, 9)
    handles, labels = grid_line.fig.axes[0].get_legend_handles_labels()
    if handles:
        grid_line.fig.legend(handles, labels, loc="upper right", ncols=2)
    grid_line.fig.tight_layout()
    
    return grid_raw.fig, grid_line.fig


def plot_individual_iq_trajectories_raw(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    square_length: int,
    zero_length: int,
    W: int,
):
    """
    Plot raw IQ plane trajectories for a single qubit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        Mapping to the qubit to plot.
    square_length : int
        Square pulse length in nanoseconds.
    zero_length : int
        Zero pulse length in nanoseconds.
    W : int
        Slice width in nanoseconds.
    """
    qubit_name = qubit["qubit"]
    
    # Check if qubit exists in dataset
    if qubit_name not in ds.qubit.values:
        ax.text(0.5, 0.5, f"No data for {qubit_name}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(qubit_name)
        return
    
    ds_qubit = ds.sel(qubit=qubit_name)
    
    # Extract I and Q arrays
    Ie = ds_qubit["Ie"].values
    Qe = ds_qubit["Qe"].values
    Ig = ds_qubit["Ig"].values
    Qg = ds_qubit["Qg"].values
    
    # Get number of shots from dataset dimension before processing
    num_shots = ds_qubit.sizes.get("n_runs", Ie.shape[0] if Ie.ndim > 1 else 1)
    
    # Handle different data shapes
    if Ie.ndim == 3:
        Ie = Ie[:, :, 0]
        Qe = Qe[:, :, 0]
        Ig = Ig[:, :, 0]
        Qg = Qg[:, :, 0]

    # Calculate variances
    Ie_var = np.std(Ie, axis=0)
    Qe_var = np.std(Qe, axis=0)
    Ig_var = np.std(Ig, axis=0)
    Qg_var = np.std(Qg, axis=0)

    # Average over shots
    Ie = np.mean(Ie, axis=0)
    Qe = np.mean(Qe, axis=0)
    Ig = np.mean(Ig, axis=0)
    Qg = np.mean(Qg, axis=0)

    # Create color gradients for time progression (dark to bright)
    n_slices = len(Ie)
    excited_colors = create_color_gradient("#EE3183", n_slices)
    ground_colors = create_color_gradient("#146BEE", n_slices)

    # Plot raw IQ trajectories with color gradient
    ax.errorbar(
        Ie,
        Qe,
        xerr=Ie_var**2,
        yerr=Qe_var**2,
        fmt="none",
        label="Excited",
        color="#EE3183",
        capsize=2,
        elinewidth=1,
        alpha=0.3,  # Make error bars more subtle
    )
    ax.errorbar(
        Ig,
        Qg,
        xerr=Ig_var**2,
        yerr=Qg_var**2,
        fmt="none",
        label="Ground",
        color="#146BEE",
        capsize=2,
        elinewidth=1,
        alpha=0.3,  # Make error bars more subtle
    )
    # Plot scatter with color gradient (dark to bright = early to late time)
    ax.scatter(Ie, Qe, s=8, label="Excited", c=excited_colors, zorder=2)
    ax.scatter(Ig, Qg, s=8, label="Ground", c=ground_colors, zorder=2)
    ax.scatter(Ie[0], Qe[0], color="green", s=80, zorder=3)
    ax.scatter(Ig[0], Qg[0], color="green", s=80, zorder=3)
    ax.scatter(Ie[-1], Qe[-1], color="black", s=80, zorder=3)
    ax.scatter(Ig[-1], Qg[-1], color="black", s=80, zorder=3)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    # Format x-axis ticks to prevent overlapping
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0), useMathText=True)
    ax.tick_params(axis='x', rotation=15)
    duration = square_length + zero_length
    ax.set_title(f"{qubit_name}, shots: {num_shots}\nsquare length: {duration}ns, zero length: {zero_length}ns\nSlice Width: {W}ns")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()


def plot_individual_iq_trajectories_corrected(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    square_length: int,
    zero_length: int,
    W: int,
):
    """
    Plot corrected IQ plane trajectories for a single qubit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        Mapping to the qubit to plot.
    square_length : int
        Square pulse length in nanoseconds.
    zero_length : int
        Zero pulse length in nanoseconds.
    W : int
        Slice width in nanoseconds.
    """
    qubit_name = qubit["qubit"]
    
    # Check if qubit exists in dataset
    if qubit_name not in ds.qubit.values:
        ax.text(0.5, 0.5, f"No data for {qubit_name}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(qubit_name)
        return
    
    ds_qubit = ds.sel(qubit=qubit_name)
    
    # Extract I and Q arrays
    Ie = ds_qubit["Ie"].values
    Qe = ds_qubit["Qe"].values
    Ig = ds_qubit["Ig"].values
    Qg = ds_qubit["Qg"].values
    
    # Get number of shots from dataset dimension before processing
    num_shots = ds_qubit.sizes.get("n_runs", Ie.shape[0] if Ie.ndim > 1 else 1)
    
    # Handle different data shapes
    if Ie.ndim == 3:
        Ie = Ie[:, :, 0]
        Qe = Qe[:, :, 0]
        Ig = Ig[:, :, 0]
        Qg = Qg[:, :, 0]

    # Calculate variances
    Ie_var = np.std(Ie, axis=0)
    Qe_var = np.std(Qe, axis=0)
    Ig_var = np.std(Ig, axis=0)
    Qg_var = np.std(Qg, axis=0)

    # Average over shots
    Ie = np.mean(Ie, axis=0)
    Qe = np.mean(Qe, axis=0)
    Ig = np.mean(Ig, axis=0)
    Qg = np.mean(Qg, axis=0)

    # Apply offset correction to square pulse region
    square_num = int(square_length / W)
    diff_Ig = Ig[-1] - Ig[0]
    diff_Qg = Qg[-1] - Qg[0]
    Ie_corr = Ie.copy()
    Qe_corr = Qe.copy()
    Ig_corr = Ig.copy()
    Qg_corr = Qg.copy()
    Ie_corr[:square_num] = Ie_corr[:square_num] + diff_Ig
    Ig_corr[:square_num] = Ig_corr[:square_num] + diff_Ig
    Qe_corr[:square_num] = Qe_corr[:square_num] + diff_Qg
    Qg_corr[:square_num] = Qg_corr[:square_num] + diff_Qg

    # Create color gradients for time progression (dark to bright)
    n_slices = len(Ie_corr)
    excited_colors = create_color_gradient("#EE3183", n_slices)
    ground_colors = create_color_gradient("#146BEE", n_slices)

    # Plot corrected trajectories with color gradient
    # Plot error bars with subtle colors
    ax.errorbar(
        Ie_corr,
        Qe_corr,
        xerr=np.abs(Ie_var) ** 2,
        yerr=np.abs(Qe_var) ** 2,
        fmt="none",
        label="Excited",
        color="#EE3183",
        capsize=2,
        elinewidth=1,
        alpha=0.3,
    )
    ax.errorbar(
        Ig_corr,
        Qg_corr,
        xerr=np.abs(Ig_var) ** 2,
        yerr=np.abs(Qg_var) ** 2,
        fmt="none",
        label="Ground",
        color="#146BEE",
        capsize=2,
        elinewidth=1,
        alpha=0.3,
    )
    # Plot lines and markers with color gradient
    for i in range(n_slices):
        if i == 0:
            # First point - plot with label for legend
            ax.plot([Ie_corr[i]], [Qe_corr[i]], "-o", label="Excited", color=excited_colors[i], markersize=4)
            ax.plot([Ig_corr[i]], [Qg_corr[i]], "-o", label="Ground", color=ground_colors[i], markersize=4)
        else:
            # Connect to previous point
            ax.plot([Ie_corr[i-1], Ie_corr[i]], [Qe_corr[i-1], Qe_corr[i]], "-", color=excited_colors[i-1], linewidth=1)
            ax.plot([Ig_corr[i-1], Ig_corr[i]], [Qg_corr[i-1], Qg_corr[i]], "-", color=ground_colors[i-1], linewidth=1)
            ax.plot([Ie_corr[i]], [Qe_corr[i]], "o", color=excited_colors[i], markersize=4)
            ax.plot([Ig_corr[i]], [Qg_corr[i]], "o", color=ground_colors[i], markersize=4)
    ax.scatter(Ie_corr[0], Qe_corr[0], color="green", s=80, zorder=3)
    ax.scatter(Ig_corr[0], Qg_corr[0], color="green", s=80, zorder=3)
    ax.scatter(Ie_corr[-1], Qe_corr[-1], color="black", s=80, zorder=3)
    ax.scatter(Ig_corr[-1], Qg_corr[-1], color="black", s=80, zorder=3)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    # Format x-axis ticks to prevent overlapping
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0), useMathText=True)
    ax.tick_params(axis='x', rotation=15)
    ax.set_title(f"{qubit_name}, shots: {num_shots}\nsquare length: {square_length}ns, zero length: {zero_length}ns\nSlice Width: {W}ns")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()


def plot_individual_iq_trajectories_line(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    square_length: int,
    zero_length: int,
    W: int,
):
    """
    Plot line IQ plane trajectories for a single qubit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        Mapping to the qubit to plot.
    square_length : int
        Square pulse length in nanoseconds.
    zero_length : int
        Zero pulse length in nanoseconds.
    W : int
        Slice width in nanoseconds.
    """
    qubit_name = qubit["qubit"]
    
    # Check if qubit exists in dataset
    if qubit_name not in ds.qubit.values:
        ax.text(0.5, 0.5, f"No data for {qubit_name}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(qubit_name)
        return
    
    ds_qubit = ds.sel(qubit=qubit_name)
    
    # Extract I and Q arrays
    Ie = ds_qubit["Ie"].values
    Qe = ds_qubit["Qe"].values
    Ig = ds_qubit["Ig"].values
    Qg = ds_qubit["Qg"].values
    
    # Get number of shots from dataset dimension before processing
    num_shots = ds_qubit.sizes.get("n_runs", Ie.shape[0] if Ie.ndim > 1 else 1)
    
    # Handle different data shapes
    if Ie.ndim == 3:
        Ie = Ie[:, :, 0]
        Qe = Qe[:, :, 0]
        Ig = Ig[:, :, 0]
        Qg = Qg[:, :, 0]

    # Average over shots
    Ie = np.mean(Ie, axis=0)
    Qe = np.mean(Qe, axis=0)
    Ig = np.mean(Ig, axis=0)
    Qg = np.mean(Qg, axis=0)

    # Create color gradients for time progression (dark to bright)
    n_slices = len(Ie)
    excited_colors = create_color_gradient("#EE3183", n_slices)
    ground_colors = create_color_gradient("#146BEE", n_slices)

    # Plot line version with color gradient
    for i in range(n_slices):
        if i == 0:
            # First point - plot with label for legend
            ax.plot([Ie[i]], [Qe[i]], "-o", label="Excited", color=excited_colors[i], markersize=4)
            ax.plot([Ig[i]], [Qg[i]], "-o", label="Ground", color=ground_colors[i], markersize=4)
        else:
            # Connect to previous point
            ax.plot([Ie[i-1], Ie[i]], [Qe[i-1], Qe[i]], "-", color=excited_colors[i-1], linewidth=1)
            ax.plot([Ig[i-1], Ig[i]], [Qg[i-1], Qg[i]], "-", color=ground_colors[i-1], linewidth=1)
            ax.plot([Ie[i]], [Qe[i]], "o", color=excited_colors[i], markersize=4)
            ax.plot([Ig[i]], [Qg[i]], "o", color=ground_colors[i], markersize=4)
    ax.scatter(Ie[0], Qe[0], color="green", s=80, zorder=3)
    ax.scatter(Ig[0], Qg[0], color="green", s=80, zorder=3)
    ax.scatter(Ie[-1], Qe[-1], color="black", s=80, zorder=3)
    ax.scatter(Ig[-1], Qg[-1], color="black", s=80, zorder=3)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    # Format x-axis ticks to prevent overlapping
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0), useMathText=True)
    ax.tick_params(axis='x', rotation=15)
    ax.set_title(f"{qubit_name}, shots: {num_shots}\nsquare length: {square_length}ns, zero length: {zero_length}ns\nSlice Width: {W}ns")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
