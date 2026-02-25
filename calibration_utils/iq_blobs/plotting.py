from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

_LABEL_BBOX = dict(boxstyle="round,pad=0.3", alpha=0.7)


def _group_by_readout_line(
    qubit_names: List[str],
    values: Dict[str, List[float]],
) -> Dict[str, Dict[str, List[float]]]:
    """Group per-qubit values by readout line (second character of the qubit name)."""
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for name in qubit_names:
        line_id = name[1] if len(name) > 1 else name
        for key, vals in values.items():
            idx = qubit_names.index(name)
            grouped[line_id][key].append(vals[idx])
    return grouped


def plot_iq_blobs(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the IQ blobs with the derived thresholds for the given qubits.

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
        plot_individual_iq_blobs(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    leg = grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    leg.legend_handles[0].set_markersize(6)
    leg.legend_handles[1].set_markersize(6)
    grid.fig.suptitle("g.s. and e.s. discriminators (rotated)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_iq_blobs(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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

    ax.plot(1e3 * fit.Ig_rot, 1e3 * fit.Qg_rot, ".", alpha=1, label="Ground", markersize=2)
    ax.plot(
        1e3 * fit.Ie_rot,
        1e3 * fit.Qe_rot,
        ".",
        alpha=1,
        label="Excited",
        markersize=2,
    )
    ax.axvline(
        1e3 * fit.rus_threshold,
        color="k",
        linestyle="--",
        lw=1,
        label="RUS Threshold",
    )
    ax.axvline(1e3 * fit.ge_threshold, color="r", 
               linestyle="--", lw=2, label="Threshold")
    ax.axis("equal")
    ax.set_xlabel("I [mV]")
    ax.set_ylabel("Q [mV]")
    ax.set_title(qubit["qubit"])


def plot_historams(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the IQ blobs with the derived thresholds for the given qubits.

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
        plot_individual_histograms(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    leg = grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    grid.fig.suptitle("g.s. and e.s. histograms (rotated)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_histograms(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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

    ax.hist(1e3 * fit.Ig_rot, bins=100, alpha=0.5, label="Ground")
    ax.hist(1e3 * fit.Ie_rot, bins=100, alpha=0.5, label="Excited")
    ax.axvline(
        1e3 * fit.rus_threshold,
        color="k",
        linestyle="--",
        lw=0.5,
        label="RUS Threshold",
    )
    ax.axvline(1e3 * fit.ge_threshold, color="r", linestyle="--", lw=0.5, label="Threshold")
    ax.set_xlabel("I Rotated [mV]")
    ax.set_ylabel("Counts")
    ax.set_title(qubit["qubit"])


def plot_snr_gaussians(ds_raw: xr.Dataset, ds_fit: xr.Dataset, qubits: List[AnyTransmon], node) -> Figure:
    """
    Plot SNR Gaussian fits for all qubits on a QubitGrid layout.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw IQ blob dataset.
    ds_fit : xr.Dataset
        Fitted dataset containing rotated IQ data.
    qubits : list of AnyTransmon
        List of qubit objects.
    node : QualibrationNode
        The calibration node (used by fit_raw_data / fit_snr_with_gaussians).

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    from calibration_utils.iq_blobs.analysis import fit_raw_data, fit_snr_with_gaussians

    _, fit_results = fit_raw_data(ds_raw, node)

    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    axes_map = {}
    for ax, qubit in grid_iter(grid):
        axes_map[qubit["qubit"]] = ax
    axes = np.array([axes_map[q.name] for q in qubits])

    fit_snr_with_gaussians(
        fits=ds_fit,
        qubits=qubits,
        node=node,
        fit_results=fit_results,
        axes=axes,
        plot=True,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    for ax_item in axes:
        leg = ax_item.get_legend()
        if leg is not None:
            leg.remove()
    grid.fig.legend(handles, labels, loc="upper right", fontsize=11)
    grid.fig.suptitle("SNR Gaussian fits")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_confusion_matrices(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the confusion matrix for the given qubits.

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
        plot_individual_confusion_matrix(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    qubit_names = [q.name for q in qubits]
    gg_vals = [float(fits.sel(qubit=q).gg) for q in qubit_names]
    ee_vals = [float(fits.sel(qubit=q).ee) for q in qubit_names]

    grid.fig.text(
        0.99, 0.98,
        f"Avg \u27E800\u27E9 = {np.mean(gg_vals):.3f} \u00B1 {np.std(gg_vals):.3f}\n"
        f"Avg \u27E811\u27E9 = {np.mean(ee_vals):.3f} \u00B1 {np.std(ee_vals):.3f}",
        fontsize=11, verticalalignment="top", horizontalalignment="right",
        bbox={**_LABEL_BBOX, "facecolor": "lightblue"},
    )

    grouped = _group_by_readout_line(qubit_names, {"gg": gg_vals, "ee": ee_vals})
    per_line_text = "\n".join(
        f"{lid}: "
        f"\u27E800\u27E9 = {np.mean(g['gg']):.3f} \u00B1 {np.std(g['gg']):.3f}, "
        f"\u27E811\u27E9 = {np.mean(g['ee']):.3f} \u00B1 {np.std(g['ee']):.3f}"
        for lid, g in sorted(grouped.items())
    )
    grid.fig.text(
        0.99, 0.02, per_line_text,
        fontsize=9, verticalalignment="bottom", horizontalalignment="right",
        bbox={**_LABEL_BBOX, "facecolor": "lightyellow"},
    )

    grid.fig.suptitle("g.s. and e.s. fidelity")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_confusion_matrix(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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

    confusion = np.array([[float(fit.gg), float(fit.ge)], [float(fit.eg), float(fit.ee)]])
    ax.imshow(confusion)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels=["|g>", "|e>"])
    ax.set_yticklabels(labels=["|g>", "|e>"])
    ax.set_ylabel("Prepared")
    ax.set_xlabel("Measured")
    ax.text(0, 0, f"{100 * confusion[0][0]:.1f}%", ha="center", va="center", color="k")
    ax.text(1, 0, f"{100 * confusion[0][1]:.1f}%", ha="center", va="center", color="w")
    ax.text(0, 1, f"{100 * confusion[1][0]:.1f}%", ha="center", va="center", color="w")
    ax.text(1, 1, f"{100 * confusion[1][1]:.1f}%", ha="center", va="center", color="k")
    ax.set_title(qubit["qubit"])
