import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""

    success: bool
    resonator_frequency: float
    frequency_shift: float
    optimal_power: float


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_power = f"Optimal readout power: {fit_results[q]['optimal_power']:.2f} dBm | "
        s_freq = f"Resonator frequency: {1e-9 * fit_results[q]['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.0f} MHz)\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_power + s_freq + s_shift)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Processes the raw dataset by converting the 'I' and 'Q' quadratures to V, or adding the RF_frequency as a coordinate for instance."""

    # Convert the 'I' and 'Q' quadratures from demodulation units to V.
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Normalize the IQ_abs with respect to the amplitude axis
    ds = ds.assign({"IQ_abs_norm": ds["IQ_abs"] / ds.IQ_abs.mean(dim=["detuning"])})
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Track the resonator dip vs readout power, fit an arctan to the trajectory,
    and determine the optimal readout power and corresponding frequency shift.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        Node containing parameters and qubit metadata.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        Fit dataset with analysis coordinates and per-qubit fit results.
    """

    ds_fit = ds
    # Track the minimum IQ value per power as a proxy for resonator frequency
    ds_fit["rr_min_response"] = ds.IQ_abs_norm.idxmin(dim="detuning")

    # --- Outlier filtering ---
    freq_step_hz = node.parameters.frequency_step_in_mhz * 1e6
    clip_left_hz = node.parameters.outlier_clip_left_mhz * 1e6
    clip_right_hz = 1e6
    outlier_threshold_hz = node.parameters.outlier_threshold_n_steps * freq_step_hz

    ds_fit["rr_min_response"] = ds_fit["rr_min_response"].where(
        (ds_fit["rr_min_response"] >= -clip_left_hz)
        & (ds_fit["rr_min_response"] <= clip_right_hz)
    )
    rolling_med = ds_fit["rr_min_response"].rolling(power=5, center=True, min_periods=1).median()
    ds_fit["rr_min_response"] = ds_fit["rr_min_response"].where(
        np.abs(ds_fit["rr_min_response"] - rolling_med) <= outlier_threshold_hz
    )

    # --- Arctan fit ---
    qubit_names = ds_fit.qubit.values
    power_vals = ds_fit.power.values
    arctan_fit_vals = np.full((len(qubit_names), len(power_vals)), np.nan)
    arctan_deriv_vals = np.full((len(qubit_names), len(power_vals)), np.nan)

    for i, q in enumerate(qubit_names):
        rr = ds_fit["rr_min_response"].sel(qubit=q).values
        valid = ~np.isnan(rr)
        if np.sum(valid) > 4:
            try:
                popt, _ = _fit_arctan(power_vals[valid], rr[valid])
                arctan_fit_vals[i] = _arctan_model(power_vals, *popt)
                arctan_deriv_vals[i] = _arctan_derivative(power_vals, *popt)
            except RuntimeError:
                pass

    ds_fit["rr_min_response_arctan_fit"] = xr.DataArray(
        arctan_fit_vals, dims=["qubit", "power"],
        coords={"qubit": qubit_names, "power": power_vals},
    )
    ds_fit["rr_min_response_avg"] = xr.DataArray(
        arctan_deriv_vals, dims=["qubit", "power"],
        coords={"qubit": qubit_names, "power": power_vals},
    )

    # Find where the derivative crosses below the threshold
    ds_fit["below_threshold"] = ds_fit.rr_min_response_avg < node.parameters.derivative_crossing_threshold_in_hz_per_dbm
    optimal_power = ds_fit.below_threshold.idxmax(dim="power")
    optimal_power -= node.parameters.buffer_from_crossing_threshold_in_dbm
    ds_fit = ds_fit.assign_coords({"optimal_power": (["qubit"], optimal_power.data)})

    # Get the resonance frequency shift at the optimal power
    freq_shift = []
    for q in node.namespace["qubits"]:
        freq_shift.append(float(
            peaks_dips(
                ds_fit.sel(power=ds_fit["optimal_power"].sel(qubit=q.name).data, method="nearest")
                .sel(qubit=q.name).IQ_abs,
                "detuning",
            ).position.data
        ))
    ds_fit = ds_fit.assign_coords({"freq_shift": (["qubit"], freq_shift)})

    # Extract the relevant fitted parameters
    fit_dataset, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_dataset, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the fit dataset and fit result dictionary."""

    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.freq_shift + full_freq
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    freq_success = np.abs(fit.freq_shift.data) < node.parameters.frequency_span_in_mhz * 1e6
    nan_success = np.isnan(fit.freq_shift.data) | np.isnan(fit.optimal_power.data)
    success_criteria = freq_success & ~nan_success
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {
        q: FitParameters(
            success=fit.sel(qubit=q).success.values.__bool__(),
            resonator_frequency=float(fit.res_freq.sel(qubit=q).values),
            frequency_shift=float(fit.freq_shift.sel(qubit=q).data),
            optimal_power=float(fit.optimal_power.sel(qubit=q).data),
        )
        for q in fit.qubit.values
    }

    return fit, fit_results


def _arctan_model(x, a, b, x0, c):
    """Arctan model: a * arctan(b * (x - x0)) + c"""
    return a * np.arctan(b * (x - x0)) + c


def _arctan_derivative(x, a, b, x0, _c):
    """Analytical derivative of the arctan model."""
    return a * b / (1 + (b * (x - x0)) ** 2)


def _fit_arctan(x, y):
    """Fit the arctan model to data with automatic initial guesses.

    The fit is constrained so that a < 0 and b > 0, enforcing a monotonically
    decreasing curve (higher frequency at low power, lower at high power).
    """
    a_guess = (y[-1] - y[0]) / np.pi
    p0 = [
        -abs(a_guess) if a_guess != 0 else -1.0,
        abs(2.0 / (x[-1] - x[0])),
        np.median(x),
        np.mean(y),
    ]
    bounds = (
        [-np.inf, 0, -np.inf, -np.inf],
        [0, np.inf, np.inf, np.inf],
    )
    return curve_fit(_arctan_model, x, y, p0=p0, bounds=bounds, maxfev=10_000)
