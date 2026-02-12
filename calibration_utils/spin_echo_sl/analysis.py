import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_decay_exp


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    T2_SL: float
    T2_SL_error: float
    success: bool


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
    # for q in fit_results.keys():
    #     s_qubit = f"Results for qubit {q}: "
    #     s_freq = f"\tQubit frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
    #     s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
    #     s_angle = (
    #         f"The integration weight angle: {fit_results[q]['iw_angle']:.3f} rad\n "
    #     )
    #     s_saturation = f"To get the desired FWHM, the saturation amplitude is updated to: {1e3 * fit_results[q]['saturation_amp']:.1f} mV | "
    #     s_x180 = f"To get the desired x180 gate, the x180 amplitude is updated to: {1e3 * fit_results[q]['x180_amp']:.1f} mV\n "
    #     if fit_results[q]["success"]:
    #         s_qubit += " SUCCESS!\n"
    #     else:
    #         s_qubit += " FAIL!\n"
    #     logger.info(
    #         s_qubit + s_freq + s_fwhm + s_freq + s_angle + s_saturation + s_x180
    #     )
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    ds_fit = ds
    # # Fit the exponential decay
    if node.parameters.use_state_discrimination:
        fit_data = fit_decay_exp(ds_fit.state, "idle_time")
        ds_fit = xr.merge([ds, fit_data.rename("fit_data")])
    else:
        # Fit I (always)
        fit_data_I = fit_decay_exp(ds_fit.I, "idle_time")
        ds_fit = xr.merge([ds, fit_data_I.rename("fit_data_I")])
        # For backward compatibility, also add fit_data as fit_data_I
        ds_fit = ds_fit.assign(fit_data=ds_fit.fit_data_I)
        # Fit Q only if skip_Q_analysis is False
        if not getattr(node.parameters, 'skip_Q_analysis', False):
            fit_data_Q = fit_decay_exp(ds_fit.Q, "idle_time")
            ds_fit = xr.merge([ds_fit, fit_data_Q.rename("fit_data_Q")])

    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit)
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset):
    """Add metadata to the dataset and fit results."""
    # Decay rate and its uncertainty
    decay = ds_fit.fit_data.sel(fit_vals="decay")
    decay_res = ds_fit.fit_data.sel(fit_vals="decay_decay")
    # T2_SL and its uncertainty
    ds_fit["T2_SL"] = -1 / ds_fit.fit_data.sel(fit_vals="decay")
    ds_fit["T2_SL"].attrs = {"long_name": "T2 SL", "units": "ns"}
    ds_fit["T2_SL_error"] = -ds_fit["T2_SL"] * (np.sqrt(decay_res) / decay)
    ds_fit["T2_SL_error"].attrs = {"long_name": "T2 SL error", "units": "ns"}
    
    # If Q fit data exists, also extract T2_SL for Q
    if "fit_data_Q" in ds_fit.data_vars:
        decay_Q = ds_fit.fit_data_Q.sel(fit_vals="decay")
        decay_res_Q = ds_fit.fit_data_Q.sel(fit_vals="decay_decay")
        ds_fit["T2_SL_Q"] = -1 / ds_fit.fit_data_Q.sel(fit_vals="decay")
        ds_fit["T2_SL_Q"].attrs = {"long_name": "T2 SL Q", "units": "ns"}
        ds_fit["T2_SL_error_Q"] = -ds_fit["T2_SL_Q"] * (np.sqrt(decay_res_Q) / decay_Q)
        ds_fit["T2_SL_error_Q"].attrs = {"long_name": "T2 SL Q error", "units": "ns"}
    
    # Assess whether the fit was successful or not
    nan_success = np.isnan(ds_fit["T2_SL"]) | np.isnan(ds_fit["T2_SL_error"])
    success_criteria = ~nan_success
    ds_fit = ds_fit.assign({"success": success_criteria})
    # Populate the FitParameters class with fitted values
    fit_results = {
        q: FitParameters(
            T2_SL=1e-9 * float(ds_fit["T2_SL"].sel(qubit=q)),
            T2_SL_error=1e-9 * float(ds_fit["T2_SL_error"].sel(qubit=q)),
            success=bool(ds_fit["success"].sel(qubit=q)),
        )
        for q in ds_fit.qubit.values
    }
    return ds_fit, fit_results
