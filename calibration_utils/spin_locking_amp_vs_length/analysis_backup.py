import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_oscillation
from iqcc_calibration_tools.quam_config.lib.instrument_limits import instrument_limits


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    opt_amp_prefactor: float
    opt_amp: float
    operation: str
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
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_amp = f"The calibrated {fit_results[q]['operation']} amplitude: {1e3 * fit_results[q]['opt_amp']:.2f} mV (x{fit_results[q]['opt_amp_prefactor']:.2f})\n "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_amp)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    
    sl_op_name = node.parameters.spin_locking_operation # Use the correct parameter
    
    # FIX: Use spin_locking_operation and q.xy_SL (Spin-Locking component)
    full_amp = np.array(
        [ds.amp_prefactor * q.xy_SL.operations[sl_op_name].amplitude for q in node.namespace["qubits"]]
    )
    
    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Analyzes the 2D Spin-Locking map (Amplitude vs. Length) to find the single best amplitude 
    pre-factor by finding the minimum average signal over all tested pulse durations.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data with dimensions (qubit, duration_spin_locking, amp_prefactor).
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the analysis results.
    """
    ds_fit = ds
    
    # The data variable is either 'state' (0 or 1) or 'I' (Voltage)
    data_var = ds.state if node.parameters.use_state_discrimination else ds.I
    
    # Get the average along the duration_spin_locking axis
    ds_fit["data_mean"] = data_var.mean(dim="duration_spin_locking")
    
    # Always find the minimum along the amp_prefactor axis
    ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmin(dim="amp_prefactor")
    
    # --- FIX 1: Pass the original dataset 'ds' to the helper function ---
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node, ds)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode, ds: xr.Dataset):
    """
    Add metadata to the dataset and populate the FitParameters class.
    
    Parameters:
    -----------
    fit : xr.Dataset
        Dataset containing the intermediate fitting results (e.g., opt_amp_prefactor).
    node : QualibrationNode
        The current node object.
    ds : xr.Dataset
        The original raw dataset (needed to retrieve the data variable for the final merge).
    """
    
    sl_op_name = node.parameters.spin_locking_operation
    # Use q.xy_SL for instrument limits
    limits = [instrument_limits(q.xy_SL) for q in node.namespace["qubits"]]
    
    # Calculate optimal absolute amplitude (V)
    current_SL_amps = xr.DataArray(
        [q.xy_SL.operations[sl_op_name].amplitude for q in node.namespace["qubits"]],
        coords=dict(qubit=fit.qubit.data),
    )
    
    # Calculate optimal absolute amplitude
    opt_amp = fit.opt_amp_prefactor * current_SL_amps
    fit = fit.assign({"opt_amp": opt_amp})
    fit.opt_amp.attrs = {"long_name": f"Optimal {sl_op_name} amplitude", "units": "V"}
    
    # Assess success criteria
    nan_success = np.isnan(fit.opt_amp_prefactor) | np.isnan(fit.opt_amp)
    amp_success = fit.opt_amp < limits[0].max_x180_wf_amplitude 
    success_criteria = ~nan_success & amp_success
    fit = fit.assign({"success": success_criteria})
    
    # Populate the FitParameters class
    fit_results = {
        q: FitParameters(
            opt_amp_prefactor=fit.sel(qubit=q).opt_amp_prefactor.values.__float__(),
            opt_amp=fit.sel(qubit=q).opt_amp.values.__float__(),
            operation=sl_op_name,
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    
    # Merge the new optimal points and the raw data variable into the final fit dataset
    data_var_name = "state" if node.parameters.use_state_discrimination else "I"
    fit_data = xr.merge([fit, ds[data_var_name]])
    return fit_data, fit_results