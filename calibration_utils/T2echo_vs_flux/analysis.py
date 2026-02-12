import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
# NOTE: qualibration_libs.analysis.fit_decay_exp assumes at most 2D input and
# slices `dat[:, :10]` / `dat[:, -10:]`, which breaks when we add a second
# non-core dimension (e.g., flux) and can change the size of that dimension.
# Use the legacy implementation which reduces along the core dimension only.
from iqcc_calibration_tools.quam_config.legacy_tools.fit import fit_decay_exp as fit_decay_exp_nd


@dataclass
class FitParameters:
    """Stores the relevant T2 echo vs flux experiment fit parameters for a single qubit"""

    # NOTE: Must be JSON-serializable because we store `asdict(FitParameters)` in node.results.
    # Therefore we use plain Python lists (not numpy arrays) and Python bool (not np.bool_).
    T2_echo: list[float]
    T2_echo_error: list[float]
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the T2 echo decay for each flux point and extract T2 vs flux.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data with dimensions (qubit, idle_time, flux).
    node : QualibrationNode
        The calibration node containing parameters and namespace.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        Dataset containing the fit results and dictionary of fit parameters per qubit.
    """
    ds_fit = ds.copy()
    
    # Determine which data variable to fit
    if node.parameters.use_state_discrimination:
        data_var = ds_fit.state
    else:
        data_var = ds_fit.I
    
    # Fit decay vs idle_time for each flux point
    # The fit_decay_exp function should handle the flux dimension
    fit_data = fit_decay_exp_nd(data_var, "idle_time")
    
    # Extract T2 echo and error for each flux point
    decay = fit_data.sel(fit_vals="decay")
    decay_res = fit_data.sel(fit_vals="decay_decay")
    
    # T2 echo = -1/decay (decay is negative)
    ds_fit["T2_echo"] = -1 / decay
    ds_fit["T2_echo"].attrs = {"long_name": "T2 echo", "units": "ns"}
    
    # T2 echo error propagation
    ds_fit["T2_echo_error"] = -ds_fit["T2_echo"] * (np.sqrt(decay_res) / decay)
    ds_fit["T2_echo_error"].attrs = {"long_name": "T2 echo error", "units": "ns"}
    
    # Assess whether the fit was successful or not
    nan_success = np.isnan(ds_fit["T2_echo"]) | np.isnan(ds_fit["T2_echo_error"])
    success_criteria = ~nan_success
    ds_fit = ds_fit.assign({"success": success_criteria})
    
    # Merge fit data into dataset
    ds_fit = xr.merge([ds_fit, fit_data.rename("fit_data")])
    
    # Populate the FitParameters class with fitted values
    fit_results = {}
    for q in ds_fit.qubit.values:
        T2_echo_values = ds_fit["T2_echo"].sel(qubit=q).values
        T2_echo_error_values = ds_fit["T2_echo_error"].sel(qubit=q).values
        success_values = ds_fit["success"].sel(qubit=q).values
        
        # Consider fit successful if at least some flux points have valid fits
        overall_success = bool(np.any(success_values))
        
        fit_results[q] = FitParameters(
            # Convert to seconds and to Python lists for JSON serialization
            T2_echo=(1e-9 * T2_echo_values).astype(float).tolist(),
            T2_echo_error=(1e-9 * T2_echo_error_values).astype(float).tolist(),
            success=overall_success,
        )
    
    return ds_fit, fit_results
