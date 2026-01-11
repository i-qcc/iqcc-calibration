import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from qualibration_libs.data import convert_IQ_to_V
from qualibrate import QualibrationNode


@dataclass
class FitParameters:
    """Stores the relevant readout trajectory experiment fit parameters for a single qubit"""

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

    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
   
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ie", "Qe","Ig", "Qg"])
    #ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig_slices", "Qg_slices"])
    # ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Iadce","Qadce"])
    # IF = node.parameters.IF*10**(-9)
    # t = ds["readout_time"]
    # print(f"somthing {np.exp(-1j*2*np.pi*IF*t)}")
    # ds = ds.assign({"S": (ds["Iadce"]+1j*ds["Qadce"]) * np.exp(-1j*2*np.pi*IF*t)})
    
    #ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["I_slices", "Q_slices"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Process the readout trajectory data and create fit results.

    For readout trajectory measurements, the main purpose is visualization rather than fitting.
    This function validates the data and creates a minimal fit result structure.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data with Ie, Qe, Ig, Qg.
    node : QualibrationNode
        The calibration node containing qubits and parameters.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        Dataset containing the processed data and dictionary of fit results per qubit.
    """
    ds_fit = ds.copy()
    
    # Calculate state separation as a metric
    if "Ie" in ds_fit and "Qe" in ds_fit and "Ig" in ds_fit and "Qg" in ds_fit:
        # Average over shots if present
        Ie = ds_fit["Ie"].mean("n_runs") if "n_runs" in ds_fit["Ie"].dims else ds_fit["Ie"]
        Qe = ds_fit["Qe"].mean("n_runs") if "n_runs" in ds_fit["Qe"].dims else ds_fit["Qe"]
        Ig = ds_fit["Ig"].mean("n_runs") if "n_runs" in ds_fit["Ig"].dims else ds_fit["Ig"]
        Qg = ds_fit["Qg"].mean("n_runs") if "n_runs" in ds_fit["Qg"].dims else ds_fit["Qg"]
        
        # Calculate separation
        separation = np.sqrt((Ie - Ig) ** 2 + (Qe - Qg) ** 2)
        ds_fit["separation"] = separation
        ds_fit["separation"].attrs = {"long_name": "State separation", "units": "V"}
        
        # Calculate maximum separation as a quality metric
        max_separation = float(separation.max().values)
        min_separation = float(separation.min().values)
        
        # Success criteria: reasonable separation between states
        success_criteria = max_separation > 0.01  # At least 10mV separation
        
        # Populate the FitParameters class
        fit_results = {}
        for q in ds_fit.qubit.values:
            fit_results[q] = FitParameters(success=bool(success_criteria))
    else:
        # If data is missing, mark as failed
        success_criteria = False
        fit_results = {
            q: FitParameters(success=False)
            for q in ds_fit.qubit.values
        }
    
    # Set outcomes for the node
    node.outcomes = {
        q: "successful" if fit_results[q].success else "failed"
        for q in fit_results.keys()
    }
    
    # Add success coordinate to dataset
    ds_fit = ds_fit.assign_coords(
        success=("qubit", [fit_results[q].success for q in ds_fit.qubit.values])
    )

    return ds_fit, fit_results


