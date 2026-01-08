import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_oscillation, oscillation


@dataclass
class FitParameters:
    """Stores the relevant power Rabi E->F experiment fit parameters for a single qubit"""

    Pi_amplitude: float
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
        s_amp = f"\tE->F Pi pulse amplitude: {1e3 * fit_results[q]['Pi_amplitude']:.2f} mV\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_amp)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Process the raw dataset for power Rabi E->F."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    
    # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
    ds = ds.assign({"IQ_abs": np.sqrt(ds.I**2 + ds.Q**2)})
    
    # Add the qubit pulse absolute amplitude to the dataset
    operation = node.parameters.operation
    abs_amp = np.array([
        ds.amp_prefactor * q.xy.operations[operation].amplitude 
        for q in node.namespace["qubits"]
    ])
    ds = ds.assign_coords({
        "abs_amp": (["qubit", "amp_prefactor"], abs_amp),
    })
    ds.abs_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}
    
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the power Rabi oscillations for E->F transition.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The calibration node with parameters.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        Dataset containing the fit results and dictionary of fit parameters for each qubit.
    """
    ds_fit = ds
    operation = node.parameters.operation
    
    # Fit the power Rabi oscillations
    fit = fit_oscillation(ds_fit.IQ_abs, "amp_prefactor")
    
    # Calculate fitted oscillation
    fit_evals = oscillation(
        ds_fit.amp_prefactor,
        fit.sel(fit_vals="a"),
        fit.sel(fit_vals="f"),
        fit.sel(fit_vals="phi"),
        fit.sel(fit_vals="offset"),
    )
    
    # Merge fit results into dataset
    ds_fit = xr.merge([ds_fit, fit.rename("fit"), fit_evals.rename("fit_evals")])
    
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node, operation)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode, operation: str):
    """Add metadata to the dataset and fit results."""
    fit_results = {}
    
    for q in node.namespace["qubits"]:
        f_fit = fit.fit.sel(qubit=q.name, fit_vals="f")
        phi_fit = fit.fit.sel(qubit=q.name, fit_vals="phi")
        
        # Ensure that phi is within [-pi/2, pi/2]
        phi_fit = phi_fit - np.pi * (phi_fit > np.pi / 2)
        
        # Amplitude factor for getting an |e> -> |f> pi pulse
        factor = float(1.0 * (np.pi - phi_fit) / (2 * np.pi * f_fit))
        
        # Calibrated |e> -> |f> pi pulse absolute amplitude
        new_pi_amp = q.xy.operations[operation].amplitude * factor
        
        # Check if amplitude is reasonable (within limits)
        if np.abs(new_pi_amp) < 0.3:  # TODO: adjust limit based on hardware
            fit_results[q.name] = FitParameters(
                Pi_amplitude=new_pi_amp,
                success=True,
            )
        else:
            # If amplitude is too high, set to a safe default
            fit_results[q.name] = FitParameters(
                Pi_amplitude=0.3,  # TODO: adjust default based on hardware
                success=False,
            )
    
    return fit, fit_results

