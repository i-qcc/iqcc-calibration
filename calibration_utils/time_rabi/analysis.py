import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Callable
import numpy as np
import xarray as xr

# Use the generic QualibrationNode for functions
from qualibrate import QualibrationNode 
from qualibration_libs.data import convert_IQ_to_V
# Import the standard oscillation fitting functions
from qualibration_libs.analysis import fit_oscillation, oscillation
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)
LOG = logging.getLogger(__name__)


@dataclass
class FitParameters:
    """Stores the relevant fit parameters for a single qubit"""
    f: float  # Rabi frequency in GHz (since x-axis is in ns)
    opt_dur_pi: float # Optimal pi-pulse duration in ns
    opt_dur_pi_half: float # Optimal pi/2-pulse duration in ns
    success: bool
    fit_params: dict # Raw fit parameters [a, f, phi, offset]
    

def process_raw_dataset(
    ds: xr.Dataset, 
    node: QualibrationNode
) -> xr.Dataset:
    """
    Process the raw dataset, e.g., converting I, Q to Voltage.
    """
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(
    ds: xr.Dataset, 
    node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Fit the raw data to an oscillation and extract the optimal pulse durations.
    """
    qubits = node.namespace["qubits"]
    fit_results = {}
    
    # Determine which signal to fit
    if node.parameters.use_state_discrimination:
        signal_name = "state"
    else:
        # Assuming I or V, let's pick the one that exists
        signal_name = "V" if "V" in ds else "I"
        
    # Get the DataArray of the signal to be fitted
    signal_da = ds[signal_name]
        
    # Use the standard library fitter
    # 1st arg: The DataArray to fit (e.g., ds.state or ds.V)
    # 2nd arg: The name of the x-axis coordinate (e.g., "duration")
    fit_data = fit_oscillation(signal_da, "duration")
    
    fit_evals = []
    
    for q in qubits:
        q_name = q.name
        try:
            q_fit = fit_data.sel(qubit=q_name)
            
            # Extract fit parameters
            a, f, phi, offset = q_fit.values
            
            # Create the dictionary for the dataclass
            fit_params_dict = {
                "a": a, "f": f, "phi": phi, "offset": offset
            }
            
            # Calculate optimal durations
            # f is in GHz (since duration is in ns), so 1/f is period in ns
            # We must handle the case where f is 0 to avoid division by zero
            if f == 0:
                raise ValueError("Fit frequency is zero, cannot calculate duration.")
            
            # Period T = 1 / f
            # pi-pulse is T / 2 = 1 / (2 * f)
            # pi/2-pulse is T / 4 = 1 / (4 * f)
            opt_dur_pi = 1 / (2 * f)
            opt_dur_pi_half = 1 / (4 * f)
            
            # Store results
            fit_results[q_name] = FitParameters(
                f=f,
                opt_dur_pi=opt_dur_pi,
                opt_dur_pi_half=opt_dur_pi_half,
                success=True,
                fit_params=fit_params_dict
            )
            
            # Generate the fitted curve for plotting
            fit_curve = oscillation(ds.duration, a, f, phi, offset)
            fit_evals.append(fit_curve)
            
        except Exception as e:
            LOG.warning(f"Fit failed for qubit {q_name}: {e}")
            fit_results[q_name] = FitParameters(
                f=0, opt_dur_pi=0, opt_dur_pi_half=0, success=False, fit_params={}
            )
            # Add a flat line for failed fits
            fit_evals.append(xr.full_like(ds.duration, np.nan))

    # Create the dataset for fitted curves
    ds_fit = xr.concat(fit_evals, dim="qubit")
    ds_fit["qubit"] = ds["qubit"]
    
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: Dict[str, dict], # <-- Note: it receives a dict of dicts
    log_callable: Callable
):
    """Log the fitted results to the node's log."""
    for qubit_name, fit_result in fit_results.items():
        # *** THIS IS THE FIX ***
        # Use dictionary key access, e.g., fit_result["success"]
        if fit_result["success"]:
            log_callable(f"Qubit {qubit_name}: [SUCCESS]")
            log_callable(f"  Rabi Freq. = {fit_result['f'] * 1e3:.2f} MHz")
            log_callable(f"  Pi-pulse   = {fit_result['opt_dur_pi']:.1f} ns")
            log_callable(f"  Pi/2-pulse = {fit_result['opt_dur_pi_half']:.1f} ns")
        else:
            log_callable(f"Qubit {qubit_name}: [FAILED]")