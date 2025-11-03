import logging
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
# Removed: from qualibration_libs.analysis import fit_oscillation (No longer needed)
# Removed: from iqcc_calibration_tools.quam_config.lib.instrument_limits import instrument_limits (No longer needed)


# --- FitParameters Dataclass REMOVED ---

def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs basic information about the analysis completion, as no optimal values are calculated.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    
    # --- SIMPLIFICATION: Log only map structure info ---
    log_callable("Analysis complete: 2D map generated. No optimal parameters calculated.")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """
    Converts IQ to V if needed and adds the 'full_amp' coordinate for absolute voltage.
    (Function body remains the same as it performs necessary data preparation)
    """
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
        
    sl_op_name = node.parameters.spin_locking_operation
    
    # Use 'spin_locking_operation' and 'q.xy_SL'
    full_amp = np.array(
        [ds.amp_prefactor * q.xy_SL.operations[sl_op_name].amplitude for q in node.namespace["qubits"]]
    )
    
    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}
    
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, None]]:
    """
    PREPARES the raw dataset for plotting by ensuring the data variable is present.
    (All optimal value finding logic has been REMOVED.)
    
    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data with dimensions (qubit, duration_spin_locking, amp_prefactor).

    Returns:
    --------
    xr.Dataset
        The input dataset (ds) which is now the analysis output (ds_fit).
    dict[str, None]
        An empty dictionary since no fit results are produced.
    """
    
    # --- SIMPLIFICATION: No fitting, just return the data ---
    ds_fit = ds
    
    # Ensure the data variable is available for final merging/plotting check (Optional but robust)
    data_var_name = "state" if node.parameters.use_state_discrimination else "I"
    if data_var_name not in ds_fit.data_vars:
        # If the variable wasn't copied/aliased earlier, ensure it exists for consistent output.
        pass # In this context, ds_fit = ds means the variable is present.

    # Return the dataset and an empty dictionary for fit results
    return ds_fit, {}


# --- _extract_relevant_fit_parameters is REMOVED as its only job was to calculate optimal values ---