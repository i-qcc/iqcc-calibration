import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips

u = unit(coerce_to_integer=True)


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy E->F transition experiment fit parameters for a single qubit"""

    anharmonicity: float
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
        s_anharmonicity = f"\tAnharmonicity: {1e-6 * fit_results[q]['anharmonicity']:.3f} MHz\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_anharmonicity)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Process the raw dataset for E->F transition spectroscopy."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    
    # Calculate IQ_abs (amplitude) with baseline removed
    ds = ds.assign({
        "IQ_abs": np.sqrt((ds.I - ds.I.mean(dim="detuning"))**2 + (ds.Q - ds.Q.mean(dim="detuning"))**2)
    })
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    
    # Add the qubit frequency coordinates
    # For E->F transition, we scan around f_01 - anharmonicity
    qubit_freqs = {q.name: q.xy.RF_frequency for q in node.namespace["qubits"]}
    anharmonicities = {q.name: q.anharmonicity for q in node.namespace["qubits"]}
    
    # Get the detuning values from the dataset
    detuning_values = ds.detuning.values
    
    # Calculate full frequency and detuning coordinates
    full_freq = np.array([
        detuning_values + qubit_freqs[q.name] - anharmonicities[q.name]
        for q in node.namespace["qubits"]
    ])
    
    detuning_coords = np.array([
        detuning_values - anharmonicities[q.name]
        for q in node.namespace["qubits"]
    ])
    
    ds = ds.assign_coords({
        "freq_full": (["qubit", "detuning"], full_freq),
        "detuning_ef": (["qubit", "detuning"], detuning_coords),
    })
    ds.freq_full.attrs = {"long_name": "RF frequency", "units": "Hz"}
    ds.detuning_ef.attrs = {"long_name": "E->F detuning", "units": "Hz"}
    
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the E->F transition frequency and calculate anharmonicity for each qubit in the dataset.

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
    
    # Find the peak with minimal prominence as defined, if no such peak found, returns nan
    # Use the I quadrature for fitting
    fit_vals = peaks_dips(ds_fit.I, dim="detuning", prominence_factor=3, remove_baseline=True)
    ds_fit = xr.merge([ds_fit, fit_vals])
    
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    
    # Calculate the modified anharmonicity from the detuning at the peak position
    # The detuning_ef at the peak position gives us the anharmonicity correction
    anharmonicities = {}
    for q in node.namespace["qubits"]:
        if not np.isnan(fit.position.sel(qubit=q.name).values):
            # The detuning_ef at the peak position gives us the anharmonicity
            # We need to select using the detuning dimension (which is the position)
            peak_pos = fit.position.sel(qubit=q.name).values
            detuning_at_peak = fit.detuning_ef.sel(qubit=q.name, detuning=peak_pos)
            anharmonicities[q.name] = detuning_at_peak.values
        else:
            anharmonicities[q.name] = np.nan
    
    # Assess whether the fit was successful or not
    success_criteria = ~np.isnan(fit.position.values)
    fit = fit.assign({"success": ("qubit", success_criteria)})
    
    fit_results = {}
    for q in node.namespace["qubits"]:
        if not np.isnan(fit.position.sel(qubit=q.name).values):
            fit_results[q.name] = FitParameters(
                anharmonicity=anharmonicities[q.name],
                success=True,
            )
        else:
            fit_results[q.name] = FitParameters(
                anharmonicity=np.nan,
                success=False,
            )
    
    return fit, fit_results

