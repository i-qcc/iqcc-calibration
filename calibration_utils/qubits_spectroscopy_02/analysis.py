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
    """Stores the relevant qubit spectroscopy 0->2 transition experiment fit parameters for a single qubit"""

    f01: float
    f02_2: float
    anharmonicity: float
    peak_width: float
    iw_angle: float
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
        s_f01 = f"\tf01 frequency: {1e-9 * fit_results[q]['f01']:.6f} GHz | "
        s_anharmonicity = f"Measured anharmonicity: {1e-6 * fit_results[q]['anharmonicity']:.2f} MHz | "
        s_peak_width = f"Peak width: {1e-6 * fit_results[q]['peak_width']:.2f} MHz | "
        s_angle = f"Integration weight angle: {fit_results[q]['iw_angle']:.4f} rad\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_f01 + s_anharmonicity + s_peak_width + s_angle)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Process the raw dataset for 0->2 transition spectroscopy."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    
    # Use the detunings stored in namespace from QUA program creation
    qubit_freqs = {q.name: q.xy.RF_frequency for q in node.namespace["qubits"]}
    detunings = node.namespace.get("detunings", {})
    
    # If detunings not in namespace (e.g., when loading data), recalculate them
    if not detunings:
        init_anharmonicity = node.parameters.initial_anharmonicity_mhz * u.MHz
        detunings = {}
        for q in node.namespace["qubits"]:
            if node.parameters.arbitrary_flux_bias is not None:
                arb_flux_bias_offset = node.parameters.arbitrary_flux_bias
                detuning = q.freq_vs_flux_01_quad_term * arb_flux_bias_offset ** 2
            elif node.parameters.arbitrary_qubit_frequency_in_ghz is not None:
                detuning = 1e9 * node.parameters.arbitrary_qubit_frequency_in_ghz - qubit_freqs[q.name]
            else:
                detuning = 0.0
            # Adjust for 0->2 transition search (f01 - α/2)
            detunings[q.name] = detuning - init_anharmonicity / 2
    
    full_freq = np.array([
        ds.detuning + qubit_freqs[q.name] + detunings[q.name] 
        for q in node.namespace["qubits"]
    ])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the 0->2 transition frequency and calculate anharmonicity for each qubit in the dataset.

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
    # Use stored init_anharmonicity from namespace if available, otherwise calculate
    init_anharmonicity = node.namespace.get("init_anharmonicity")
    if init_anharmonicity is None:
        init_anharmonicity = node.parameters.initial_anharmonicity_mhz * u.MHz
    
    # Search for frequency for which the amplitude is farthest from the mean to indicate the approximate location of the peak
    shifts = np.abs((ds_fit.IQ_abs - ds_fit.IQ_abs.mean(dim="detuning"))).idxmax(dim="detuning")
    # Find the rotation angle to align the separation along the 'I' axis
    angle = np.arctan2(
        ds_fit.sel(detuning=shifts).Q - ds_fit.Q.mean(dim="detuning"),
        ds_fit.sel(detuning=shifts).I - ds_fit.I.mean(dim="detuning"),
    )
    ds_fit = ds_fit.assign({"iw_angle": angle})
    # Rotate the data to the new I axis
    ds_fit = ds_fit.assign({"I_rot": ds_fit.I * np.cos(ds_fit.iw_angle) + ds_fit.Q * np.sin(ds_fit.iw_angle)})
    # Find the peak with minimal prominence as defined, if no such peak found, returns nan
    fit_vals = peaks_dips(ds_fit.I_rot, dim="detuning", prominence_factor=5)
    ds_fit = xr.merge([ds_fit, fit_vals])
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node, init_anharmonicity)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode, init_anharmonicity):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    
    # Get the qubit frequencies
    qubit_freqs = {q.name: q.xy.RF_frequency for q in node.namespace["qubits"]}
    
    # Use stored detunings from namespace if available, otherwise recalculate
    detunings = node.namespace.get("detunings", {})
    if not detunings:
        detunings = {}
        for q in node.namespace["qubits"]:
            if node.parameters.arbitrary_flux_bias is not None:
                arb_flux_bias_offset = node.parameters.arbitrary_flux_bias
                detuning = q.freq_vs_flux_01_quad_term * arb_flux_bias_offset ** 2
            elif node.parameters.arbitrary_qubit_frequency_in_ghz is not None:
                detuning = 1e9 * node.parameters.arbitrary_qubit_frequency_in_ghz - qubit_freqs[q.name]
            else:
                detuning = 0.0
            # Adjust for 0->2 transition search (f01 - α/2)
            detunings[q.name] = detuning - init_anharmonicity / 2
    
    # Get the fitted 0->2 transition frequency
    # The position is relative to the detuning axis, so we need to convert it to absolute frequency
    full_freqs = np.array([
        fit.position.sel(qubit=q.name).values + qubit_freqs[q.name] + detunings[q.name]
        for q in node.namespace["qubits"]
    ])
    
    # Calculate f02_2 (the detuning at which we found the peak, relative to f01 - α/2)
    f02_2_values = np.array([
        fit.position.sel(qubit=q.name).values + init_anharmonicity / 2
        for q in node.namespace["qubits"]
    ])
    
    # Calculate anharmonicity: α = 2 * f02_2 (where f02_2 is the detuning from f01 - α/2)
    anharmonicity_values = 2 * f02_2_values
    
    # Get the peak width
    peak_width_values = np.abs(fit.width.values)
    
    # Get optimum iw angle
    prev_angles = np.array(
        [q.resonator.operations["readout"].integration_weights_angle for q in node.namespace["qubits"]]
    )
    fit = fit.assign({"iw_angle": (prev_angles + fit.iw_angle) % (2 * np.pi)})
    fit.iw_angle.attrs = {"long_name": "integration weight angle", "units": "rad"}
    
    # Assess whether the fit was successful or not
    # Check if position is not NaN
    success_criteria = ~np.isnan(fit.position.values)
    fit = fit.assign({"success": ("qubit", success_criteria)})
    
    fit_results = {}
    for i, q in enumerate(node.namespace["qubits"]):
        if not np.isnan(fit.position.sel(qubit=q.name).values):
            fit_results[q.name] = FitParameters(
                f01=qubit_freqs[q.name],
                f02_2=f02_2_values[i],
                anharmonicity=anharmonicity_values[i],
                peak_width=peak_width_values[i],
                iw_angle=fit.sel(qubit=q.name).iw_angle.values.__float__(),
                success=True,
            )
        else:
            fit_results[q.name] = FitParameters(
                f01=qubit_freqs[q.name],
                f02_2=np.nan,
                anharmonicity=np.nan,
                peak_width=np.nan,
                iw_angle=fit.sel(qubit=q.name).iw_angle.values.__float__(),
                success=False,
            )
    
    return fit, fit_results

