# import logging
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from calibration_utils.cryoscope import cryoscope_tools


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    frequency: float
    fwhm: float
    iw_angle: float
    saturation_amp: float
    x180_amp: float
    success: bool


# def log_fitted_results(fit_results: Dict, log_callable=None):
#     """
#     Logs the node-specific fitted results for all qubits from the fit results

#     Parameters:
#     -----------
#     fit_results : dict
#         Dictionary containing the fitted results for all qubits.
#     logger : logging.Logger, optional
#         Logger for logging the fitted results. If None, a default logger is used.

#     """
#     if log_callable is None:
#         log_callable = logging.getLogger(__name__).info
#     for q in fit_results.keys():
#         s_qubit = f"Results for qubit {q}: "
#         s_freq = f"\tQubit frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
#         s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
#         s_angle = f"The integration weight angle: {fit_results[q]['iw_angle']:.3f} rad\n "
#         s_saturation = f"To get the desired FWHM, the saturation amplitude is updated to: {1e3 * fit_results[q]['saturation_amp']:.1f} mV | "
#         s_x180 = f"To get the desired x180 gate, the x180 amplitude is updated to: {1e3 * fit_results[q]['x180_amp']:.1f} mV\n "
#         if fit_results[q]["success"]:
#             s_qubit += " SUCCESS!\n"
#         else:
#             s_qubit += " FAIL!\n"
#         log_callable(s_qubit + s_freq + s_fwhm + s_freq + s_angle + s_saturation + s_x180)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
    # ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    # ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}

    ds = ds.assign_coords(
        {
            "full_freq": (  # Full frequency including RF and flux-induced shifts
                ["qubit", "detuning"],
                np.array([ds.detuning + q.xy.RF_frequency + q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in node.namespace["qubits"]]),
            ),
            "full_detuning": (  # Frequency shift due to flux
                ["qubit", "detuning"],
                np.array([ds.detuning + q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in node.namespace["qubits"]]),
            ),
            "flux": (  # Flux at given detuning
                ["qubit", "detuning"],
                np.array([np.sqrt(ds.detuning / q.freq_vs_flux_01_quad_term + node.parameters.flux_amp**2) for q in node.namespace["qubits"]]),
            )
        }
    )
    ds.full_freq.attrs["long_name"] = "Full Frequency"
    ds.full_freq.attrs["units"] = "Hz"
    ds.full_detuning.attrs["long_name"] = "Full Detuning"
    ds.full_detuning.attrs["units"] = "Hz"
    ds.flux.attrs["long_name"] = "Flux"
    ds.flux.attrs["units"] = "V"

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
    # Extract frequency points and reshape data for analysis
    # freqs = ds['detuning'].values

    # Transpose to ensure ('qubit', 'time', 'freq') order for analysis
    stacked = ds.transpose('qubit', 'time', 'detuning')

    # Fit Gaussian to each spectrum to find center frequencies
    center_freqs = xr.apply_ufunc(
        lambda states: cryoscope_tools.fit_gaussian(ds['detuning'].values, states),
        stacked,
        input_core_dims=[['detuning']],
        output_core_dims=[[]],  # no dimensions left after fitting
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    ).rename({"state": "center_frequency"})

    # Add flux-induced frequency shift to center frequencies
    center_freqs = center_freqs.center_frequency + np.array([q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 * np.ones_like(ds.time) for q in node.namespace["qubits"]])

    # Calculate flux response from frequency shifts
    flux_response = np.sqrt(center_freqs / xr.DataArray([q.freq_vs_flux_01_quad_term for q in node.namespace["qubits"]], coords={"qubit": center_freqs.qubit}, dims=["qubit"]))

    # Store results in dataset
    ds = xr.Dataset({"center_freqs": center_freqs, "flux_response": flux_response})

    # Perform exponential fitting for each qubit
    fit_results = {}
    for q in node.namespace["qubits"]:
        fit_results[q.name] = {}
        t_data = flux_response.sel(qubit=q.name).time.values
        y_data = flux_response.sel(qubit=q.name).values
        fit_successful, best_fractions, best_components, best_a_dc, best_rms = cryoscope_tools.optimize_start_fractions(
            t_data, y_data, node.parameters.fitting_base_fractions, bounds_scale=0.5
            )

        fit_results[q.name]["fit_successful"] = fit_successful
        fit_results[q.name]["best_fractions"] = best_fractions
        fit_results[q.name]["best_components"] = best_components
        fit_results[q.name]["best_a_dc"] = best_a_dc
        fit_results[q.name]["best_rms"] = best_rms

    node.results["fit_results"] = fit_results

    return ds, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    limits = [instrument_limits(q.xy) for q in node.namespace["qubits"]]
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    # Get the fitted resonator frequency
    full_freq = np.array([q.xy.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + full_freq
    fit = fit.assign({"res_freq": ("qubit", res_freq.data)})
    fit.res_freq.attrs = {"long_name": "qubit xy frequency", "units": "Hz"}
    # Get the fitted FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign({"fwhm": fwhm})
    fit.fwhm.attrs = {"long_name": "qubit fwhm", "units": "Hz"}
    # Get optimum iw angle
    prev_angles = np.array(
        [q.resonator.operations["readout"].integration_weights_angle for q in node.namespace["qubits"]]
    )
    fit = fit.assign({"iw_angle": (prev_angles + fit.iw_angle) % (2 * np.pi)})
    fit.iw_angle.attrs = {"long_name": "integration weight angle", "units": "rad"}
    # Get saturation amplitude
    x180_length = np.array([q.xy.operations["x180"].length * 1e-9 for q in node.namespace["qubits"]])
    used_amp = np.array(
        [
            q.xy.operations["saturation"].amplitude * node.parameters.operation_amplitude_factor
            for q in node.namespace["qubits"]
        ]
    )
    factor_cw = node.parameters.target_peak_width / fit.width
    fit = fit.assign({"saturation_amplitude": factor_cw * used_amp / node.parameters.operation_amplitude_factor})
    # get expected x180 amplitude
    factor_x180 = np.pi / (fit.width * x180_length)
    fit = fit.assign({"x180_amplitude": factor_x180 * used_amp})

    # Assess whether the fit was successful or not
    freq_success = np.abs(res_freq) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    fwhm_success = np.abs(fwhm) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    saturation_amp_success = np.abs(fit.saturation_amplitude) < limits[0].max_wf_amplitude
    # x180amp_success = np.abs(fit.x180_amplitude.data) < limits[0].max_x180_wf_amplitude
    success_criteria = freq_success & fwhm_success & saturation_amp_success
    fit = fit.assign({"success": success_criteria})

    fit_results = {
        q: FitParameters(
            frequency=fit.sel(qubit=q).res_freq.values.__float__(),
            fwhm=fit.sel(qubit=q).fwhm.values.__float__(),
            iw_angle=fit.sel(qubit=q).iw_angle.values.__float__(),
            saturation_amp=fit.sel(qubit=q).saturation_amplitude.values.__float__(),
            x180_amp=fit.sel(qubit=q).x180_amplitude.values.__float__(),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
