# %%
"""
        QUBIT SPECTROSCOPY VERSUS FLUX
This sequence involves doing a qubit spectroscopy for several flux biases in order to exhibit the qubit frequency
versus flux response.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Identification of the approximate qubit frequency ("qubit_spectroscopy").

Before proceeding to the next node:
    - Update the qubit frequency, labeled as "f_01", in the state.
    - Update the relevant flux points in the state.
    - Save the current state by calling machine.save("quam")
"""
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List


class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 100
    dc_offset: float = 0.02
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    timeout: int = 100
    idle_time: int = 2000
    n_avg_for_fft: int = 2**20 + 1

node = QualibrationNode(
    name="99_1bit_SA_ramsey",
    parameters_class=Parameters
)

node.parameters = Parameters()
simulate = node.parameters.simulate

# from qm.qua import *
# from qm import SimulationConfig
# from qualang_tools.results import progress_counter, fetching_tool
# from qualang_tools.plot import interrupt_on_close
# from qualang_tools.loops import from_array
# from qualang_tools.units import unit
# from iqcc_calibration_tools.quam_config.components import QuAM
# from iqcc_calibration_tools.quam_config.macros import qua_declaration, active_reset
import xarray as xr
# import xrft
# import matplotlib.pyplot as plt
# import numpy as np
from scipy import signal

# %% ====================================================================
# CROSS-PSD ANALYSIS FUNCTIONS (Yan et al. 2012 Implementation)
# ====================================================================

"""
Cross-PSD Analysis Implementation based on Yan et al. 2012
Implements the interleaved cross-PSD method to eliminate statistical white noise floor
while preserving underlying 1/f noise signals in binary time series data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import xarray as xr

def cross_psd_yan_method(data_q, time_stamp_q, dt=None, num_segments=8):
    """
    Calculate Cross-PSD using Yan et al. interleaved method with segment averaging
    
    Parameters:
    -----------
    data_q : array-like
        Binary time series data (0s and 1s)
    time_stamp_q : array-like  
        Timestamps corresponding to data_q
    dt : float, optional
        Time step. If None, calculated from time_stamp_q
    num_segments : int, optional
        Number of segments to divide data into for averaging (default: 8)
        
    Returns:
    --------
    dict containing:
        - frequencies : frequency axis
        - cross_psd : averaged cross power spectral density
        - regular_psd : averaged regular PSD for comparison
        - white_noise_floor : estimated white noise floor
        - segment_results : individual segment results
        - statistics : statistical information
    """
    
    # Convert to numpy arrays
    data_q = np.array(data_q)
    time_stamp_q = np.array(time_stamp_q)
    
    # Calculate time step if not provided
    if dt is None:
        dt = np.mean(np.diff(time_stamp_q))
    
    N = len(data_q)
    
    # Ensure even number of points for interleaving
    if N % 2 == 1:
        data_q = data_q[:-1]
        time_stamp_q = time_stamp_q[:-1]
        N = len(data_q)
    
    # Calculate segment size
    segment_size = N // num_segments
    if segment_size < 4:  # Minimum segment size for meaningful analysis
        print(f"Warning: Data too short for {num_segments} segments. Using fewer segments.")
        num_segments = max(1, N // 4)
        segment_size = N // num_segments
    
    print(f"Dividing {N} data points into {num_segments} segments of ~{segment_size} points each")
    
    # Initialize arrays for averaging
    all_cross_psds = []
    all_regular_psds = []
    segment_results = []
    
    # Process each segment
    for seg in range(num_segments):
        start_idx = seg * segment_size
        end_idx = min((seg + 1) * segment_size, N)
        
        # Extract segment data
        seg_data = data_q[start_idx:end_idx]
        seg_time = time_stamp_q[start_idx:end_idx]
        
        # Ensure even number of points for interleaving
        if len(seg_data) % 2 == 1:
            seg_data = seg_data[:-1]
            seg_time = seg_time[:-1]
        
        seg_N = len(seg_data)
        
        # Split into interleaved series (Yan et al. method)
        # z'_n = z_{2n-1} and z''_n = z_{2n} (n = 1, ..., N/2)
        z_prime = seg_data[::2]  # Even indices (0, 2, 4, ...)
        z_double_prime = seg_data[1::2]  # Odd indices (1, 3, 5, ...)
        
        # Calculate FFTs
        Z_prime = np.fft.fft(z_prime)
        Z_double_prime = np.fft.fft(z_double_prime)
        
        # Frequency axis (up to Nyquist frequency)
        # When interleaving, effective sampling rate becomes 1/(2*dt)
        # Convert dt from nanoseconds to seconds for proper Hz scaling
        freqs = np.fft.fftfreq(seg_N//2, d=2*dt*1e-9)
        freqs = freqs[:seg_N//4 + 1]  # Only positive frequencies + DC
        
        # Cross-PSD calculation (Eq. 12 from Yan et al.)
        cross_psd = np.zeros_like(freqs, dtype=complex)
        
        # DC component (k=0)
        cross_psd[0] = (2*np.pi)**2 * 0.5 * Z_prime[0] * np.conj(Z_double_prime[0]) / (seg_N//2 * dt)
        
        # Non-DC components (k≠0)
        for k in range(1, len(freqs)):
            cross_psd[k] = (2*np.pi)**2 * Z_prime[k] * np.conj(Z_double_prime[k]) / (seg_N//2 * dt)
        
        # Regular PSD for comparison (Eq. 6 from Yan et al.)
        Z_full = np.fft.fft(seg_data)
        regular_psd = np.zeros_like(freqs)
        
        # DC component
        regular_psd[0] = (2*np.pi)**2 * 0.5 * np.abs(Z_full[0])**2 * dt**2 / (seg_N * dt)
        
        # Non-DC components  
        for k in range(1, len(freqs)):
            regular_psd[k] = (2*np.pi)**2 * np.abs(Z_full[k])**2 * dt**2 / (seg_N * dt)
        
        # Store segment results
        all_cross_psds.append(np.abs(cross_psd))
        all_regular_psds.append(regular_psd)
        
        segment_results.append({
            'segment': seg,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'length': seg_N,
            'cross_psd': np.abs(cross_psd),
            'regular_psd': regular_psd,
            'frequencies': freqs
        })
    
    # Average across segments (Eq. 7 from Yan et al.)
    # Find minimum frequency array length for consistent averaging
    min_freq_len = min(len(seg['frequencies']) for seg in segment_results)
    
    # Truncate all arrays to same length and average
    averaged_cross_psd = np.zeros(min_freq_len)
    averaged_regular_psd = np.zeros(min_freq_len)
    
    for seg_result in segment_results:
        averaged_cross_psd += seg_result['cross_psd'][:min_freq_len]
        averaged_regular_psd += seg_result['regular_psd'][:min_freq_len]
    
    averaged_cross_psd /= num_segments
    averaged_regular_psd /= num_segments
    
    # Use frequencies from first segment (they should all be the same)
    final_freqs = segment_results[0]['frequencies'][:min_freq_len]
    
    # White noise floor estimation (Eq. 8 from Yan et al.)
    p = np.mean(data_q)  # Probability of switching
    sigma_b_squared = p * (1 - p)  # Variance of Bernoulli process
    white_noise_floor = (2*np.pi)**2 * sigma_b_squared * dt
    
    return {
        'frequencies': final_freqs,
        'cross_psd': averaged_cross_psd,
        'regular_psd': averaged_regular_psd,
        'white_noise_floor': white_noise_floor,
        'segment_results': segment_results,
        'statistics': {
            'p': p,
            'sigma_b_squared': sigma_b_squared,
            'N': N,
            'dt': dt,
            'num_segments': num_segments,
            'segment_size': segment_size
        }
    }

def plot_cross_psd_results(results, qubit_name="Qubit"):
    """
    Plot Cross-PSD results with comparison to regular PSD
    
    Parameters:
    -----------
    results : dict
        Output from cross_psd_yan_method
    qubit_name : str
        Name for plot titles
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    freqs = results['frequencies']
    cross_psd = results['cross_psd']
    regular_psd = results['regular_psd']
    white_noise_floor = results['white_noise_floor']
    
    # Plot 1: Cross-PSD vs Regular PSD
    ax1.loglog(freqs[1:], cross_psd[1:], 'b-', label='Cross-PSD (Yan et al.)', linewidth=2)
    ax1.loglog(freqs[1:], regular_psd[1:], 'r--', label='Regular PSD', alpha=0.7)
    ax1.axhline(y=white_noise_floor, color='k', linestyle=':', 
                label=f'White noise floor = {white_noise_floor:.2e}')
    
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title(f'{qubit_name}: Cross-PSD vs Regular PSD')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Noise reduction factor
    noise_reduction = regular_psd[1:] / cross_psd[1:]
    ax2.loglog(freqs[1:], noise_reduction, 'g-', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Noise Reduction Factor')
    ax2.set_title(f'{qubit_name}: Noise Reduction (Regular PSD / Cross-PSD)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_1f_noise(cross_psd_results, freq_range=None):
    """
    Analyze 1/f noise characteristics from Cross-PSD results
    
    Parameters:
    -----------
    cross_psd_results : dict
        Output from cross_psd_yan_method
    freq_range : tuple, optional
        (f_min, f_max) frequency range for 1/f fitting
        
    Returns:
    --------
    dict containing 1/f noise parameters
    """
    
    freqs = cross_psd_results['frequencies']
    psd = cross_psd_results['cross_psd']
    
    # Remove DC component for fitting
    freqs_fit = freqs[1:]
    psd_fit = psd[1:]
    
    # Default frequency range for 1/f fitting
    if freq_range is None:
        # Focus on low frequencies where 1/f noise dominates
        f_min = freqs_fit[0]
        f_max = freqs_fit[len(freqs_fit)//4]  # Use first quarter of frequency range
    else:
        f_min, f_max = freq_range
    
    # Select frequency range for fitting
    mask = (freqs_fit >= f_min) & (freqs_fit <= f_max)
    freqs_fit_range = freqs_fit[mask]
    psd_fit_range = psd_fit[mask]
    
    # Fit 1/f noise: PSD = A / f^α
    # Linear fit in log space: log(PSD) = log(A) - α*log(f)
    log_freqs = np.log(freqs_fit_range)
    log_psd = np.log(psd_fit_range)
    
    # Linear regression
    coeffs = np.polyfit(log_freqs, log_psd, 1)
    alpha = -coeffs[0]  # 1/f exponent
    log_A = coeffs[1]  # Amplitude
    A = np.exp(log_A)
    
    # Calculate R-squared
    log_psd_fit = log_A - alpha * log_freqs
    ss_res = np.sum((log_psd - log_psd_fit)**2)
    ss_tot = np.sum((log_psd - np.mean(log_psd))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'alpha': alpha,
        'A': A,
        'r_squared': r_squared,
        'freq_range': (f_min, f_max),
        'fitted_psd': A / freqs_fit_range**alpha
    }

def analyze_ramsey_data_yan_style(data_q, time_stamp_q, qubit, idle_time, dc_offset, num_segments=16):
    """
    Complete Yan et al. style analysis of Ramsey data
    
    Parameters:
    -----------
    data_q : array-like
        Binary time series data
    time_stamp_q : array-like
        Timestamps
    qubit : qubit object
        Qubit object with physical parameters
    idle_time : float
        Ramsey idle time in nanoseconds
    dc_offset : float
        DC offset voltage
    num_segments : int
        Number of segments for averaging
        
    Returns:
    --------
    dict containing all analysis results
    """
    
    print(f"\n{'='*20} ANALYZING {qubit.name} {'='*20}")
    
    # ====================================================================
    # STEP 1: DATA PREPARATION AND CROSS-PSD ANALYSIS
    # ====================================================================
    
    print("\nSTEP 1: Data Preparation and Cross-PSD Analysis")
    print("-" * 50)
    
    # 1.1: Load raw data
    print("1.1: Loading raw data...")
    data_q = np.array(data_q)
    time_stamp_q = np.array(time_stamp_q)
    
    print(f"   - Data length: {len(data_q)} points")
    print(f"   - Time range: {time_stamp_q[0]:.3f} - {time_stamp_q[-1]:.3f} ns")
    print(f"   - Sampling rate: {1/np.mean(np.diff(time_stamp_q)):.2e} Hz")
    
    # 1.2: Divide data into segments
    print("\n1.2: Dividing data into segments...")
    segment_size = len(data_q) // num_segments
    print(f"   - Number of segments: {num_segments}")
    print(f"   - Segment size: ~{segment_size} points each")
    
    # 1.3: Apply Cross-PSD to each segment
    print("\n1.3: Applying Cross-PSD to each segment...")
    cross_psd_result = cross_psd_yan_method(data_q, time_stamp_q, num_segments=num_segments)
    
    # 1.4: Average Cross-PSD results
    print("1.4: Averaging Cross-PSD results across segments...")
    print(f"   - Final frequency range: {cross_psd_result['frequencies'][0]:.3f} - {cross_psd_result['frequencies'][-1]:.3f} Hz")
    print(f"   - Frequency resolution: {np.mean(np.diff(cross_psd_result['frequencies'])):.3f} Hz")
    print(f"   - White noise floor: {cross_psd_result['white_noise_floor']:.2e}")
    
    # ====================================================================
    # STEP 2: 1/f NOISE ANALYSIS
    # ====================================================================
    
    print("\nSTEP 2: 1/f Noise Parameter Extraction")
    print("-" * 50)
    
    noise_analysis = analyze_1f_noise(cross_psd_result)
    
    print(f"   - 1/f exponent (α): {noise_analysis['alpha']:.3f}")
    print(f"   - 1/f amplitude (A): {noise_analysis['A']:.2e}")
    print(f"   - Fit quality (R²): {noise_analysis['r_squared']:.3f}")
    print(f"   - Fit frequency range: {noise_analysis['freq_range'][0]:.3f} - {noise_analysis['freq_range'][1]:.3f} Hz")
    
    # ====================================================================
    # STEP 3: PHYSICAL UNIT CONVERSIONS
    # ====================================================================
    
    print("\nSTEP 3: Physical Unit Conversions")
    print("-" * 50)
    
    # Ramsey time and frequency sensitivity
    ramsey_time = idle_time * 1e-9  # Convert to seconds
    frequency_sensitivity = 1 / (2 * np.pi * ramsey_time) * 0.5
    
    print(f"   - Ramsey time: {ramsey_time*1e9:.1f} ns")
    print(f"   - Frequency sensitivity: {frequency_sensitivity:.2e} Hz/rad")
    
    # Convert to frequency noise density
    cross_psd_frequency = cross_psd_result['cross_psd'] * (frequency_sensitivity)**2
    
    # Convert to voltage noise density
    alpha = -qubit.freq_vs_flux_01_quad_term
    V0 = dc_offset
    voltage_sensitivity = 1 / (alpha * 2 * V0)
    cross_psd_voltage_density = cross_psd_frequency * (voltage_sensitivity)**2
    cross_psd_voltage_rms = np.sqrt(cross_psd_voltage_density) * 1e9  # Convert to nV/√Hz
    
    print(f"   - Flux sensitivity (α): {alpha:.2e} Hz/V²")
    print(f"   - Working point voltage: {V0:.3f} V")
    print(f"   - Voltage sensitivity: {voltage_sensitivity:.2e} V/Hz")
    print(f"   - Max voltage noise: {np.max(cross_psd_voltage_rms[1:]):.1f} nV/√Hz")
    
    # Store physical unit results
    cross_psd_result['frequency_psd'] = cross_psd_frequency
    cross_psd_result['voltage_psd_density'] = cross_psd_voltage_density
    cross_psd_result['voltage_psd_rms'] = cross_psd_voltage_rms
    cross_psd_result['physical_params'] = {
        'ramsey_time': ramsey_time,
        'frequency_sensitivity': frequency_sensitivity,
        'voltage_sensitivity': voltage_sensitivity,
        'alpha': alpha,
        'V0': V0
    }
    
    print(f"   - Analysis complete for {qubit.name}")
    
    return {
        'qubit_name': qubit.name,
        'cross_psd_results': cross_psd_result,
        'noise_analysis': noise_analysis,
        'frequencies': cross_psd_result['frequencies'],
        'cross_psd': cross_psd_result['cross_psd'],
        'regular_psd': cross_psd_result['regular_psd'],
        'frequency_psd': cross_psd_frequency,
        'voltage_psd_rms': cross_psd_voltage_rms,
        'white_noise_floor': cross_psd_result['white_noise_floor'],
        'summary': {
            'data_length': len(data_q),
            'sampling_rate': 1/cross_psd_result['statistics']['dt'],
            'switching_probability': cross_psd_result['statistics']['p'],
            'num_segments': cross_psd_result['statistics']['num_segments'],
            'segment_size': cross_psd_result['statistics']['segment_size'],
            '1f_exponent': noise_analysis['alpha'],
            '1f_amplitude': noise_analysis['A'],
            'fit_quality': noise_analysis['r_squared'],
            'ramsey_time_ns': ramsey_time*1e9,
            'frequency_sensitivity': frequency_sensitivity,
            'voltage_sensitivity': voltage_sensitivity,
            'max_voltage_noise': np.max(cross_psd_voltage_rms[1:])
        }
    }

# import matplotlib
# from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
# from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray
# from qualibration_libs.analysis.feature_detection import peaks_dips

from qualibrate import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration, readout_state, active_reset
from qualibration_libs.data.processing import convert_IQ_to_V
from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray, load_dataset
from qualibration_libs.analysis.fitting import fit_oscillation_decay_exp, oscillation_decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# matplotlib.use("TKAgg")


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None:
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

freqs_MHZ = np.arange(-2, 0, 25e-3)  # Integer values from -5e6 to 0 with step 50000

idle_time = node.parameters.idle_time  # Integer values from 20 to 1000 with step 100
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
dc = node.parameters.dc_offset
n_avg = node.parameters.num_averages
###################
# The QUA program #
###################

# %% program for finding optimal freq offset and idle time
with program() as find_optimal_freq_offset_and_idle_time:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    current_state = [declare(int) for _ in range(num_qubits)]
    init_state = [declare(int) for _ in range(num_qubits)]
    final_state = [declare(int) for _ in range(num_qubits)]    
    # freq = declare(int)  # QUA variable for the flux dc level
    phi = declare(fixed)
    t = declare(int)
    assign(t, idle_time >> 2)
    # dt = declare(fixed, 1e-9)
    freq_MHZ = declare(fixed)

    for i, qubit in enumerate(qubits):
        machine.initialize_qpu(flux_point=flux_point, target=qubit)            
        wait(1000)
        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            
            with for_(*from_array(freq_MHZ, freqs_MHZ)):        
                assign(phi, Cast.mul_fixed_by_int(freq_MHZ * 1e-3, 4 * t))
                # Ramsey sequence
                qubit.reset_qubit_active()
                # qubit.reset_qubit_active_simple()
                qubit.align()
                with strict_timing_():
                    qubit.xy.play("x90")
                    qubit.xy.frame_rotation_2pi(phi)
                    qubit.xy.wait(t + 1)
                    qubit.z.wait(duration=qubit.xy.operations["x90"].length // 4)
                    qubit.z.play(
                        "const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=t
                    )
                    qubit.xy.play("x90")
                
                # Measure the state of the resonators
                qubit.readout_state(current_state[i])
                # assign(state[i], init_state[i] ^ current_state[i])
                assign(state[i], current_state[i])
                assign(init_state[i], current_state[i])
                save(state[i], state_st[i])
                qubit.wait(500)
    
                reset_frame(qubit.xy.name)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            state_st[i].buffer(len(freqs_MHZ)).average().save(f"state{i + 1}")



# %%

###########################
# Run or Simulate Program #
###########################
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, find_optimal_freq_offset_and_idle_time, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    
else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(find_optimal_freq_offset_and_idle_time)
        for i in range(num_qubits):
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_avg)



# %%
if not simulate:
    handles = job.result_handles
    ds = fetch_results_as_xarray(handles, qubits, {"freq": freqs_MHZ})

    node.results = {}
    node.results['ds'] = ds

# %%
   
opt_freq = {}
if not simulate:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        opt_freq[qubit['qubit']] = np.abs(ds.sel(qubit = qubit['qubit']).state-0.5).idxmin('freq')
        ds.sel(qubit = qubit['qubit']).state.plot(ax =ax)
        ax.axhline(0.5, color='k')
        ax.plot(opt_freq[qubit['qubit']], 0.5, 'o')
        ax.set_title(qubit['qubit'])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('State')
    grid.fig.suptitle('Avg state vs. detuning')
    plt.tight_layout()
    plt.show()
    node.results['figure_raw'] = grid.fig
# %%
n_avg = node.parameters.n_avg_for_fft

phis = {qubit.name: (opt_freq[qubit.name].values * 1e-3 * idle_time) for qubit in qubits}
phis
# %% create program for T2 spectoscopy
with program() as Ramsey_noise_spec:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    current_state = [declare(int) for _ in range(num_qubits)]
    init_state = [declare(int) for _ in range(num_qubits)]
    final_state = [declare(int) for _ in range(num_qubits)]
    t = declare(int)
    phi = declare(fixed)
    assign(t, idle_time >> 2)
    
    for i, qubit in enumerate(qubits):
        machine.initialize_qpu(flux_point=flux_point, target=qubit)                  
        wait(1000)
        # update_frequency(qubit.xy.name, int(opt_freq[qubit.name]) + qubit.xy.intermediate_frequency)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            
            assign(phi, phis[qubit.name])
            # Ramsey sequence
            qubit.reset_qubit_active()
            qubit.align()
            with strict_timing_():
                qubit.xy.play("x90",  timestamp_stream=f'time_stamp{i+1}')
                qubit.xy.frame_rotation_2pi(phi)
                qubit.xy.wait(t + 1)
                qubit.z.wait(duration=qubit.xy.operations["x90"].length // 4)
                qubit.z.play(
                    "const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=t
                )
                qubit.xy.play("x90")
            
            # Measure the state of the resonators
            qubit.readout_state(current_state[i])
            # assign(state[i], init_state[i] ^ current_state[i])
            assign(state[i], current_state[i])
            assign(init_state[i], current_state[i])
            save(state[i], state_st[i])
            # qubit.wait(500)            
            reset_frame(qubit.xy.name)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            state_st[i].buffer(n_avg).save(f"state{i + 1}")


# %% 
###########################
# Run or Simulate Program #
###########################
simulate = node.parameters.simulate

if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, Ramsey_noise_spec, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    
else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(Ramsey_noise_spec)
        for i in range(num_qubits):
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_avg)


# %%
if not simulate:
    def extract_string(input_string):
        # Find the index of the first occurrence of a digit in the input string
        index = next((i for i, c in enumerate(input_string) if c.isdigit()), None)

        if index is not None:
            # Extract the substring from the start of the input string to the index
            extracted_string = input_string[:index]
            return extracted_string
        else:
            return None
        
    stream_handles = job.result_handles.keys()
    meas_vars = list(set([extract_string(handle) for handle in stream_handles if extract_string(handle) is not None]))
    meas_vars = meas_vars[::-1]
    values = np.array(
        [
        np.array([job.result_handles.get(f"state{i + 1}").fetch_all() for i, qubit in enumerate(qubits)]),
    np.array(np.array( [job.result_handles.get(f"time_stamp{i + 1}").fetch_all() for i, qubit in enumerate(qubits)]).tolist()).squeeze(-1),
    ]
        
        )

    if np.array(values).shape[-1] == 1:
        values = np.array(values).squeeze(axis=-1)

    measurement_axis = {"n": np.arange(0,n_avg)}
        
    measurement_axis["qubit"] = [qubit.name for qubit in qubits]
    measurement_axis = {key: measurement_axis[key] for key in reversed(measurement_axis.keys())}


    ds = xr.Dataset(
        {f"{meas_var}": ([key for key in measurement_axis.keys()], values[i]) for i, meas_var in enumerate(meas_vars)},
        coords=measurement_axis,
    )
    ds['time_stamp'] = ds['time_stamp']*4

    node.results['ds_final'] = ds
# %%


# %%
if not simulate:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds.sel(qubit = qubit['qubit']).state.plot.hist(bins=3,ax=ax)
        ax.set_title(qubit['qubit'])
        ax.set_xlabel('State')
        ax.set_ylabel('Counts')
    grid.fig.suptitle('Histogram of qubit states')
    plt.tight_layout()
    plt.show()
    node.results['figure_bins'] = grid.fig

# %% ====================================================================
# YAN ET AL. STYLE NOISE ANALYSIS
# ====================================================================

if not simulate:
    print("\n" + "="*80)
    print("YAN ET AL. STYLE NOISE ANALYSIS")
    print("="*80)
    
    # Initialize results storage
    analysis_results = {}
    
    # Analyze each qubit
    for qubit in qubits:
        data_q = ds.state.sel(qubit=qubit.name).values
        time_stamp_q = ds.time_stamp.sel(qubit=qubit.name).values
        
        # Single function call for complete analysis
        analysis_results[qubit.name] = analyze_ramsey_data_yan_style(
            data_q, time_stamp_q, qubit, idle_time, dc, num_segments=16
        )
    
    # Store results
    node.results['analysis_results'] = analysis_results
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("YAN ET AL. STYLE NOISE ANALYSIS SUMMARY")
    print("="*80)
    
    for qubit_name, results in analysis_results.items():
        summary = results['summary']
        print(f"\n{qubit_name}:")
        print(f"  Data Statistics:")
        print(f"    - Data points: {summary['data_length']}")
        print(f"    - Segments: {summary['num_segments']}")
        print(f"    - Segment size: {summary['segment_size']} points")
        print(f"    - Switching probability: {summary['switching_probability']:.3f}")
        
        print(f"  1/f Noise Parameters:")
        print(f"    - Exponent (α): {summary['1f_exponent']:.3f}")
        print(f"    - Amplitude (A): {summary['1f_amplitude']:.2e}")
        print(f"    - Fit quality (R²): {summary['fit_quality']:.3f}")
        
        print(f"  Physical Parameters:")
        print(f"    - Ramsey time: {summary['ramsey_time_ns']:.1f} ns")
        print(f"    - Frequency sensitivity: {summary['frequency_sensitivity']:.2e} Hz/rad")
        print(f"    - Voltage sensitivity: {summary['voltage_sensitivity']:.2e} V/Hz")
        print(f"    - Max voltage noise: {summary['max_voltage_noise']:.1f} nV/√Hz")
    
    print(f"\nAnalysis complete! Results stored in node.results['analysis_results']")
    print("="*80)

# %% ====================================================================
# QUBIT GRID PLOTTING
# ====================================================================

if not simulate:
    print("\n" + "="*80)
    print("CREATING QUBIT GRID PLOTS")
    print("="*80)
    
    # Create xarray datasets for plotting
    qubit_names = list(analysis_results.keys())
    
    # Check if all qubits have the same frequency axis
    freq_lengths = [len(analysis_results[q]['frequencies']) for q in qubit_names]
    if len(set(freq_lengths)) > 1:
        print(f"Warning: Different qubits have different frequency axis lengths: {freq_lengths}")
        print("Using the minimum length for all qubits...")
        min_freq_len = min(freq_lengths)
        frequencies = analysis_results[qubit_names[0]]['frequencies'][:min_freq_len]
    else:
        frequencies = analysis_results[qubit_names[0]]['frequencies']
    
    # Cross-PSD data - truncate to same length if needed
    cross_psd_data = np.array([analysis_results[q]['cross_psd'][:len(frequencies)] for q in qubit_names])
    regular_psd_data = np.array([analysis_results[q]['regular_psd'][:len(frequencies)] for q in qubit_names])
    frequency_psd_data = np.array([analysis_results[q]['frequency_psd'][:len(frequencies)] for q in qubit_names])
    voltage_psd_data = np.array([analysis_results[q]['voltage_psd_rms'][:len(frequencies)] for q in qubit_names])
    
    # Create datasets
    ds_cross_psd = xr.Dataset({
        'cross_psd': (['qubit', 'frequency'], cross_psd_data),
        'regular_psd': (['qubit', 'frequency'], regular_psd_data)
    }, coords={
        'qubit': qubit_names,
        'frequency': frequencies
    })
    
    ds_physical = xr.Dataset({
        'frequency_psd': (['qubit', 'frequency'], frequency_psd_data),
        'voltage_psd': (['qubit', 'frequency'], voltage_psd_data)
    }, coords={
        'qubit': qubit_names,
        'frequency': frequencies
    })
    
    # Plot 1: Cross-PSD vs Regular PSD
    print("Creating Cross-PSD comparison plots...")
    grid_cross = QubitGrid(ds_cross_psd, [q.grid_location for q in qubits])
    
    for ax, qubit in grid_iter(grid_cross):
        qubit_name = qubit['qubit']
        freqs = ds_cross_psd.frequency.values[1:]  # Skip DC
        cross_psd = ds_cross_psd.cross_psd.sel(qubit=qubit_name).values[1:]
        regular_psd = ds_cross_psd.regular_psd.sel(qubit=qubit_name).values[1:]
        white_noise_floor = analysis_results[qubit_name]['white_noise_floor']
        
        ax.loglog(freqs, cross_psd, 'b-', label='Cross-PSD (Yan et al.)', linewidth=2)
        ax.loglog(freqs, regular_psd, 'r--', label='Regular PSD', alpha=0.7)
        ax.axhline(y=white_noise_floor, color='k', linestyle=':', 
                   label=f'White noise floor = {white_noise_floor:.2e}')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title(f'{qubit_name}: Cross-PSD vs Regular PSD')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    grid_cross.fig.suptitle('Cross-PSD Analysis (Yan et al. Method)')
    plt.tight_layout()
    plt.show()
    node.results['figure_cross_psd_grid'] = grid_cross.fig
    
    # Plot 2: Physical Units
    print("Creating physical unit plots...")
    grid_physical = QubitGrid(ds_physical, [q.grid_location for q in qubits])
    
    for ax, qubit in grid_iter(grid_physical):
        qubit_name = qubit['qubit']
        freqs = ds_physical.frequency.values[1:]  # Skip DC
        frequency_psd = ds_physical.frequency_psd.sel(qubit=qubit_name).values[1:]
        voltage_psd = ds_physical.voltage_psd.sel(qubit=qubit_name).values[1:]
        
        ax.loglog(freqs, frequency_psd, 'b-', label='Frequency Noise', linewidth=2)
        ax2 = ax.twinx()
        ax2.loglog(freqs, voltage_psd, 'r-', label='Voltage Noise', linewidth=2)
        ax.set_xlim([1,1e3])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Frequency Noise Density (Hz²/Hz)', color='b')
        ax2.set_ylabel('Voltage Noise (nV/√Hz)', color='r')
        ax.set_title(f'{qubit_name}: Physical Units')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    grid_physical.fig.suptitle('Physical Unit Analysis')
    plt.tight_layout()
    plt.show()
    node.results['figure_physical_units_grid'] = grid_physical.fig
    
    print("All plots created successfully!")

node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
# node.save()
# %%

# %%
