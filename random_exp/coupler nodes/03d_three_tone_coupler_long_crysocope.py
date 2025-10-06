"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the state.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the relevant flux biases in the state.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration
from qualibration_libs.data.processing import convert_IQ_to_V
from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray, load_dataset
from qualibration_libs.analysis.fitting import fit_oscillation
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings
from iqcc_calibration_tools.quam_config.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
from iqcc_calibration_tools.analysis.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
import iqcc_calibration_tools.analysis.cryoscope_tools as cryoscope_tools


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = None
    qubits: Optional[List[str]] = None
    num_averages: int = 2000
    frequency_span_in_mhz: float = 40
    frequency_step_in_mhz: float = 0.5
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    
    duration_in_ns: Optional[int] = 20000
    time_axis: Literal["linear", "log"] = "log"
    time_step_in_ns: Optional[int] = 1000000 # for linear time axis
    time_step_num: Optional[int] = 40 # for log time axis
    min_wait_time_in_ns: Optional[int] = 40
    
    flux_amp : float = 0.022
    
    control_drive_operation: Literal["x180_Square", "x180"] = "x180_Square"
    control_pulse_duration: int = 20 # clock cycles
    control_pulse_amplitude: float = 2.0
    
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    
    reset_type: Literal['active', 'thermal'] = "active"
    coupler_detuning_in_MHz: Optional[float] = 400
    wait_extra_time : Optional[bool] = True

node = QualibrationNode(name="03d_Three_Tone_Coupler_Long_Cryoscope", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)
qubit_pair_names = [qp.name for qp in qubit_pairs]

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    
# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

# Flux bias sweep
if node.parameters.time_axis == "linear":
    times = np.arange(node.parameters.min_wait_time_in_ns // 4, node.parameters.duration_in_ns // 4, node.parameters.time_step_in_ns // 4, dtype=np.int32)
elif node.parameters.time_axis == "log":
    times = np.logspace(np.log10(node.parameters.min_wait_time_in_ns // 4), np.log10(node.parameters.duration_in_ns // 4), node.parameters.time_step_num, dtype=np.int32)
    # Remove repetitions from times
    times = np.unique(times)
    
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

detuning = node.parameters.coupler_detuning_in_MHz * 1e6
coupler_IFs = {qp.name: qp.coupler.RF_frequency - detuning - qp.qubit_control.xy.opx_output.upconverter_frequency for qp in qubit_pairs}

with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubit_pairs)
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_stream_target = [declare_stream() for _ in range(num_qubit_pairs)]
    df = declare(int)  # QUA variable for the readout frequency
    t_delay = declare(int)  # QUA variable for delay time scan
    duration = node.parameters.duration_in_ns * u.ns
    
    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit_pairs[0].qubit_target)

    align()
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        for i, qp in enumerate(qubit_pairs):
            with for_(*from_array(df, dfs)):  # type: ignore
                with for_each_(t_delay, times):
                    # Qubit initialization
                    qubit_control = qp.qubit_control
                    qubit_target = qp.qubit_target

                    # Update the qubit frequency
                    qubit_control.xy.update_frequency(qubit_control.xy.intermediate_frequency)
                    if node.parameters.reset_type == "active":
                        active_reset_simple(qubit_control)
                        active_reset_simple(qubit_target)
                        qp.align()

                    else:
                        qubit_control.reset_qubit_thermal()
                        qubit_target.reset_qubit_thermal()
                        qp.align()
                    
                    if node.parameters.wait_extra_time:
                        qubit_control.xy.wait(node.parameters.duration_in_ns // 4)
                    qp.align()
                    #update the frequency of the control qubit
                    qubit_control.xy.update_frequency(df + coupler_IFs[qp.name])

                    # Qubit manipulation
                    # Apply saturation pulse to all qubits

                    qp.coupler.play("const", amplitude_scale=node.parameters.flux_amp / qp.coupler.operations["const"].amplitude, duration=t_delay+200)
                    qubit_control.xy.wait(t_delay)
                    qubit_control.xy.play(
                        node.parameters.control_drive_operation,
                            amplitude_scale=node.parameters.control_pulse_amplitude,
                            duration=node.parameters.control_pulse_duration
                        )
                    qp.align()

                    # qubit_target.xy.play("saturation",duration=1000)
                    qubit_target.xy.play("x180")

                    qp.align()
                    # Qubit readout
                    readout_state(qubit_target, state_target[i])
                    # save data
                    save(state_target[i], state_stream_target[i])
                
        # Measure sequentially
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_stream_target[i].buffer(len(times)).buffer(len(dfs)).buffer(num_qubit_pairs).average().save(f"state_target{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
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
    
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(multi_res_spec_vs_flux)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"time": times*4, "freq": dfs,  "qp": qubit_pair_names})
        # Convert IQ data into volts
        # ds = convert_IQ_to_V(ds, qubit_pairs)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + qp.coupler.RF_frequency - detuning for qp in qubit_pairs])
        ds = ds.assign_coords({"freq_full_control": (["qp", "freq"], RF_freq)})
        detuned_freq = np.array([dfs - detuning for qp in qubit_pairs])*1e-6
        ds = ds.assign_coords({"detunings": (["qp", "freq"], detuned_freq)})        
        ds.freq_full_control.attrs["long_name"] = "Frequency"
        ds.freq_full_control.attrs["units"] = "GHz"
        ds.detunings.attrs["long_name"] = "Detuning"
        ds.detunings.attrs["units"] = "MHz"

        # Remove the redundant dimension "qubit" from the dataset
        ds = ds.isel(qubit = 0).drop_dims("qubit", errors="ignore")

    # Add the dataset to the node
    node.results = {"ds": ds}
    



    # %% {Data_analysis}
    import xarray as xr
    # Extract frequency points and reshape data for analysis
    freqs = ds['freq'].values

    # Transpose to ensure ('qubit', 'time', 'freq') order for analysis
    stacked = 1-ds.transpose('qp', 'time', 'freq')

    # Fit Gaussian to each spectrum to find center frequencies
    center_freqs = xr.apply_ufunc(
        lambda states: cryoscope_tools.fit_gaussian(freqs, states),
        stacked,
        input_core_dims=[['freq']],
        output_core_dims=[[]],  # no dimensions left after fitting
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    ).rename({"state_target": "center_frequency"})

    # Add flux-induced frequency shift to center frequencies
    center_freqs = center_freqs.center_frequency - detuning

    # Calculate flux response from frequency shifts
    flux_response = np.sqrt(-1*center_freqs )
    flux_response_normalized = flux_response / flux_response.isel(time = slice(-5, None)).mean(dim = "time")
    # Store results in dataset
    ds['center_freqs'] = center_freqs
    ds['flux_response'] = flux_response
    ds['flux_response_normalized'] = flux_response_normalized


    # %% {Plot the flux response}

    from scipy.optimize import minimize
    from scipy.optimize import curve_fit

    def single_exp_decay(t, amp, tau):
        """Single exponential decay without offset
        
        Args:
            t (array): Time points
            amp (float): Amplitude of the decay
            tau (float): Time constant of the decay
            
        Returns:
            array: Exponential decay values
        """
        return amp * np.exp(-t/tau)


    def sequential_exp_fit(t, y, start_fractions, verbose=True):
        """
        Fit multiple exponentials sequentially by:
        1. First fit a constant term from the tail of the data
        2. Fit the longest time constant using the latter part of the data
        3. Subtract the fit
        4. Repeat for faster components
        
        Args:
            t (array): Time points in nanoseconds
            y (array): Data points (normalized amplitude)
            start_fractions (list): List of fractions (0 to 1) indicating where to start fitting each component
            verbose (bool): Whether to print detailed fitting information
            
        Returns:
            tuple: (components, a_dc, residual) where:
                - components: List of (amplitude, tau) pairs for each fitted component
                - a_dc: Fitted constant term
                - residual: Residual after subtracting all components
        """
        components = []  # List to store (amplitude, tau) pairs
        t_offset = t - t[0]  # Make time start at 0
        
        # Find the flat region in the tail by looking at local variance
        window = max(5, len(y) // 20)  # Window size by dividing signal into 20 equal pieces or at least 5 points
        rolling_var = np.array([np.var(y[i:i+window]) for i in range(len(y)-window)])
        # Find where variance drops below threshold, indicating flat region
        var_threshold = np.mean(rolling_var) * 0.1  # 10% of mean variance
        try:
            flat_start = np.where(rolling_var < var_threshold)[0][-1]
            # Use the flat region to estimate constant term
            a_dc = np.mean(y[flat_start:])
        except IndexError:
            print("No flat region found, using last point of the signal as constant term")
        
        a_dc = y[-1]
        

        if verbose:
            print(f"\nFitted constant term: {a_dc:.3e}")
        
        y_residual = y.copy() - a_dc
        
        for i, start_frac in enumerate(start_fractions):
            # Calculate start index for this component
            start_idx = int(len(t) * start_frac)
            if verbose:
                print(f"\nFitting component {i+1} using data from t = {t[start_idx]:.1f} ns (fraction: {start_frac:.3f})")
            
            # Fit current component
            try:
                # Initial guess for parameters
                p0 = [
                    y_residual[start_idx],  # amplitude
                    t_offset[start_idx] / 3  # tau
                ]
                
                # Set bounds for the fit
                bounds = (
                    [-np.inf, 0.1],  # lower bounds: amplitude can be negative, tau must be positive (0.1 ns is arbitrary)
                    [np.inf, np.inf]  # upper bounds
                )
                
                # Perform the fit on the current interval
                t_fit = t_offset[start_idx:]
                y_fit = y_residual[start_idx:]
                popt, _ = curve_fit(single_exp_decay, t_fit, y_fit, p0=p0, bounds=bounds)
                
                # Store the components
                amp, tau = popt
                components.append((amp, tau))
                if verbose:
                    print(f"Found component: amplitude = {amp:.3e}, tau = {tau:.3f} ns")
                
                # Subtract this component from the entire signal
                y_residual -= amp * np.exp(-t_offset/tau)
                
            except (RuntimeError, ValueError) as e:
                if verbose:
                    print(f"Warning: Fitting failed for component {i+1}: {e}")
                break
        
        return components, a_dc, y_residual


    def optimize_start_fractions(t, y, base_fractions, bounds_scale=0.5, verbose=True):
        """
        Optimize the start_fractions by minimizing the RMS between the data and the fitted sum 
        of exponentials using scipy.optimize.minimize.
        
        Args:
            t (array): Time points in nanoseconds
            y (array): Data points (normalized amplitude)
            base_fractions (list): Initial guess for start fractions
            bounds_scale (float): Scale factor for bounds around base fractions (0.5 means ±50%)
            
        Returns:
            tuple: (best_fractions, best_components, best_dc, best_rms)
        """
        
        def objective(x):
            """
            Objective function to minimize: RMS between the data and the fitted sum of 
            exponentials.
            """
            # Ensure fractions are ordered in descending order
            if not np.all(np.diff(x) < 0):
                return 1e6  # Return large value if constraint is violated
                    
            components, _, residual = sequential_exp_fit(t, y, x, verbose=verbose)
            if len(components) == len(base_fractions):
                current_rms = np.sqrt(np.mean(residual**2))
            else:
                current_rms = 1e6 # Return large value if fitting fails
                
            return current_rms
        
        # Define bounds for optimization
        bounds = []
        for base in base_fractions:
            min_val = base * (1 - bounds_scale)
            max_val = base * (1 + bounds_scale)
            bounds.append((min_val, max_val))
        
        print("\nOptimizing start_fractions using scipy.optimize.minimize...")
        print(f"Initial values: {[f'{f:.5f}' for f in base_fractions]}")
        print(f"Bounds: ±{bounds_scale*100}% around initial values")
        
        # Run optimization
        result = minimize(
            objective,
            x0=base_fractions,
            bounds=bounds,
            method='Nelder-Mead',  # This method works well for non-smooth functions
            options={'disp': True, 'maxiter': 200}
        )
        
        # Get final results
        if result.success:
            best_fractions = result.x
            components, a_dc, best_residual = sequential_exp_fit(t, y, best_fractions, verbose=False)
            best_rms = np.sqrt(np.mean(best_residual**2))
            
            print("\nOptimization successful!")
            print(f"Initial fractions: {[f'{f:.5f}' for f in base_fractions]}")
            print(f"Optimized fractions: {[f'{f:.5f}' for f in best_fractions]}")
            print(f"Final RMS: {best_rms:.3e}")
            print(f"Number of iterations: {result.nit}")
        else:
            print("\nOptimization failed. Using initial values.")
            best_fractions = base_fractions
            components, a_dc, best_residual = sequential_exp_fit(t, y, best_fractions, verbose=False)
            best_rms = np.sqrt(np.mean(best_residual**2))
        
        return result.success, best_fractions, components, a_dc, best_rms


    # %% {Fit the flux response}
    # node.parameters.fitting_base_fractions = [0.4, 0.15, 0.02]
    fit_results = {}
    for q in qubit_pairs:
        fit_results[q.name] = {}
        t_data = flux_response_normalized.sel(qp=q.name).time.values[2:]
        y_data = flux_response_normalized.sel(qp=q.name).values[2:]
        fit_successful, best_fractions, best_components, best_a_dc, best_rms = optimize_start_fractions(
            t_data, y_data, [0.8,0.2,0.1], bounds_scale=0.5
            )

        fit_results[q.name]["fit_successful"] = fit_successful
        fit_results[q.name]["best_fractions"] = best_fractions
        fit_results[q.name]["best_components"] = best_components
        fit_results[q.name]["best_a_dc"] = best_a_dc
        fit_results[q.name]["best_rms"] = best_rms

    node.results["fit_results"] = fit_results    


    # %% {Plotting}
    
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        ds.sel(qp=qp['qubit']).state_target.plot(
            ax=ax,
            y="detunings",
            x="time"
        )
        ax.plot(ds.time,1e-6*ds.center_freqs.sel(qp=qp['qubit']),'r',alpha=0.5)
        ax.set_title(qp["qubit"] )
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Detuning (MHz)")
    
    plt.tight_layout()
    plt.show()  
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        ds.flux_response_normalized.sel(qp=qp['qubit']).plot(ax=ax)
        ax.set_title(qp["qubit"] )
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Flux Response (normalized)")
    
    plt.tight_layout()
    plt.show()        
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        ds.flux_response_normalized.sel(qp=qp['qubit']).plot(ax=ax)
        ax.set_title(qp["qubit"] )
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Flux Response (normalized)")
        ax.set_xscale("log")
        
        if fit_results[qp["qubit"]]["fit_successful"]:
            best_a_dc = fit_results[qp["qubit"]]["best_a_dc"]
            t_offset = t_data - t_data[0]
            y_fit = np.ones_like(t_data, dtype=float) * best_a_dc  # Start with fitted constant
            fit_text = f'a_dc = {best_a_dc:.3f}\n'
            for i, (amp, tau) in enumerate(fit_results[qp["qubit"]]["best_components"]):
                y_fit += amp * np.exp(-t_offset/tau)
                fit_text += f'a{i+1} = {amp / best_a_dc:.5f}, τ{i+1} = {tau:.0f}ns\n'

            ax.plot(t_data, y_fit, color='r', label='Full Fit', linewidth=2) # Plot full fit
            ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=8)     
        # ax.set_ylim([0.95,1.02])      
    
    plt.tight_layout()
    plt.show()        
    # %%
    


    # %% {Update_state}

    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                if fit_results[qp.name]["fit_successful"]:
                    A_list = [component[0] / fit_results[qp.name]["best_a_dc"] for component in fit_results[qp.name]["best_components"]]
                    tau_list = [component[1] for component in fit_results[qp.name]["best_components"]]
                    A_c, tau_c, scale = cryoscope_tools.decompose_exp_sum_to_cascade(A=A_list, tau=tau_list, A_dc=1)
                    # qp.coupler.opx_output.exponential_filter = list(zip(A_c, tau_c))
                    # print("updated the exponential filter")
    # %% {Save_results}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()


# %%
