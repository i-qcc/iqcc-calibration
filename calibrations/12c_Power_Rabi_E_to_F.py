# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from calibration_utils.power_rabi_e_to_f import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from quam.components import pulses


# %% {Node initialisation}
description = """
        POWER RABI E->F
This sequence involves executing the E->F transition pulse (EF_x180) with varying amplitudes
and measuring the state of the resonator. The qubit is first prepared in the |e> state by
applying an x180 pulse, then the E->F transition pulse is applied at the frequency f_01 - anharmonicity.
By fitting the Rabi oscillations, the optimal amplitude for the E->F pi pulse is determined.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port).
    - Having found the rough qubit frequency and pi pulse duration (rabi_chevron_duration or time_rabi).
    - Having calibrated the E->F transition (node 12a).
    - Having optimized the GEF readout frequency (node 12b).
    - Set the qubit frequency, desired pi pulse duration and rough pi pulse amplitude in the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the qubit pulse amplitude (pi_amp) in the state.
    - Save the current state by calling machine.save("quam")
"""


# %% {Imports}
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from quam.components import pulses
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration
from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray
from qualibration_libs.analysis import fit_oscillation, oscillation
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np



# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 200
    operation: str = "EF_x180"
    min_amp_factor: float = 0.0
    max_amp_factor: float = 2.0
    amp_factor_step: float = 0.02
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    timeout: int = 100

node = QualibrationNode(name="12c_Power_Rabi_E_to_F", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
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
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
        
num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # The number of averages
    operation = node.parameters.operation  # The qubit operation to play
    
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    
    flux_point = node.parameters.flux_point_joint_or_independent

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, _, n_st = node.machine.declare_qua_variables()
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            if flux_point == "independent":
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
            elif flux_point == "joint":
                node.machine.initialize_qpu(target=multiplexed_qubits[0])
            else:
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
            align()

            # Update readout frequency to GEF frequency if available
            for i, qubit in multiplexed_qubits.items():
                if hasattr(qubit.resonator, "GEF_frequency_shift"):
                    update_frequency(
                        qubit.resonator.name,
                        qubit.resonator.intermediate_frequency + qubit.resonator.GEF_frequency_shift,
                    )

            n = declare(int)  # QUA variable for the number of averages
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amps)):
                    for i, qubit in multiplexed_qubits.items():
                        # Reset the qubit frequency
                        update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                        qubit.align()
                        
                        # Drive the qubit to the excited state |e>
                        qubit.xy.play("x180")
                        qubit.align()
                        
                        # Update the qubit frequency to scan around the expected f_01 - anharmonicity (E->F transition)
                        update_frequency(
                            qubit.xy.name, qubit.xy.intermediate_frequency - qubit.anharmonicity
                        )
                        qubit.align()
                        
                        # Play the E->F transition pulse with varying amplitude
                        qubit.xy.play(operation, amplitude_scale=a)
                        qubit.align()
                        
                        # Measure the resonator
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(amps)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amps)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, power_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    node.add_node_info_subtitle(fig_raw_fit)
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_result = node.results["fit_results"][q.name]
            # Update the E->F pi pulse amplitude
            q.xy.operations["EF_x180"].amplitude = fit_result["Pi_amplitude"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
