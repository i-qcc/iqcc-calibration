# %% {Imports}
import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from calibration_utils.spin_echo_sl import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot


# %% {Description}
description = """ T2 SL MEASUREMENT
The sequence consists in playing an SL sequence (y90 - SL(x,t) - -y90 - measurement) for 
different idle times.
"""
node = QualibrationNode[Parameters, Quam](name="06c_spin_locking", description=description, parameters=Parameters())

# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["Q3"]
    pass

# Instantiate the QUAM class from the state file
node.machine = Quam.load()

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # The number of averages
    # Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
    idle_times = get_idle_times_in_clock_cycles(node.parameters)
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(8 * idle_times, attrs={"long_name": "idle time", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        shot = declare(int)
        t = declare(int)

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            for i, qubit in multiplexed_qubits.items():
                with for_(shot, 0, shot < n_avg, shot + 1):
                    save(shot, n_st)
                    with for_each_(t, idle_times):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            reset_frame(qubit.xy.name)
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        # Qubit manipulation
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("-y90")
                            qubit.xy_SL.play("x180_BlackmanIntegralPulse_Rise")
                            qubit.xy_SL.play("x180_Square",duration = 12+2*t, amplitude_scale=1.0)
                            qubit.xy_SL.play("x180_BlackmanIntegralPulse_Fall")
                            qubit.xy.play("-y90")
                            qubit.align()
                        align()
                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            # Measure the state of the resonators
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                # save data
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}

plt.grid(True)
plt.show()
# %%
