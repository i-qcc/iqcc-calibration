# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam
from calibration_utils.iq_blobs_raw_adc import Parameters, plot_raw_adc_traces
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
RAW ADC
Collects raw ADC traces (Igraw, Qgraw, Ieraw, Qeraw) from input1 real and image.
Measures |g> and |e> states N times. No demodulation or post-processing.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="07b_iq_blobs_raw_adc",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """
    Allow the user to locally set the node parameters for debugging purposes, or
    execution in the Python IDE.
    """
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["Q1"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Create the sweep axes and generate the QUA program from the pulse sequence and the
    node parameters.
    """
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_runs = node.parameters.num_shots  # Number of runs
    operation = node.parameters.operation
    readout_length = qubits[0].resonator.operations[operation].length
    # Register the sweep axes: n_runs and readout_time for raw ADC traces
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}),
        "readout_time": xr.DataArray(
            np.arange(0, readout_length, 1),
            attrs={"long_name": "readout time", "units": "ns"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()
        adc_g_st = [declare_stream(adc_trace=True) for _ in range(num_qubits)]
        adc_e_st = [declare_stream(adc_trace=True) for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)

                # Ground state iq blobs for all qubits
                for i, qubit in multiplexed_qubits.items():
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                align()
                for i, qubit in multiplexed_qubits.items():
                    reset_if_phase(qubit.resonator.name)
                    qubit.resonator.measure(operation, stream=adc_g_st[i])
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                align()

                # Excited state iq blobs for all qubits
                for i, qubit in multiplexed_qubits.items():
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                align()
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x180")
                    qubit.align()
                    reset_if_phase(qubit.resonator.name)
                    qubit.resonator.measure(operation, stream=adc_e_st[i])
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                stream_g = adc_g_st[i].input1()
                stream_e = adc_e_st[i].input1()
                stream_g.real().buffer(n_runs).save(f"Igraw{i + 1}")
                stream_g.image().buffer(n_runs).save(f"Qgraw{i + 1}")
                stream_e.real().buffer(n_runs).save(f"Ieraw{i + 1}")
                stream_e.image().buffer(n_runs).save(f"Qeraw{i + 1}")


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


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw".
    """
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            pass
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset (no demodulation)
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


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot raw ADC traces (Igraw, Qgraw, Ieraw, Qeraw) vs readout time. No demodulation."""
    fig = plot_raw_adc_traces(node.results["ds_raw"], node.namespace["qubits"])
    node.add_node_info_subtitle(fig)
    plt.show()
    node.results["figures"] = {"raw_adc_traces": fig}


# %% {Save_results}
# @node.run_action()
# def save_results(node: QualibrationNode[Parameters, Quam]):
#     node.save()
# %% 