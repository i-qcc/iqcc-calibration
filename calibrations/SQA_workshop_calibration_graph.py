# %%
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = ["qC1"]


parameters = Parameters()
reset_type = "active_simple" if len(parameters.qubits) == 1 else "thermal"


g = QualibrationGraph(
    name="SQA2025_users",
    parameters=parameters,
    nodes={
        "resonator_spectroscopy": library.nodes["02a_resonator_spectroscopy"].copy(
            name="resonator_spectroscopy",
            multiplexed=True,
            num_shots = 50,
            frequency_span_in_mhz = 20.0,
            frequency_step_in_mhz = 0.2,
        ),
        "qubit_spectroscopy": library.nodes["03a_qubit_spectroscopy"].copy(
            name="qubit_spectroscopy",
            multiplexed=True,
            num_shots = 50,
            frequency_span_in_mhz = 50.0,
            frequency_step_in_mhz = 0.25,
            ),
        "power_rabi": library.nodes["04b_power_rabi"].copy(
            name="power_rabi",
            multiplexed=True,
            num_shots = 30,
            min_amp_factor = 0.001,
            max_amp_factor = 1.99,
            amp_factor_step = 0.05,
            max_number_pulses_per_sweep = 1,
            ),
        "IQ_blobs": library.nodes["07_iq_blobs"].copy(
            name="IQ_blobs",
            multiplexed=True,
            reset_type=reset_type,
            num_shots = 200,
            ),
        "power_rabi_error_amplification_x180": library.nodes["04b_power_rabi"].copy(
            name="power_rabi_error_amplification_x180",
            multiplexed=True,
            max_number_pulses_per_sweep=100,
            min_amp_factor=0.98,
            max_amp_factor=1.02,
            amp_factor_step=0.002,
            use_state_discrimination=True,
            reset_type=reset_type,
        ),
        "power_rabi_error_amplification_x90": library.nodes["04b_power_rabi"].copy(
            name="power_rabi_error_amplification_x90",
            multiplexed=True,
            max_number_pulses_per_sweep=100,
            min_amp_factor=0.98,
            max_amp_factor=1.02,
            amp_factor_step=0.002,
            operation="x90",
            use_state_discrimination=True,
            reset_type=reset_type,
        ),
        "DRAG_calibration": library.nodes["10b_drag_calibration_180_minus_180"].copy(
            name="DRAG_calibration",
            multiplexed=True,
            amp_factor_step = 0.04,
            use_state_discrimination=True,
            reset_type=reset_type,
        ),
        "Randomized_benchmarking": library.nodes["11_single_qubit_randomized_benchmarking"].copy(
            name="Randomized_benchmarking",
            multiplexed=True,
            num_random_sequences = 40,
            num_shots = 100,
            max_circuit_depth = 1024,
            use_state_discrimination=True,
            reset_type=reset_type,
        ),
    },
    connectivity=[
        ("resonator_spectroscopy", "qubit_spectroscopy"),
        ("qubit_spectroscopy", "power_rabi"),
        ("power_rabi", "IQ_blobs"),
        ("IQ_blobs", "power_rabi_error_amplification_x180"),
        ("power_rabi_error_amplification_x180", "power_rabi_error_amplification_x90"),
        ("power_rabi_error_amplification_x90", "DRAG_calibration"),
        ("DRAG_calibration", "Randomized_benchmarking"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()
