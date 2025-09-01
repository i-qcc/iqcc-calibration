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
reset_type = "active" if len(parameters.qubits) == 1 else "thermal"


g = QualibrationGraph(
    name="Retuning Graph 1Q",
    parameters=parameters,
    nodes={
        "resonator_spectroscopy": library.nodes["02a_resonator_spectroscopy"].copy(
            name="resonator_spectroscopy",
            multiplexed=True,
            frequency_span_in_mhz=20,
        ),
        "resonator_spectroscopy_vs_flux": library.nodes["02b_resonator_spectroscopy_vs_flux"].copy(
            name="resonator_spectroscopy_vs_flux", min_flux_offset_in_v=-0.2, max_flux_offset_in_v=0.2
        ),
        "qubit_spectroscopy": library.nodes["03a_qubit_spectroscopy"].copy(
            name="qubit_spectroscopy",
            frequency_span_in_mhz=60,
            multiplexed=True,
        ),
        "power_rabi": library.nodes["04b_power_rabi"].copy(
            name="power_rabi",
            multiplexed=True,
        ),
        "readout_frequency_optimization": library.nodes["08a_readout_frequency_optimization"].copy(
            name="readout_frequency_optimization", multiplexed=True
        ),
        "IQ_blobs": library.nodes["07_iq_blobs"].copy(
            name="IQ_blobs",
            multiplexed=True,
            reset_type=reset_type,
        ),
        "ramsey_vs_flux_calibration": library.nodes["09_ramsey_vs_flux_calibration"].copy(
            name="ramsey_vs_flux_calibration",
            flux_span=0.005,
            flux_num=11,
            reset_type=reset_type,
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
            use_state_discrimination=True,
            reset_type=reset_type,
        ),
        "Randomized_benchmarking": library.nodes["11_single_qubit_randomized_benchmarking"].copy(
            name="Randomized_benchmarking",
            use_state_discrimination=True,
            reset_type=reset_type,
        ),
    },
    connectivity=[
        ("resonator_spectroscopy", "resonator_spectroscopy_vs_flux"),
        ("resonator_spectroscopy_vs_flux", "qubit_spectroscopy"),
        ("qubit_spectroscopy", "power_rabi"),
        ("power_rabi", "readout_frequency_optimization"),
        ("readout_frequency_optimization", "IQ_blobs"),
        ("IQ_blobs", "ramsey_vs_flux_calibration"),
        ("ramsey_vs_flux_calibration", "power_rabi_error_amplification_x180"),
        ("power_rabi_error_amplification_x180", "power_rabi_error_amplification_x90"),
        ("power_rabi_error_amplification_x90", "DRAG_calibration"),
        ("DRAG_calibration", "Randomized_benchmarking"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()
