# %%
from typing import List
from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate import GraphParameters
from qualibrate import QualibrationGraph
from qualibrate import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = ["Q4"]


parameters = Parameters()
# reset_type = "active_simple" if len(parameters.qubits) == 1 else "thermal"
reset_type = "thermal"

g = QualibrationGraph(
    name="digital_filters_calibration_graph",
    parameters=parameters,
    nodes={
        "iq_blobs_no_filters": library.nodes["07_iq_blobs"].copy(
            name="iq_blobs_no_filters",
            reset_type=reset_type,
            ),
        "XY_Z_delay_no_filters": library.nodes["14b_XY_Z_delay_4nS"].copy(
            name="XY_Z_delay_no_filters",
            reset_type=reset_type,
            ),
        "ramsey_vs_flux_calibration_no_filters": library.nodes["09_ramsey_vs_flux_calibration"].copy(
            name="ramsey_vs_flux_calibration_no_filters",
            flux_span=0.005,
            flux_num=11,
            reset_type=reset_type,
            ),
        "cryoscope_qubit_spectroscopy": library.nodes["13b_cryoscope_qubit_spectroscopy"].copy(
            name="cryoscope_qubit_spectroscopy",
            num_shots=20,
            duration_in_ns=1200,
            time_axis="log",
            time_step_num=40,
            flux_amp=0.2,
            update_state=True,
            reset_type=reset_type,
            ),
        "iq_blobs_with_filters": library.nodes["07_iq_blobs"].copy(
            name="iq_blobs_with_filters",
            reset_type=reset_type,
            ),
        "XY_Z_delay_with_filters": library.nodes["14b_XY_Z_delay_4nS"].copy(
            name="XY_Z_delay_with_filters",
            reset_type=reset_type,
            ),
        "ramsey_vs_flux_calibration_with_filters": library.nodes["09_ramsey_vs_flux_calibration"].copy(
            name="ramsey_vs_flux_calibration_with_filters",
            flux_span=0.005,
            flux_num=11,
            reset_type=reset_type,
            ),
        "cryoscope_qubit_spectroscopy_with_filters": library.nodes["13b_cryoscope_qubit_spectroscopy"].copy(
            name="cryoscope_qubit_spectroscopy_with_filters",
            num_shots=20,
            duration_in_ns=1200,
            time_axis="log",
            time_step_num=40,
            flux_amp=0.2,
            update_state=False,
            reset_type=reset_type,
            ),
    },
    connectivity=[
        ("iq_blobs_no_filters", "XY_Z_delay_no_filters"),
        ("XY_Z_delay_no_filters", "ramsey_vs_flux_calibration_no_filters"),
        ("ramsey_vs_flux_calibration_no_filters", "cryoscope_qubit_spectroscopy"),
        ("cryoscope_qubit_spectroscopy", "iq_blobs_with_filters"),
        ("iq_blobs_with_filters", "XY_Z_delay_with_filters"),
        ("XY_Z_delay_with_filters", "ramsey_vs_flux_calibration_with_filters"),
        ("ramsey_vs_flux_calibration_with_filters", "cryoscope_qubit_spectroscopy_with_filters"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()
