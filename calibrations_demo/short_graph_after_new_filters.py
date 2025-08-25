# %%
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = ["Q3"]


parameters = Parameters()

g = QualibrationGraph(
    name="SQA2025_after_filters",
    parameters=parameters,
    nodes={
        "XY_Z_delay": library.nodes["14b_XY_Z_delay_4nS"].copy(
            name="XY_Z_delay",
        ),
        "IQ_blobs": library.nodes["07_iq_blobs"].copy(
            name="IQ_blobs"
            ),
        "ramsey_vs_flux_calibration": library.nodes["09_ramsey_vs_flux_calibration"].copy(
            name="ramsey_vs_flux_calibration",
        ),
    },
    connectivity=[
        ("IQ_blobs", "XY_Z_delay"),
        ("XY_Z_delay", "ramsey_vs_flux_calibration"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()
