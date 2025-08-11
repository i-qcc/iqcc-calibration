# %%
from pathlib import Path
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from iqcc_calibration_tools.quam_config.lib.iqcc_cloud_data_storage_utils.upload_state_and_wiring import save_quam_state_to_cloud

# %%
library = QualibrationLibrary.get_active_library()

# %%
class Parameters(GraphParameters):
    qubits: List[str] = None

parameters = Parameters()

# Get the relevant QuAM components
if parameters.qubits is None:
    machine = Quam.load()
    parameters.qubits = [q.name for q in machine.active_qubits]

simulate = False
multiplexed = True
flux_point = "joint"
reset_type_thermal_or_active = "thermal"


g = QualibrationGraph(
    name="graph_retune_fine_qc_gilboa",
    parameters=parameters,
    nodes={
        "IQ_blobs": library.nodes["07b_IQ_Blobs"].copy(
            flux_point_joint_or_independent=flux_point,
            multiplexed=multiplexed,
            name="IQ_blobs",
            reset_type_thermal_or_active="thermal",
        ),
        "T2Ramsey": library.nodes["06_Ramsey"].copy(name="T2Ramsey",
        simulate=simulate),
        "power_rabi_x180": library.nodes["04_Power_Rabi"].copy(
            flux_point_joint_or_independent=flux_point,
            operation_x180_or_any_90="x180",
            name="power_rabi_x180",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            min_amp_factor=0.8,
            max_amp_factor=1.2,
            amp_factor_step=0.002,
            max_number_rabi_pulses_per_sweep=40,
            update_x90=True,
            state_discrimination=False,
            multiplexed=multiplexed,
        ),
        "single_qubit_randomized_benchmarking": library.nodes["10a_Single_Qubit_Randomized_Benchmarking"].copy(
            flux_point_joint_or_independent=flux_point, 
            multiplexed=True, 
            delta_clifford=100,
            num_random_sequences=1000,
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            name="single_qubit_randomized_benchmarking"
        ),
    },
    connectivity=[
        ("IQ_blobs", "T2Ramsey"),
        ("T2Ramsey", "power_rabi_x180"),
        ("power_rabi_x180", "single_qubit_randomized_benchmarking"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=True),
)
# %%

g.run()
# %%

save_quam_state_to_cloud(as_new_parent=True)
# %%
