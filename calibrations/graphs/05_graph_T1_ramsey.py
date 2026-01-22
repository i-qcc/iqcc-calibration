# %%
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary
from iqcc_calibration_tools.quam_config.components.quam_root import Quam

# %%
library = QualibrationLibrary.get_active_library()

# %%

class Parameters(GraphParameters):
    qubits: List[str] = None
  
multiplexed = True
reset_type = "thermal"

nodes = {
    "T1": library.nodes["05_T1"],
    "ramsey": library.nodes["06a_ramsey"],
}

node_params = {
    "T1": {
        "multiplexed": multiplexed,
        "reset_type": reset_type,
        "num_shots": 1000,
        "wait_time_num_points": 50,
        "max_wait_time_in_ns": 20000,
        "log_or_linear_sweep": "log",
        "use_state_discrimination": True
    },
    "ramsey": {
        "multiplexed": multiplexed,
        "reset_type": reset_type,
        "num_shots": 100,
        "frequency_detuning_in_mhz": 1.0,
        "max_wait_time_in_ns": 7000,
    },
}


g = QualibrationGraph(
    name="graph_T1_ramsey",
    parameters=Parameters(),
    nodes=nodes,
    connectivity=[
        ("T1", "ramsey"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=True),
)
# %%

machine = Quam.load()
active_qubits = [q.name for q in machine.active_qubits]



g.run(qubits=active_qubits, nodes=node_params)
# %%
