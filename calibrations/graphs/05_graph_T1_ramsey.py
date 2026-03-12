# %%
from typing import List
from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate import GraphParameters
from qualibrate import QualibrationGraph
from qualibrate import QualibrationLibrary
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam

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
    "T2echo": library.nodes["06b_echo"],
}

node_params = {
    "T1": {
        "multiplexed": False,
        "reset_type": reset_type,
        "num_shots": 100,
        "wait_time_num_points": 25,
        "max_wait_time_in_ns": 100000,
        "log_or_linear_sweep": "log",
        "use_state_discrimination": True
    },
    "ramsey": {
        "multiplexed": False,
        "reset_type": reset_type,
        "num_shots": 25,
        "frequency_detuning_in_mhz": 1.0,
        "wait_time_num_points": 50,
        "max_wait_time_in_ns": 20000,
    },
    "T2echo": {
        "multiplexed": False,
        "reset_type": reset_type,
        "num_shots": 100,
        "wait_time_num_points": 50,
        "max_wait_time_in_ns": 40000,
        "log_or_linear_sweep": "log",
        "use_state_discrimination": True,
    },
}


g = QualibrationGraph(
    name="graph_T1_ramsey",
    parameters=Parameters(),
    nodes=nodes,
    connectivity=[
        ("T1", "ramsey"),
        ("ramsey", "T2echo"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=True),
)
# %%

machine = Quam.load()
active_qubits = [q.name for q in machine.active_qubits]



g.run(qubits=active_qubits, nodes=node_params)
# %%
