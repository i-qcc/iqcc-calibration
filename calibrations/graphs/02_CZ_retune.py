# %%
from typing import List

from calibration_utils.T1 import parameters
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary
from iqcc_calibration_tools.quam_config.components.quam_root import Quam

machine = Quam.load()
qubit_pairs = machine.active_qubit_pair_names
multiplexed = True

node_params = {
    "chevron": {"max_time_in_ns": 128, "reset_type": "active"},
    "conditional_phase": {},
    "phase_compensation": {}
    }

# %% 
library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    targets_name = "qubit_pairs"
    qubit_pairs: List[str] = None

g = QualibrationGraph(
    name="CZ_retune",
    parameters=Parameters(),
    nodes={
        # "chevron": library.nodes["30c_chevron_11_02"].copy(name="chevron"),
        "chevron": library.nodes["30b_02_11_oscillations_1nS"].copy(name="chevron"),
        "conditional_phase": library.nodes["32c_cz_conditional_phase"].copy(name="conditional_phase"),
        "phase_compensation": library.nodes["33d_cz_phase_compensation"].copy(name="phase_compensation"),
    },
    connectivity=[
        ("chevron", "conditional_phase"),
        ("conditional_phase", "phase_compensation"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)



# %%

g.run(qubit_pairs=qubit_pairs, nodes=node_params)
# %%
