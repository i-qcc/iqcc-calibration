# %%
from typing import List
from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate import GraphParameters
from qualibrate import QualibrationGraph
from qualibrate import QualibrationLibrary
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam

machine = Quam.load()
qubit_pairs = machine.active_qubit_pair_names
multiplexed = True
reset_type = "thermal" # I find that active reset gives reduced performance

node_params = {
    # "chevron": {"max_time_in_ns": 96, "reset_type": "active", "num_averages": 20},
    "confusion_matrix": {"reset_type": reset_type, "multiplexed": multiplexed},
    "conditional_phase": {"operation": "cz", "amp_range": 0.03, "amp_step": 0.003, "reset_type": reset_type, "multiplexed": multiplexed},
    "phase_compensation": {"operation": "cz", "reset_type": reset_type, "multiplexed": multiplexed},
    "bell_state_tomography": {"reset_type": "thermal", "multiplexed": multiplexed} # active reset hangs
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
        # "chevron": library.nodes["30b_02_11_oscillations_1nS"].copy(name="chevron"),
        "conditional_phase": library.nodes["32c_cz_conditional_phase"].copy(name="conditional_phase"),
        "phase_compensation": library.nodes["33d_cz_phase_compensation"].copy(name="phase_compensation"),
        "confusion_matrix": library.nodes["34_2Q_confusion_matrix"].copy(name="confusion_matrix"),
        "bell_state_tomography": library.nodes["40b_Bell_state_tomography"].copy(name="bell_state_tomography")
    },
    connectivity=[
        # ("chevron", "conditional_phase"),
        ("conditional_phase", "phase_compensation"),
        ("phase_compensation", "confusion_matrix"),
        ("confusion_matrix", "bell_state_tomography")
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)



# %%

g.run(qubit_pairs=qubit_pairs, nodes=node_params)
# %%
