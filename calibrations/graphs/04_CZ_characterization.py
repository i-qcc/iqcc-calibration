# %%
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary
from iqcc_calibration_tools.quam_config.components.quam_root import Quam

machine = Quam.load()
qubit_pairs = machine.active_qubit_pair_names
multiplexed = True

node_params = {
    "ramsey_flux_calibration" : {"multiplexed": multiplexed,
            "num_shots": 200,
            "flux_span": 0.02,
            "max_wait_time_in_ns": 500,
            "wait_time_step_in_ns": 5,
            "flux_num": 11,
            "frequency_detuning_in_mhz": 4},
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
        "bell_state_tomography": library.nodes["40b_Bell_state_tomography"].copy(name="bell_state_tomography"),
        "ramsey_flux_calibration": library.nodes["09_ramsey_vs_flux_calibration"].copy(name="ramsey_flux_calibration"),
    },
    connectivity=[
        ("bell_state_tomography", "ramsey_flux_calibration"),
        ("ramsey_flux_calibration", "bell_state_tomography"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)



# %%

g.run(qubit_pairs=qubit_pairs, nodes=node_params)
# %%
