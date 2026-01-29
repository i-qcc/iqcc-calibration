# %%
from pathlib import Path
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = None

parameters = Parameters()

# Get the relevant QuAM components
if parameters.qubits is None:
    machine = Quam.load()
    parameters.qubits = [q.name for q in machine.active_qubits]


multiplexed = True
flux_point = "joint"
reset_type_thermal_or_active = "thermal"


g = QualibrationGraph(
    name="retune_reduced_qc_qwfix",
    parameters=parameters,
    nodes={
        # "qubit_spectroscopy": library.nodes["03a_Qubit_Spectroscopy"].copy(name="qubit_spectroscopy"),
        "power_rabi_x180": library.nodes["04_Power_Rabi"].copy(    
            name="power_rabi_x180"
        ),
        "T1": library.nodes["05_T1"].copy(name="T1", 
        multiplexed=True),
        "T2Ramsey": library.nodes["06_Ramsey"].copy(name="T2Ramsey"),
        "readout_frequency_optimization": library.nodes["07a_Readout_Frequency_Optimization"].copy(name="readout_frequency_optimization"),
        "readout_power_optimization": library.nodes["07c_Readout_Power_Optimization"].copy(name="readout_power_optimization"),
        # "IQ_blobs": library.nodes["07b_IQ_Blobs"].copy(
        #     flux_point_joint_or_independent=flux_point,
        #     multiplexed=multiplexed,
        #     name="IQ_blobs",
        #     reset_type_thermal_or_active="thermal",
        # ),
        "drag_calibration": library.nodes["09b_DRAG_Calibration_180_minus_180"].copy(name="drag_calibration", 
        min_amp_factor=-1.99,
        max_amp_factor=1.99,
        reset_type_thermal_or_active=reset_type_thermal_or_active),
        
        "single_qubit_randomized_benchmarking": library.nodes["10a_Single_Qubit_Randomized_Benchmarking"].copy(
            
            multiplexed=False, 
            name="single_qubit_randomized_benchmarking",
            reset_type_thermal_or_active=reset_type_thermal_or_active
        ),
    },
    connectivity=[
        # ("qubit_spectroscopy", "power_rabi_x180"),
        ("power_rabi_x180", "T1"),
        ("T1", "T2Ramsey"),
        ("T2Ramsey", "readout_frequency_optimization"),
        ("readout_frequency_optimization", "readout_power_optimization"),
        ("readout_power_optimization", "drag_calibration"),
        ("drag_calibration", "single_qubit_randomized_benchmarking"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=True),
)


# %%

g.run()
# %%
