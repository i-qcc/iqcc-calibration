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
simulate = False


g = QualibrationGraph(
    name="graph_2b_3d_04_05",
    parameters=parameters,
    nodes={
        "resonator_spectroscopy_vs_flux": library.nodes["02b_Resonator_Spectroscopy_vs_Flux"].copy(
            name="resonator_spectroscopy_vs_flux",
            num_averages=20,
            min_flux_offset_in_v=-0.3,
            max_flux_offset_in_v=0.3,
            num_flux_points=101,
            simulate=simulate
        ),
        "qubit_spectroscopy_vs_readout_amp": library.nodes["03d_Qubit_Spectroscopy_vs_readout_amp"].copy(
            name="qubit_spectroscopy_vs_readout_amp",
            num_averages=500,
            frequency_span_in_mhz=50,
            frequency_step_in_mhz=0.25,
            readout_amp_start=0.75,
            readout_amp_stop=1.5,
            readout_amp_steps=10,
            flux_point_joint_or_independent=flux_point,
            simulate=simulate
        ),
        "power_rabi": library.nodes["04_Power_Rabi"].copy(    
            name="power_rabi",
            num_averages=50,
            operation_x180_or_any_90="x180",
            min_amp_factor=0.0,
            max_amp_factor=1.5,
            amp_factor_step=0.05,
            max_number_rabi_pulses_per_sweep=1,
            flux_point_joint_or_independent=flux_point,
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            multiplexed=multiplexed,
            simulate=simulate
        ),
        "T1": library.nodes["05_T1"].copy(
            name="T1", 
            num_averages=100,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=100000,
            wait_time_step_in_ns=600,
            flux_point_joint_or_independent_or_arbitrary=flux_point,
            reset_type=reset_type_thermal_or_active,
            multiplexed=False,
            simulate=simulate
        ),
    },
    connectivity=[
        ("resonator_spectroscopy_vs_flux", "qubit_spectroscopy_vs_readout_amp"),
        ("qubit_spectroscopy_vs_readout_amp", "power_rabi"),
        ("power_rabi", "T1"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=True),
)


# %%

g.run()
# %% 