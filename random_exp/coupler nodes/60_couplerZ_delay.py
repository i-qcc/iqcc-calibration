# %%
"""
    XY-Z delay as describe in page 108 at https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf
"""
import warnings

from datetime import datetime, timezone, timedelta
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from typing import Optional, Literal

class Parameters(NodeParameters):
    qubit_pairs: Optional[str] = None
    num_averages: int = 100
    delay_span: int = 70 # in clock cycles
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    reset_type_thermal_or_active: Literal['thermal', 'active'] = "thermal"
    which_qubit : Literal['control', 'target'] = "target"
    flux_amp: float = 0.03
    simulate: bool = False
    timeout: int = 100
    


node = QualibrationNode(
    name="60_couplerZ_delay",
    parameters=Parameters()
)

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.units import unit
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration, active_reset, readout_state
import matplotlib.pyplot as plt
from qualang_tools.bakery import baking
import numpy as np

from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray
from iqcc_calibration_tools.analysis.plot_utils import QubitPairGrid, grid_iter, grid_pair_names

# matplotlib.use("TKAgg")


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()
# Generate the OPX and Octave configurations
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == '':
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[q] for q in node.parameters.qubit_pairs]


config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

num_qubit_pairs = len(qubit_pairs)

# %%
###################
# The QUA program #
###################

n_avg = node.parameters.num_averages  # Number of averages

relative_time = np.arange(-node.parameters.delay_span + 4, node.parameters.delay_span,
                          1)  # x-axis for plotting - Must be in clock cycles.

n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

# %%

with program() as xy_z_delay_calibration:
    I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubit_pairs)
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_stream_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_stream_target = [declare_stream() for _ in range(num_qubit_pairs)]
    t = declare(int)  # QUA variable for the flux pulse segment index

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit_pairs[0].qubit_control)
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        if flux_point != "joint":
            machine.set_all_fluxes(flux_point=flux_point, target=qp.qubit_control)

        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_each_(t, relative_time):
                qp.align()
                if node.parameters.which_qubit == "control":
                    qubit = qp.qubit_control
                else:
                    qubit = qp.qubit_target

                # qubit reset
                if reset_type == "active":
                    active_reset(qubit)
                else:
                    qubit.resonator.wait(machine.thermalization_time * u.ns)
                    qp.align()

                qp.align()
                with strict_timing_():
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_control.z.wait(qubit.xy.operations['x180'].length // 4)
                    qp.qubit_target.z.wait(qp.qubit_target.xy.operations['x180'].length // 4)
                    qp.coupler.wait(qubit.xy.operations['x180'].length // 4)
                    
                    qp.qubit_control.z.wait(node.parameters.delay_span)
                    if node.parameters.which_qubit == "control":
                        qp.coupler.wait(node.parameters.delay_span + t)
                    else:
                        qp.coupler.wait(node.parameters.delay_span)
                        qp.qubit_target.z.wait(node.parameters.delay_span + t)
                        qp.qubit_target.z.play("const", amplitude_scale=node.parameters.flux_amp/qp.coupler.operations['const'].amplitude,duration = 4)
                    qp.qubit_control.z.play("const", amplitude_scale=qp.detuning/qp.qubit_control.z.operations['const'].amplitude,duration = 4)
                    qp.coupler.play("const", amplitude_scale=node.parameters.flux_amp/qp.coupler.operations['const'].amplitude,duration = 4)

                qp.align()
                readout_state(qp.qubit_control, state_control[i])
                readout_state(qp.qubit_target, state_target[i])
                save(state_control[i], state_stream_control[i])
                save(state_target[i], state_stream_target[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i, qp in enumerate(qubit_pairs):
            state_stream_control[i].buffer(len(relative_time)).average().save(f"state_control{i + 1}")
            state_stream_target[i].buffer(len(relative_time)).average().save(f"state_target{i + 1}")

# %%

###########################
# Run or Simulate Program #
###########################
simulate = node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, xy_z_delay_calibration, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    quit()
else:
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(xy_z_delay_calibration)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # fetch results
            n = results.fetch_all()[0]
            # progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %%
handles = job.result_handles
ds = fetch_results_as_xarray(handles, qubit_pairs, {"relative_time": relative_time*4})
ds.relative_time.attrs['long_name'] = 'timing_delay'
ds.relative_time.attrs['units'] = 'nS'
node.results = {}
node.results['ds'] = ds

# %%
fig, ax = plt.subplots(1)
ds.state_control.plot(ax=ax)
ds.state_target.plot(ax=ax)
ax.set_xlabel("Relative Time")
ax.set_ylabel("State")
ax.set_title("Coupler Z Delay")
ax.legend()
plt.tight_layout()
plt.show()
# %%

# %%

# find where the valus of ds.state.sel(sequence=0) is above 0.5 and return the mean of the relative_time
delays_control = (ds.state_control.where(ds.state_control < 0.5) / ds.state_control.where(ds.state_control < 0.5) * ds.relative_time).mean(dim="relative_time")
delays_target = (ds.state_target.where(ds.state_target > 0.5) / ds.state_target.where(ds.state_target > 0.5) * ds.relative_time).mean(dim="relative_time")
delays = (delays_control + delays_target) / 2
# %%

# %%
grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
grid = QubitPairGrid(grid_names, qubit_pair_names)    

flux_delays = []
for ax, qp in grid_iter(grid):
    # x = ds.relative_time
    qp = qp['qubit']
    ds.state_control.sel(qubit=qp).plot( ax = ax)
    ds.state_target.sel(qubit=qp).plot( ax = ax)
    ax.axvline(delays_control.sel(qubit=qp), color="C0", linestyle="--", label="control")
    ax.axvline(delays_target.sel(qubit=qp), color="C1", linestyle="--", label="target")
    
    ax.set_xlabel("Relative Time")
    ax.set_ylabel("State")
    ax.set_title(f"{qp}")

    ax.legend()

grid.fig.suptitle(f'Coupler Z Delay Fitting \n {date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type_thermal_or_active}')
plt.tight_layout()
plt.show()

node.results['figure'] = grid.fig

# %% {Update_state}
with node.record_state_updates():
    for i, qp in enumerate(qubit_pairs):
        if delays.sel(qubit=qp.name) is not None:
            qp.coupler.opx_output.delay += int(np.round(delays.sel(qubit=qp.name).values))

# %% {Save_results}
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()

# %%