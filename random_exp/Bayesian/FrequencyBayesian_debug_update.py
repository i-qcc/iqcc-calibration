# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from qualang_tools.units import unit
from qualang_tools.multi_user import qm_session
from qm.qua import *
from typing import Optional, List, Literal
import numpy as np
from qualang_tools.results import progress_counter, fetching_tool
u = unit(coerce_to_integer=True)

# %% {Node_parameters}
class Parameters(NodeParameters):
    # Define which qubits to measure
    qubits: Optional[List[str]] = ["Q5"]

    # Experiment parameters
    num_repetitions: int = 100
    detuning: int = 7 * u.MHz
    # min_wait_time_in_ns: int = 16
    min_wait_time_in_ns: int = 36
    max_wait_time_in_ns: int = 200
    wait_time_step_in_ns: int = 72
    
    physical_detuning: int = 2 * u.MHz

    # Bayesian parameters
    f_min: float = 6.9 #MHz
    f_max: float = 7.2 #MHz
    df: float = 0.1 #MHz

    # Control parameters
    reset_type: Literal["active", "thermal"] = "thermal"
    use_state_discrimination: bool = True

    # Execution parameters
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False

# Create experiment node
node = QualibrationNode(name="FrequencyBayes", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
# Initialize unit handling
u = unit(coerce_to_integer=True)

# Load QuAM configuration
machine = Quam.load()

# Generate hardware configurations
config = machine.generate_config()

# Connect to quantum control hardware
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Get qubit objects
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]

num_qubits = len(qubits)
    


# %% {QUA_program}
# Set up experiment parameters
v_f = np.arange(node.parameters.f_min, node.parameters.f_max + 0.5 * node.parameters.df, node.parameters.df)

qubit = qubits[0]
# Define QUA program
with program() as BayesFreq:
    
    n = declare(int, value = 1)    
    n_st = declare_stream()
    t = declare(int)
    frequencies = declare(fixed, value=v_f.tolist())
    t_sample = declare(fixed) #normalization for time in us
    C = declare(fixed)
    f_idx = declare(int)
    new_freqeuncy = declare(int)
    new_freqeuncy_st = declare_stream()
    
    # Main experiment loop

    machine.initialize_qpu(flux_point="joint", target=qubit)
    align()
    save(n, n_st)
    with for_(t, 13, t < 50, t + 26):
        qubit.xy.wait(t)
        align()
        assign(t_sample, Cast.mul_fixed_by_int(1e-3, t * 4))
        with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
            assign(C, Math.cos2pi(frequencies[f_idx] * t_sample))
    
    assign(new_freqeuncy, Cast.mul_int_by_fixed(1_000_000, C))
    qubit.xy.update_frequency(new_freqeuncy+qubit.xy.intermediate_frequency)                   

    with stream_processing():
        n_st.save("n")
# %% {Simulate_or_execute}

with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    job = qm.execute(BayesFreq)
    results = fetching_tool(job, ["n"], mode="live")
    while results.is_processing():
        # Fetch results
        n = results.fetch_all()[0]
        # Progress bar
        progress_counter(n, node.parameters.num_repetitions, start_time=results.start_time)
# %%