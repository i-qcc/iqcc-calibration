from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters
from typing import List
from dataclasses import field

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 500
    min_wait_time_in_ns: int = 48
    max_wait_time_in_ns: int = 500
    num_time_steps: int = 500
    drive_amp_scale: float = 0.1 # 1.0
    target_freq_in_Mhz: float = 60 

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
