from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters
from typing import List
from dataclasses import field

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    amplitude_scales: List[float] = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
