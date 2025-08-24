# Make iqcc_calibration_tools a Python package

from iqcc_cloud_client.computers import *  # noqa: F403, F401
from .patches import apply_iqcc_patches


# Apply patches to external libraries
apply_iqcc_patches()
