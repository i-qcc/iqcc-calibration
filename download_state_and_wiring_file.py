import os
import json
from iqcc_cloud_client import IQCC_Cloud

# Set your quantum computer backend
quantum_computer_backend = "gilboa" # "your_quantum_computer_backend" for example arbel
qc = IQCC_Cloud(quantum_computer_backend=quantum_computer_backend)

# Get the latest state and wiring files
latest_wiring = qc.state.get_latest("wiring")
latest_state = qc.state.get_latest("state")

# # Get the state folder path from environment variable
# quam_state_folder_path = os.environ["QUAM_STATE_PATH"]

# # Save the files
# with open(os.path.join(quam_state_folder_path, "wiring.json"), "w") as f:
#     json.dump(latest_wiring.data, f, indent=4)

# with open(os.path.join(quam_state_folder_path, "state.json"), "w") as f:
#     json.dump(latest_state.data, f, indent=4)

# Get the state folder path from environment variable
state_folder_path = os.path.join(os.getcwd(), "quam_state")
print(state_folder_path)

# Save the state and wiring files
with open(os.path.join(state_folder_path, "state.json"), "w") as f:
    json.dump(latest_state.data, f, indent=4)

with open(os.path.join(state_folder_path, "wiring.json"), "w") as f:
    json.dump(latest_wiring.data, f, indent=4)