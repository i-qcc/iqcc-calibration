import json

# Load the state.json file
with open('quam_state/state.json', 'r') as f:
    state = json.load(f)

# Create dictionary to store grid locations
grid_locations = {}

# Collect grid locations for all qubits
for qubit_name, qubit_data in state['qubits'].items():
    if 'grid_location' in qubit_data:
        grid_locations[qubit_name] = qubit_data['grid_location']

# Print the dictionary in a format that can be copied to Python code
print("grid_locations = {")
for qubit, location in sorted(grid_locations.items()):
    print(f'    "{qubit}": "{location}",')
print("}") 