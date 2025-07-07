import json
import numpy as np

# ANSI escape codes for text formatting
RED = '\033[91m'
RESET = '\033[0m'

def format_frequency(freq, threshold=400, width=12):
    """Format frequency with red if it's close to threshold in absolute value"""
    s_num = f"{freq:.3f}"
    if abs(freq) > threshold:
        return f"{RED}{s_num.rjust(width)}{RESET}"
    else:
        return s_num.rjust(width)

def extract_frequencies(state_file_path, wiring_file_path):
    # Load both files
    with open(state_file_path, 'r') as f:
        state = json.load(f)
    with open(wiring_file_path, 'r') as f:
        wiring = json.load(f)
    
    # Initialize lists to store frequencies
    qubit_names = []
    xy_if_freqs = []
    xy_lo_freqs = []
    xy_total_freqs = []
    rr_if_freqs = []
    rr_lo_freqs = []
    rr_total_freqs = []
    anharmonicities = []
    x180_drag_detunings = []
    x180_drag_amplitudes = []
    y90_drag_axis_angles = []
    
    # Create a mapping of port IDs to their frequencies
    port_freq_map = {}
    for controller_id, controller_data in state['ports']['mw_outputs'].items():
        for fem_id, fem_data in controller_data.items():
            for port_id, port_data in fem_data.items():
                # Only add ports that have upconverter_frequency
                if 'upconverter_frequency' in port_data:
                    port_freq_map[(controller_id, int(fem_id), int(port_id))] = port_data['upconverter_frequency']
    
    # Extract frequencies for each qubit
    for qubit_name, qubit_data in state['qubits'].items():
        # XY frequencies
        if 'xy' in qubit_data:
            # Get intermediate frequency
            xy_if_freq = qubit_data['xy'].get('intermediate_frequency', 0)
            
            # Get x180_DragCosine detuning
            x180_drag_detuning = 0
            x180_drag_amplitude = 0
            if 'operations' in qubit_data['xy'] and 'x180_DragCosine' in qubit_data['xy']['operations']:
                x180_drag_detuning = qubit_data['xy']['operations']['x180_DragCosine'].get('detuning', 0)
                x180_drag_amplitude = qubit_data['xy']['operations']['x180_DragCosine'].get('amplitude', 0)
            
            # Get y90_DragCosine axis_angle
            y90_drag_axis_angle = 0
            if 'operations' in qubit_data['xy'] and 'y90_DragCosine' in qubit_data['xy']['operations']:
                y90_drag_axis_angle = qubit_data['xy']['operations']['y90_DragCosine'].get('axis_angle', 0)
            
            # Get port information from wiring
            if qubit_name in wiring['wiring']['qubits']:
                # XY port reference
                xy_port_ref = wiring['wiring']['qubits'][qubit_name]['xy']['opx_output']
                if xy_port_ref.startswith('#/ports/mw_outputs/'):
                    # Extract port information from the reference
                    path_parts = xy_port_ref.split('/')
                    controller_id = path_parts[3]
                    fem_id = int(path_parts[4])
                    port_id = int(path_parts[5])
                    
                    # Get LO frequency from our mapping
                    xy_lo_freq = port_freq_map.get((controller_id, fem_id, port_id))
                    if xy_lo_freq is not None:
                        # Calculate total frequency
                        xy_total_freq = xy_lo_freq + xy_if_freq
                        
                        # RR frequencies
                        if 'resonator' in qubit_data:
                            rr_if_freq = qubit_data['resonator'].get('intermediate_frequency', 0)
                            
                            # Get RR port reference
                            rr_port_ref = wiring['wiring']['qubits'][qubit_name]['rr']['opx_output']
                            if rr_port_ref.startswith('#/ports/mw_outputs/'):
                                # Extract port information from the reference
                                path_parts = rr_port_ref.split('/')
                                controller_id = path_parts[3]
                                fem_id = int(path_parts[4])
                                port_id = int(path_parts[5])
                                
                                # Get LO frequency from our mapping
                                rr_lo_freq = port_freq_map.get((controller_id, fem_id, port_id))
                                if rr_lo_freq is not None:
                                    # Calculate total frequency
                                    rr_total_freq = rr_lo_freq + rr_if_freq
                                    
                                    # Get anharmonicity and convert to MHz
                                    anharmonicity_hz = qubit_data.get('anharmonicity', 0)
                                    
                                    # Store all values
                                    qubit_names.append(qubit_name)
                                    xy_if_freqs.append(xy_if_freq)
                                    xy_lo_freqs.append(xy_lo_freq)
                                    xy_total_freqs.append(xy_total_freq)
                                    rr_if_freqs.append(rr_if_freq)
                                    rr_lo_freqs.append(rr_lo_freq)
                                    rr_total_freqs.append(rr_total_freq)
                                    anharmonicities.append(anharmonicity_hz)
                                    x180_drag_detunings.append(x180_drag_detuning)
                                    x180_drag_amplitudes.append(x180_drag_amplitude)
                                    y90_drag_axis_angles.append(y90_drag_axis_angle)
    
    # Create a structured output
    output = {
        'qubit': qubit_names,
        'xy_intermediate_frequency': xy_if_freqs,
        'xy_lo_frequency': xy_lo_freqs,
        'xy_total_frequency': xy_total_freqs,
        'rr_intermediate_frequency': rr_if_freqs,
        'rr_lo_frequency': rr_lo_freqs,
        'rr_total_frequency': rr_total_freqs,
        'anharmonicity': anharmonicities,
        'x180_drag_detuning': x180_drag_detunings,
        'x180_drag_amplitude': x180_drag_amplitudes,
        'y90_drag_axis_angle': y90_drag_axis_angles
    }
    
    return output

if __name__ == '__main__':
    # Path to your state file
    state_file = 'quam_state/state.json'
    wiring_file = 'quam_state/wiring.json'
    
    # Extract frequencies
    frequencies = extract_frequencies(state_file, wiring_file)
    
    # Create a list of tuples for sorting
    qubit_data = list(zip(
        frequencies['qubit'],
        frequencies['xy_intermediate_frequency'],
        frequencies['xy_lo_frequency'],
        frequencies['xy_total_frequency'],
        frequencies['rr_intermediate_frequency'],
        frequencies['rr_lo_frequency'],
        frequencies['rr_total_frequency'],
        frequencies['anharmonicity'],
        frequencies['x180_drag_detuning'],
        frequencies['x180_drag_amplitude'],
        frequencies['y90_drag_axis_angle']
    ))
    
    # Sort by qubit name alphanumerically
    qubit_data.sort(key=lambda x: x[0])
    
    # Print results in a table format
    print("\nQubit Frequencies (sorted alphanumerically):")
    print("-" * 135)
    print(f"{'Qubit':<6} {'XY IF (MHz)':>13} {'XY LO (GHz)':>13} {'XY Total (GHz)':>15} {'RR IF (MHz)':>13} {'RR LO (GHz)':>13} {'RR Total (GHz)':>15} {'Anharm (MHz)':>14} {'X180 Det (MHz)':>15} {'X180 Amp (V)':>15} {'Y90 Axis Angle (deg)':>15}")
    print("-" * 135)
    
    for qubit, xy_if, xy_lo, xy_total, rr_if, rr_lo, rr_total, anharm, x180_det, x180_amp, y90_axis_angle in qubit_data:
        # Convert to appropriate units
        xy_if_freq = xy_if / 1e6
        xy_lo_freq = xy_lo / 1e9
        xy_total_freq = xy_total / 1e9
        rr_if_freq = rr_if / 1e6
        rr_lo_freq = rr_lo / 1e9
        rr_total_freq = rr_total / 1e9
        anharm_freq_mhz = anharm / 1e6 # Convert from Hz to MHz
        x180_det_mhz = x180_det / 1e6 # Convert from Hz to MHz
        
        # Format frequencies with red if they're close to 400 MHz
        xy_if_freq_str = format_frequency(xy_if_freq, width=13)
        rr_if_freq_str = format_frequency(rr_if_freq, width=13)
        
        # Print with fixed column widths
        print(f"{qubit:<6} {xy_if_freq_str} {xy_lo_freq:13.3f} {xy_total_freq:15.3f} {rr_if_freq_str} {rr_lo_freq:13.3f} {rr_total_freq:15.3f} {anharm_freq_mhz:13.3f} {x180_det_mhz:15.3f} {x180_amp:15.3f} {y90_axis_angle:15.3f}") 