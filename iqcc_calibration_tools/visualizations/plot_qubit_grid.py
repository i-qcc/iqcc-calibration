#!/usr/bin/env python3
"""
Script to plot all qubits in a grid according to their grid locations,
marking active qubits and active qubit pairs.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from math import floor, log10
from typing import Dict, Tuple, List, Set, Optional
from pathlib import Path

from iqcc_calibration_tools.quam_config.components.quam_root import Quam


def parse_grid_location(location_str: str) -> Tuple[int, int]:
    """Parse grid location string like '2,4' into (x, y) tuple."""
    x, y = map(int, location_str.split(','))
    return (x, y)


def get_qubit_color(rb_fidelity_pct: float) -> str:
    """
    Get color for qubit based on 1Q RB fidelity.
    
    Args:
        rb_fidelity_pct: RB fidelity as percentage (e.g., 98.5 for 98.5%)
    
    Returns:
        Color string (hex format)
    """
    # Dark Red: ≤ 98.5%, Dark Green: ≥ 99.9%, gradient in between
    min_fidelity = 98.5
    max_fidelity = 99.9
    
    # Clamp value to range
    normalized = np.clip((rb_fidelity_pct - min_fidelity) / (max_fidelity - min_fidelity), 0, 1)
    
    # Interpolate between dark red (#B30000), orange (#FFA500), and dark green (#008000)
    # Dark red RGB: (0.7, 0, 0), Dark green RGB: (0, 0.5, 0)
    if normalized < 0.5:
        # Dark red to orange
        t = normalized * 2
        r = 0.9 + t * (1.0 - 0.9)   # From dark red (0.9) to orange (1.0)
        g = t * 0.647               # 165/255 for orange
        b = 0.0
    else:
        # Orange to dark green
        t = (normalized - 0.5) * 2
        r = 1.0 - t * 1.0           # From orange (1.0) to dark green (0.0)
        g = 0.647 + t * (0.5 - 0.647)  # From orange (0.647) to dark green (0.5)
        b = 0.0
    
    return mcolors.rgb2hex([r, g, b])


def get_pair_color(bell_fidelity_pct: float) -> str:
    """
    Get color for qubit pair based on Bell state fidelity.
    
    Args:
        bell_fidelity_pct: Bell state fidelity as percentage (e.g., 95.0 for 95.0%)
    
    Returns:
        Color string (hex format)
    """
    # Dark Red: ≤ 95%, Dark Green: ≥ 99%, gradient in between
    min_fidelity = 95.0
    max_fidelity = 99.0
    
    # Clamp value to range
    normalized = np.clip((bell_fidelity_pct - min_fidelity) / (max_fidelity - min_fidelity), 0, 1)
    
    # Interpolate between dark red (#B30000), orange (#FFA500), and dark green (#008000)
    # Dark red RGB: (0.7, 0, 0), Dark green RGB: (0, 0.5, 0)
    if normalized < 0.5:
        # Dark red to orange
        t = normalized * 2
        r = 0.7 + t * (1.0 - 0.7)   # From dark red (0.7) to orange (1.0)
        g = t * 0.647               # 165/255 for orange
        b = 0.0
    else:
        # Orange to dark green
        t = (normalized - 0.5) * 2
        r = 1.0 - t * 1.0           # From orange (1.0) to dark green (0.0)
        g = 0.647 + t * (0.5 - 0.647)  # From orange (0.647) to dark green (0.5)
        b = 0.0
    
    return mcolors.rgb2hex([r, g, b])


def extract_qubit_grid_locations(machine: Quam) -> Dict[str, Tuple[int, int]]:
    """Extract grid locations for all qubits."""
    qubit_grids = {}
    
    for qubit_name, qubit in machine.qubits.items():
        if hasattr(qubit, 'grid_location') and qubit.grid_location:
            qubit_grids[qubit_name] = parse_grid_location(qubit.grid_location)
    
    return qubit_grids


def get_active_qubits(machine: Quam) -> Set[str]:
    """Get set of active qubit names."""
    return set(machine.active_qubit_names)


def get_active_qubit_pairs(machine: Quam) -> List[Tuple[str, str]]:
    """Get list of active qubit pairs as tuples of (q1, q2)."""
    active_pair_names = machine.active_qubit_pair_names
    pairs = []
    
    for pair_name in active_pair_names:
        # Pair names are like "qB1-qB2"
        parts = pair_name.split('-')
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    
    return pairs


def extract_bell_state_fidelities(machine: Quam) -> Dict[Tuple[str, str], float]:
    """
    Extract Bell state fidelities for qubit pairs.
    Returns a dictionary mapping (q1, q2) tuples to fidelity values.
    """
    fidelities = {}
    
    for pair_name, qubit_pair in machine.qubit_pairs.items():
        # Parse pair name like "qB1-qB2"
        parts = pair_name.split('-')
        if len(parts) != 2:
            continue
        
        q1, q2 = parts[0], parts[1]
        pair_tuple = (q1, q2)
        
        # Search through macros to find fidelity
        if hasattr(qubit_pair, 'macros') and qubit_pair.macros:
            for macro_name, macro in qubit_pair.macros.items():
                # Skip if macro is a string reference (like "#./cz_unipolar")
                if isinstance(macro, str):
                    continue
                
                # Check if this macro has fidelity data
                # Structure: macro.fidelity["Bell_State"]["Fidelity"]
                # Example from JSON: qubit_pair.macros["cz_unipolar"].fidelity["Bell_State"]["Fidelity"]
                try:
                    # Check if macro has fidelity attribute
                    if not hasattr(macro, 'fidelity'):
                        continue
                    
                    fidelity_attr = macro.fidelity
                    if fidelity_attr is None:
                        continue
                    
                    # Try to access Bell_State fidelity
                    bell_state = None
                    
                    # Try direct dict access first
                    try:
                        bell_state = fidelity_attr['Bell_State']
                    except (KeyError, TypeError):
                        # Fallback to .get() if available
                        if hasattr(fidelity_attr, 'get'):
                            bell_state = fidelity_attr.get('Bell_State')
                    
                    if bell_state is not None:
                        # Extract Fidelity value from Bell_State
                        try:
                            if isinstance(bell_state, dict):
                                fidelity = bell_state.get('Fidelity')
                            elif hasattr(bell_state, 'get'):
                                fidelity = bell_state.get('Fidelity')
                            elif hasattr(bell_state, '__getitem__'):
                                fidelity = bell_state['Fidelity']
                            else:
                                fidelity = getattr(bell_state, 'Fidelity', None)
                            
                            if fidelity is not None:
                                fidelities[pair_tuple] = fidelity
                                break  # Use the first fidelity found
                        except (KeyError, TypeError, AttributeError):
                            pass
                            
                except (KeyError, TypeError, AttributeError):
                    # Skip if access fails
                    continue
    
    return fidelities


def extract_single_qubit_rb(machine: Quam, qubit_names: Set[str]) -> Dict[str, float]:
    """
    Extract single qubit randomized benchmarking (1Q RB) values.
    Extracts for all provided qubits, using the 'averaged' value from gate_fidelity.
    Returns a dictionary mapping qubit names to 1Q RB fidelity values.
    """
    rb_values = {}
    
    for qubit_name in qubit_names:
        if qubit_name not in machine.qubits:
            continue
        
        qubit = machine.qubits[qubit_name]
        try:
            # Access gate_fidelity directly like in calibration files (e.g., q.gate_fidelity["averaged"])
            # Check if gate_fidelity exists and has 'averaged' key (like: "averaged" not in q.gate_fidelity)
            if hasattr(qubit, 'gate_fidelity') and qubit.gate_fidelity is not None:
                gate_fidelity = qubit.gate_fidelity
                # Try direct access first (most common pattern in calibration files)
                try:
                    if 'averaged' in gate_fidelity:
                        rb_value = gate_fidelity['averaged']
                        if rb_value is not None:
                            rb_values[qubit_name] = rb_value
                except (KeyError, TypeError):
                    # If direct access fails, try .get() method
                    try:
                        rb_value = gate_fidelity.get('averaged') if hasattr(gate_fidelity, 'get') else None
                        if rb_value is not None:
                            rb_values[qubit_name] = rb_value
                    except (AttributeError, TypeError):
                        pass
        except (AttributeError, KeyError, TypeError):
            # Skip if gate_fidelity doesn't exist or doesn't have 'averaged'
            pass
    
    return rb_values


def plot_qubit_grid(
    qubit_grids: Dict[str, Tuple[int, int]],
    active_qubits: Set[str],
    active_pairs: List[Tuple[str, str]],
    fidelities: Optional[Dict[Tuple[str, str], float]] = None,
    rb_values: Optional[Dict[str, float]] = None,
    output_file: Optional[str] = None
):
    """
    Plot all qubits on a grid, highlighting active qubits and active pairs.
    
    Args:
        qubit_grids: Dictionary mapping qubit names to (x, y) grid coordinates
        active_qubits: Set of active qubit names
        active_pairs: List of tuples (q1, q2) representing active qubit pairs
        fidelities: Dictionary mapping (q1, q2) tuples to Bell state fidelity values
        rb_values: Dictionary mapping qubit names to 1Q RB fidelity values
        output_file: Optional path to save the plot
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract coordinates
    all_x = [x for x, y in qubit_grids.values()]
    all_y = [y for x, y in qubit_grids.values()]
    
    # Set up the plot limits with some padding
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    
    # Draw grid lines
    for x in range(x_min, x_max + 1):
        ax.axvline(x + 0.5, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)
    for y in range(y_min, y_max + 1):
        ax.axhline(y + 0.5, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Draw active qubit pairs (connections) first, so they appear behind qubits
    for q1, q2 in active_pairs:
        if q1 in qubit_grids and q2 in qubit_grids:
            x1, y1 = qubit_grids[q1]
            x2, y2 = qubit_grids[q2]
            
            # Determine line color based on Bell state fidelity
            pair_tuple = (q1, q2)
            fidelity = fidelities.get(pair_tuple) if fidelities else None
            if fidelity is None:
                fidelity = fidelities.get((q2, q1)) if fidelities else None
            
            if fidelity is not None:
                fidelity_pct = fidelity * 100
                line_color = get_pair_color(fidelity_pct)
            else:
                # Default blue if no fidelity data
                line_color = 'blue'
            
            ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=6, alpha=0.6, zorder=1)
            
            # Add fidelity label if available
            if fidelity is not None:
                # Position label at midpoint of the line
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # Offset perpendicular to the line to avoid overlap
                dx = x2 - x1
                dy = y2 - y1
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    # Perpendicular offset (rotate 90 degrees) - reduced from 0.3 to 0.15 to bring labels closer
                    offset_x = -dy / length * 0.15
                    offset_y = dx / length * 0.15
                else:
                    offset_x, offset_y = 0.15, 0.15
                
                label_x = mid_x + offset_x
                label_y = mid_y + offset_y
                
                # Format fidelity as percentage with 2 decimal places
                ax.text(label_x, label_y, f'{fidelity_pct:.2f}%', 
                       fontsize=7, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               edgecolor=line_color, alpha=0.8, linewidth=1),
                       zorder=6, weight='bold')
    
    # Plot all qubits
    for qubit_name, (x, y) in qubit_grids.items():
        is_active = qubit_name in active_qubits
        
        # Determine qubit color based on RB fidelity
        if is_active and rb_values and qubit_name in rb_values:
            rb_value = rb_values[qubit_name]
            rb_pct = rb_value * 100
            qubit_color = get_qubit_color(rb_pct)
            border_color = 'black'  # Black border for active qubits
        elif is_active:
            # Active qubit but no RB data - use default green
            qubit_color = 'green'
            border_color = 'black'  # Black border for active qubits
        else:
            # Inactive qubits remain gray
            qubit_color = 'lightgray'
            border_color = 'gray'
        
        if is_active:
            # Active qubits: larger, colored circle
            circle = plt.Circle((x, y), 0.25, color=qubit_color, zorder=3)
            ax.add_patch(circle)
            # Add border for better visibility
            border = plt.Circle((x, y), 0.25, fill=False, edgecolor=border_color, 
                              linewidth=2, zorder=4)
            ax.add_patch(border)
        else:
            # Inactive qubits: smaller, gray circle
            circle = plt.Circle((x, y), 0.15, color=qubit_color, zorder=2)
            ax.add_patch(circle)
        
        # Build qubit label text - include RB fidelity if available
        label_text = qubit_name
        if rb_values and qubit_name in rb_values:
            rb_value = rb_values[qubit_name]
            # Format as percentage with 4 significant digits
            rb_pct = rb_value * 100
            if rb_pct > 0:
                digits = 4 - int(floor(log10(abs(rb_pct))))
                rb_str = f'{rb_pct:.{max(0, digits-1)}f}%'
            else:
                rb_str = f'{rb_pct:.3f}%'
            label_text = f'{qubit_name}\n{rb_str}'
        
        # Add qubit label inside the circle (with RB fidelity if available)
        # Determine text color for good contrast
        # For active qubits with RB data, use white text (works well on red/orange/green)
        # For inactive qubits, use black text on light gray
        if is_active:
            text_color = 'white'  # White text on colored backgrounds
        else:
            text_color = 'black'  # Black text on light gray
        
        ax.text(x, y, label_text, fontsize=7, ha='center', va='center',
               weight='bold' if is_active else 'normal',
               color=text_color,
               zorder=6)
    
    # Set labels and title
    ax.set_xlabel('Grid X Coordinate', fontsize=12)
    ax.set_ylabel('Grid Y Coordinate', fontsize=12)
    ax.set_title('Qubit Grid Layout\n(Active qubits in green, Active pairs connected)', 
                fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Clear any existing legends to avoid duplicates
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    
    # Add legend in upper left
    active_patch = mpatches.Patch(color='green', label='Active Qubit')
    inactive_patch = mpatches.Patch(color='lightgray', label='Inactive Qubit')
    pair_line = plt.Line2D([0], [0], color='blue', linewidth=2.5, label='Active Pair')
    fidelity_label = plt.Line2D([0], [0], color='blue', linewidth=0, marker='s', 
                                markersize=8, markerfacecolor='white', 
                                markeredgecolor='blue', markeredgewidth=1,
                                label='Bell State Fidelity (%)')
    rb_label = plt.Line2D([0], [0], color='green', linewidth=0, marker='o', 
                          markersize=10, markerfacecolor='green', 
                          markeredgecolor='darkgreen', markeredgewidth=1,
                          label='1Q RB Fidelity (%)')
    ax.legend(handles=[active_patch, inactive_patch, pair_line, fidelity_label, rb_label], 
             loc='upper left', fontsize=9, framealpha=0.9)
    
    # Add statistics text box in lower left to avoid overlap with legend
    stats_text = f'Total Qubits: {len(qubit_grids)}\n'
    stats_text += f'Active Qubits: {len(active_qubits)}\n'
    stats_text += f'Active Pairs: {len(active_pairs)}'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    """Main function to load data and create the plot."""
    # Load Quam object
    print("Loading Quam state...")
    try:
        machine = Quam.load()
    except Exception as e:
        print(f"Error loading Quam state: {e}")
        print("Make sure QUAM_STATE_PATH environment variable is set or provide a path to Quam.load()")
        return
    
    # Extract data
    print("Extracting qubit grid locations...")
    qubit_grids = extract_qubit_grid_locations(machine)
    print(f"Found {len(qubit_grids)} qubits with grid locations")
    
    print("Extracting active qubits...")
    active_qubits = get_active_qubits(machine)
    print(f"Found {len(active_qubits)} active qubits")
    
    print("Extracting active qubit pairs...")
    active_pairs = get_active_qubit_pairs(machine)
    print(f"Found {len(active_pairs)} active qubit pairs")
    
    print("Extracting Bell state fidelities...")
    fidelities = extract_bell_state_fidelities(machine)
    print(f"Found {len(fidelities)} pairs with fidelity data")
    
    print("Extracting single qubit RB values (averaged) for active qubits...")
    # Extract RB values only for active qubits
    rb_values = extract_single_qubit_rb(machine, active_qubits)
    print(f"Found {len(rb_values)} active qubits with 1Q RB data")
    
    # Create plot
    print("\nGenerating plot...")
    output_file = Path(__file__).parent / 'qubit_grid_plot.png'
    plot_qubit_grid(qubit_grids, active_qubits, active_pairs, fidelities, rb_values,
                   output_file=str(output_file))
    
    print("\nDone!")


if __name__ == "__main__":
    main()