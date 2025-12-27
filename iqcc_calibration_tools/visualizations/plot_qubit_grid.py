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

# Constants
QUBIT_RADIUS_ACTIVE = 0.25
QUBIT_RADIUS_INACTIVE = 0.15
ARROWHEAD_LENGTH = 0.15
ARROWHEAD_WIDTH = 0.12
ARROW_LINEWIDTH = 3
ARROW_ALPHA = 0.6
LABEL_OFFSET = 0.15


def parse_grid_location(location_str: str) -> Tuple[int, int]:
    """Parse grid location string like '2,4' into (x, y) tuple."""
    x, y = map(int, location_str.split(','))
    return (x, y)


def _interpolate_color(normalized: float, red_start: float = 0.7) -> Tuple[float, float, float]:
    """
    Interpolate color from dark red through orange to dark green.
    
    Args:
        normalized: Normalized value in [0, 1]
        red_start: Starting red value (0.7 for pairs, 0.9 for qubits)
    
    Returns:
        RGB tuple
    """
    if normalized < 0.5:
        # Dark red to orange
        t = normalized * 2
        r = red_start + t * (1.0 - red_start)
        g = t * 0.647  # 165/255 for orange
        b = 0.0
    else:
        # Orange to dark green
        t = (normalized - 0.5) * 2
        r = 1.0 - t * 1.0
        g = 0.647 + t * (0.5 - 0.647)
        b = 0.0
    return (r, g, b)


def get_qubit_color(rb_fidelity_pct: float) -> str:
    """
    Get color for qubit based on 1Q RB fidelity.
    
    Args:
        rb_fidelity_pct: RB fidelity as percentage (e.g., 98.5 for 98.5%)
    
    Returns:
        Color string (hex format)
    """
    min_fidelity, max_fidelity = 98.5, 99.9
    normalized = np.clip((rb_fidelity_pct - min_fidelity) / (max_fidelity - min_fidelity), 0, 1)
    return mcolors.rgb2hex(_interpolate_color(normalized, red_start=0.9))


def get_pair_color(bell_fidelity_pct: float) -> str:
    """
    Get color for qubit pair based on Bell state fidelity.
    
    Args:
        bell_fidelity_pct: Bell state fidelity as percentage (e.g., 95.0 for 95.0%)
    
    Returns:
        Color string (hex format)
    """
    min_fidelity, max_fidelity = 92.5, 99.0
    normalized = np.clip((bell_fidelity_pct - min_fidelity) / (max_fidelity - min_fidelity), 0, 1)
    return mcolors.rgb2hex(_interpolate_color(normalized, red_start=0.7))


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


def _get_qubit_name(qubit_obj) -> str:
    """Extract qubit name from qubit object."""
    return qubit_obj.name if hasattr(qubit_obj, 'name') else str(qubit_obj)


def get_active_qubit_pairs(machine: Quam) -> List[Tuple[str, str]]:
    """
    Get list of active qubit pairs as tuples of (control, target).
    Preserves the control->target direction from the pair objects.
    """
    pairs = []
    for pair_name in machine.active_qubit_pair_names:
        pair_obj = machine.qubit_pairs.get(pair_name)
        if pair_obj and hasattr(pair_obj, 'qubit_control') and hasattr(pair_obj, 'qubit_target'):
            pairs.append((_get_qubit_name(pair_obj.qubit_control), 
                         _get_qubit_name(pair_obj.qubit_target)))
        else:
            # Fallback: parse pair name (e.g., "qB1-qB2")
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
    
    # Draw active qubit pairs (arrows) first, so they appear behind qubits
    for control, target in active_pairs:
        if control in qubit_grids and target in qubit_grids:
            x1, y1 = qubit_grids[control]
            x2, y2 = qubit_grids[target]
            
            # Determine arrow color based on Bell state fidelity
            fidelity = None
            if fidelities:
                fidelity = fidelities.get((control, target)) or fidelities.get((target, control))
            
            arrow_color = get_pair_color(fidelity * 100) if fidelity else 'blue'
            
            # Calculate arrow direction and adjust start/end points
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            
            if length > 0:
                # Normalize direction vector
                dx_norm, dy_norm = dx / length, dy / length
                
                # Adjust start/end points to qubit circle edges
                x_start = x1 + dx_norm * QUBIT_RADIUS_ACTIVE
                y_start = y1 + dy_norm * QUBIT_RADIUS_ACTIVE
                x_end = x2 - dx_norm * QUBIT_RADIUS_ACTIVE
                y_end = y2 - dy_norm * QUBIT_RADIUS_ACTIVE
                
                # Draw arrow line
                ax.plot([x_start, x_end], [y_start, y_end], 
                       color=arrow_color, linewidth=ARROW_LINEWIDTH, 
                       alpha=ARROW_ALPHA, zorder=1)
                
                # Draw unfilled arrowhead
                perp_x = -dy_norm * ARROWHEAD_WIDTH / 2
                perp_y = dx_norm * ARROWHEAD_WIDTH / 2
                base_x1 = x_end - dx_norm * ARROWHEAD_LENGTH + perp_x
                base_y1 = y_end - dy_norm * ARROWHEAD_LENGTH + perp_y
                base_x2 = x_end - dx_norm * ARROWHEAD_LENGTH - perp_x
                base_y2 = y_end - dy_norm * ARROWHEAD_LENGTH - perp_y
                
                arrow_props = dict(color=arrow_color, linewidth=ARROW_LINEWIDTH, 
                                  alpha=ARROW_ALPHA, zorder=2)
                ax.plot([x_end, base_x1], [y_end, base_y1], **arrow_props)
                ax.plot([x_end, base_x2], [y_end, base_y2], **arrow_props)
            
            # Add fidelity label if available
            if fidelity is not None:
                fidelity_pct = fidelity * 100
                # Position label at midpoint of the arrow
                mid_x = (x_start + x_end) / 2
                mid_y = (y_start + y_end) / 2
                
                # Offset perpendicular to the arrow to avoid overlap
                if length > 0:
                    offset_x = -dy_norm * LABEL_OFFSET
                    offset_y = dx_norm * LABEL_OFFSET
                else:
                    offset_x = offset_y = LABEL_OFFSET
                
                label_x = mid_x + offset_x
                label_y = mid_y + offset_y
                
                # Format fidelity as percentage with 2 decimal places
                ax.text(label_x, label_y, f'{fidelity_pct:.2f}%', 
                       fontsize=7, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               edgecolor=arrow_color, alpha=0.8, linewidth=1),
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
        
        radius = QUBIT_RADIUS_ACTIVE if is_active else QUBIT_RADIUS_INACTIVE
        circle = plt.Circle((x, y), radius, color=qubit_color, zorder=3 if is_active else 2)
        ax.add_patch(circle)
        
        if is_active:
            # Add border for active qubits
            border = plt.Circle((x, y), radius, fill=False, edgecolor=border_color, 
                              linewidth=2, zorder=4)
            ax.add_patch(border)
        
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
    ax.set_title('Qubit Grid Layout\n(Active qubits in green, Active pairs shown as arrows: control→target)', 
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
    # Create an arrow line for the legend using Line2D with right arrow marker
    pair_arrow = plt.Line2D([0], [0], color='blue', linewidth=2.5, 
                            marker='>', markersize=10, markeredgecolor='blue',
                            label='Active Pair (control→target)')
    fidelity_label = plt.Line2D([0], [0], color='blue', linewidth=0, marker='s', 
                                markersize=8, markerfacecolor='white', 
                                markeredgecolor='blue', markeredgewidth=1,
                                label='Bell State Fidelity (%)')
    rb_label = plt.Line2D([0], [0], color='green', linewidth=0, marker='o', 
                          markersize=10, markerfacecolor='green', 
                          markeredgecolor='darkgreen', markeredgewidth=1,
                          label='1Q RB Fidelity (%)')
    ax.legend(handles=[active_patch, inactive_patch, pair_arrow, fidelity_label, rb_label], 
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