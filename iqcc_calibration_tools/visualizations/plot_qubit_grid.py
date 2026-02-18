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
from datetime import datetime, timezone, timedelta
import re

from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam

# Constants
STALE_DASH_COLOR = "black"  # Dash shown when fidelity data is outdated or missing
OUTDATED_THRESHOLD_HOURS = 5  # Data older than this (hours) is considered outdated; change as needed
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


def is_within_last_hour(
    updated_at_str: Optional[str],
    threshold_hours: float = OUTDATED_THRESHOLD_HOURS,
) -> bool:
    """
    Return True if updated_at_str is within the given threshold (hours), False otherwise.
    Missing or invalid date/time (None, empty, wrong format) is treated as outdated → False.
    updated_at_str format: "YYYY-MM-DD HH:MM:SS GMT+N" (e.g. "2025-02-03 14:30:00 GMT+2").
    """
    if not updated_at_str or not isinstance(updated_at_str, str):
        return False  # no date & time → treat as outdated
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+GMT\+(\d+)$", updated_at_str.strip())
    if not match:
        return False
    try:
        dt_naive = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        tz_hours = int(match.group(2))
        tz = timezone(timedelta(hours=tz_hours))
        dt_aware = dt_naive.replace(tzinfo=tz)
        now = datetime.now(tz)
        return (now - dt_aware) <= timedelta(hours=threshold_hours)
    except (ValueError, TypeError):
        return False


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


def _safe_get_nested(obj, *keys, default=None):
    """
    Safely extract nested dictionary/attribute values.
    
    Args:
        obj: Object to extract from (dict or object with attributes)
        *keys: Keys/attributes to traverse
        default: Default value if extraction fails
    
    Returns:
        Extracted value or default
    """
    current = obj
    for key in keys:
        if current is None:
            return default
        try:
            if isinstance(current, dict):
                current = current.get(key)
            elif hasattr(current, 'get'):
                current = current.get(key)
            elif hasattr(current, '__getitem__'):
                current = current[key]
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return default
        except (KeyError, TypeError, AttributeError):
            return default
    return current if current is not None else default


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


def extract_bell_state_fidelities(machine: Quam) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], Optional[str]]]:
    """
    Extract Bell state fidelities and their updated_at strings for qubit pairs.
    Returns (fidelities dict, updated_at dict).
    """
    fidelities = {}
    updated_at = {}
    
    for pair_name, qubit_pair in machine.qubit_pairs.items():
        parts = pair_name.split('-')
        if len(parts) != 2:
            continue
        pair_tuple = (parts[0], parts[1])
        if not (hasattr(qubit_pair, 'macros') and qubit_pair.macros):
            continue
        for macro in qubit_pair.macros.values():
            if isinstance(macro, str) or not hasattr(macro, 'fidelity'):
                continue
            bell = _safe_get_nested(macro, 'fidelity', 'Bell_State')
            if bell is None:
                continue
            fidelity = _safe_get_nested(bell, 'Fidelity') if isinstance(bell, dict) else getattr(bell, 'Fidelity', None)
            if fidelity is not None:
                fidelities[pair_tuple] = fidelity
                upd = bell.get('updated_at') if isinstance(bell, dict) else getattr(bell, 'updated_at', None)
                updated_at[pair_tuple] = upd
                break
    return fidelities, updated_at


def extract_standard_rb_fidelities(machine: Quam) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], Optional[str]]]:
    """
    Extract Standard RB fidelities and their updated_at strings for qubit pairs.
    Returns (fidelities dict, updated_at dict).
    """
    fidelities = {}
    updated_at = {}
    
    for pair_name, qubit_pair in machine.qubit_pairs.items():
        parts = pair_name.split('-')
        if len(parts) != 2:
            continue
        pair_tuple = (parts[0], parts[1])
        if not (hasattr(qubit_pair, 'macros') and qubit_pair.macros):
            continue
        cz_macro = qubit_pair.macros.get('cz')
        if cz_macro is None or isinstance(cz_macro, str):
            continue
        std_rb = _safe_get_nested(cz_macro, 'fidelity', 'StandardRB')
        if std_rb is None:
            continue
        fidelity = _safe_get_nested(std_rb, 'average_gate_fidelity') if isinstance(std_rb, dict) else getattr(std_rb, 'average_gate_fidelity', None)
        if fidelity is not None:
            fidelities[pair_tuple] = fidelity
            upd = std_rb.get('updated_at') if isinstance(std_rb, dict) else getattr(std_rb, 'updated_at', None)
            updated_at[pair_tuple] = upd
    return fidelities, updated_at


def extract_single_qubit_rb(machine: Quam, qubit_names: Set[str]) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """
    Extract single qubit RB values and their updated_at strings.
    Returns (rb_values dict, updated_at dict).
    """
    rb_values = {}
    updated_at = {}
    
    for qubit_name in qubit_names:
        if qubit_name not in machine.qubits:
            continue
        qubit = machine.qubits[qubit_name]
        gate_fidelity = _safe_get_nested(qubit, 'gate_fidelity')
        if gate_fidelity is None:
            continue
        rb_value = gate_fidelity.get('averaged') if isinstance(gate_fidelity, dict) else getattr(gate_fidelity, 'averaged', None)
        if rb_value is not None:
            rb_values[qubit_name] = rb_value
            upd = gate_fidelity.get('averaged_updated_at') if isinstance(gate_fidelity, dict) else getattr(gate_fidelity, 'averaged_updated_at', None)
            updated_at[qubit_name] = upd
    return rb_values, updated_at


def plot_qubit_grid(
    qubit_grids: Dict[str, Tuple[int, int]],
    active_qubits: Set[str],
    active_pairs: List[Tuple[str, str]],
    fidelities: Optional[Dict[Tuple[str, str], float]] = None,
    rb_values: Optional[Dict[str, float]] = None,
    standard_rb_fidelities: Optional[Dict[Tuple[str, str], float]] = None,
    bell_updated_at: Optional[Dict[Tuple[str, str], Optional[str]]] = None,
    standard_rb_updated_at: Optional[Dict[Tuple[str, str], Optional[str]]] = None,
    rb_updated_at: Optional[Dict[str, Optional[str]]] = None,
    output_file: Optional[str] = None,
    outdated_threshold_hours: float = OUTDATED_THRESHOLD_HOURS,
):
    """
    Plot all qubits on a grid, highlighting active qubits and active pairs.
    If updated_at for a fidelity is older than outdated_threshold_hours, a "-" is shown instead.
    
    Args:
        qubit_grids: Dictionary mapping qubit names to (x, y) grid coordinates
        active_qubits: Set of active qubit names
        active_pairs: List of tuples (q1, q2) representing active qubit pairs
        fidelities: Dictionary mapping (q1, q2) tuples to Bell state fidelity values
        rb_values: Dictionary mapping qubit names to 1Q RB fidelity values
        standard_rb_fidelities: Dictionary mapping (q1, q2) tuples to Standard RB fidelity values
        bell_updated_at: Optional dict of (q1, q2) -> updated_at string for Bell fidelities
        standard_rb_updated_at: Optional dict of (q1, q2) -> updated_at string for Standard RB
        rb_updated_at: Optional dict of qubit name -> updated_at string for 1Q RB
        output_file: Optional path to save the plot
        outdated_threshold_hours: Data older than this (hours) is shown as outdated "-". Default from OUTDATED_THRESHOLD_HOURS.
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
            
            # Get fidelities for this pair (check both directions)
            pair_key = (control, target)
            reverse_key = (target, control)
            
            bell_fidelity = (fidelities.get(pair_key) or fidelities.get(reverse_key)) if fidelities else None
            standard_rb_fidelity = (standard_rb_fidelities.get(pair_key) or standard_rb_fidelities.get(reverse_key)) if standard_rb_fidelities else None
            bell_upd = (bell_updated_at or {}).get(pair_key) or (bell_updated_at or {}).get(reverse_key)
            std_rb_upd = (standard_rb_updated_at or {}).get(pair_key) or (standard_rb_updated_at or {}).get(reverse_key)
            bell_stale = bell_fidelity is not None and not is_within_last_hour(bell_upd, threshold_hours=outdated_threshold_hours)
            standard_rb_stale = standard_rb_fidelity is not None and not is_within_last_hour(std_rb_upd, threshold_hours=outdated_threshold_hours)
            pair_has_stale_or_missing = (
                (bell_fidelity is not None and bell_stale)
                or (standard_rb_fidelity is not None and standard_rb_stale)
                or (bell_fidelity is None and standard_rb_fidelity is None)
            )
            # Arrow black when any pair data is outdated or missing; else color by Bell fidelity
            if pair_has_stale_or_missing:
                arrow_color = "black"
            else:
                arrow_color = get_pair_color(bell_fidelity * 100) if bell_fidelity else "blue"
            
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
            
            # Add fidelity labels if available
            if bell_fidelity is not None or standard_rb_fidelity is not None:
                # Position label at midpoint of the arrow
                mid_x, mid_y = (x_start + x_end) / 2, (y_start + y_end) / 2
                
                # Calculate label offset
                is_vertical = length > 0 and abs(dx_norm) < 0.1
                if is_vertical:
                    offset_x, offset_y = LABEL_OFFSET * 1.5, 0
                elif length > 0:
                    offset_x, offset_y = -dy_norm * LABEL_OFFSET, dx_norm * LABEL_OFFSET
                else:
                    offset_x = offset_y = LABEL_OFFSET
                
                label_x, label_y = mid_x + offset_x, mid_y + offset_y
                
                # Build labels: symbol + value or " -" (outdated: symbol and dash both black, dash 2x size)
                labels = []  # list of (symbol_str, value_str, color_for_symbol, color_for_dash_or_value)
                if bell_fidelity is not None:
                    if bell_stale:
                        labels.append(("◆", " -", STALE_DASH_COLOR, STALE_DASH_COLOR))
                    else:
                        bell_pct = bell_fidelity * 100
                        c = get_pair_color(bell_pct)
                        labels.append((f"◆ {bell_pct:.2f}%", "", c, c))
                if standard_rb_fidelity is not None:
                    if standard_rb_stale:
                        labels.append(("■", " -", STALE_DASH_COLOR, STALE_DASH_COLOR))
                    else:
                        rb_pct = standard_rb_fidelity * 100
                        c = get_pair_color(rb_pct)
                        labels.append((f"■ {rb_pct:.2f}%", "", c, c))
                
                text_kwargs = dict(fontsize=9, ha="center", va="center", zorder=6, weight="bold")
                dash_fontsize = 18  # 2x pair label size for the dash
                if len(labels) == 2:
                    for i, (sym, val, c1, c2) in enumerate(labels):
                        dy = 0.04 if i == 0 else -0.04
                        if val:
                            ax.text(label_x, label_y + dy, sym, color=c1, **text_kwargs)
                            ax.text(label_x + 0.08, label_y + dy, val, color=c2, fontsize=dash_fontsize, ha="center", va="center", zorder=6, weight="bold")
                        else:
                            ax.text(label_x, label_y + dy, sym, color=c1, **text_kwargs)
                elif labels:
                    sym, val, c1, c2 = labels[0]
                    if val:
                        ax.text(label_x - 0.04, label_y, sym, color=c1, **text_kwargs)
                        ax.text(label_x + 0.04, label_y, val, color=c2, fontsize=dash_fontsize, ha="center", va="center", zorder=6, weight="bold")
                    else:
                        ax.text(label_x, label_y, sym, color=c1, **text_kwargs)
    
    # Plot all qubits
    for qubit_name, (x, y) in qubit_grids.items():
        is_active = qubit_name in active_qubits
        rb_stale = is_active and rb_values and qubit_name in rb_values and not is_within_last_hour((rb_updated_at or {}).get(qubit_name), threshold_hours=outdated_threshold_hours)
        
        # Determine qubit color: gray (like inactive) when 1Q fidelity is outdated, else by RB or green
        if rb_stale:
            qubit_color = 'lightgray'
            border_color = 'gray'
        elif is_active and rb_values and qubit_name in rb_values:
            rb_value = rb_values[qubit_name]
            rb_pct = rb_value * 100
            qubit_color = get_qubit_color(rb_pct)
            border_color = 'black'
        elif is_active:
            qubit_color = 'green'
            border_color = 'black'
        else:
            qubit_color = 'lightgray'
            border_color = 'gray'
        
        radius = QUBIT_RADIUS_ACTIVE if is_active else QUBIT_RADIUS_INACTIVE
        circle = plt.Circle((x, y), radius, color=qubit_color, zorder=3 if is_active else 2)
        ax.add_patch(circle)
        
        if is_active:
            border = plt.Circle((x, y), radius, fill=False, edgecolor=border_color,
                              linewidth=2, zorder=4)
            ax.add_patch(border)
        
        # Build qubit label: name + RB % or " -" when stale
        text_kw = dict(fontsize=7, ha="center", va="center", zorder=6, weight="bold" if is_active else "normal")
        if is_active and rb_values and qubit_name in rb_values:
            if rb_stale:
                ax.text(x, y + 0.06, qubit_name, color="black", **text_kw)  # black on lightgray
                ax.text(x, y - 0.06, "-", color=STALE_DASH_COLOR, fontsize=14, ha="center", va="center", zorder=6, weight="bold")  # 2x qubit fontsize
            else:
                rb_value = rb_values[qubit_name]
                rb_pct = rb_value * 100
                if rb_pct > 0:
                    digits = 4 - int(floor(log10(abs(rb_pct))))
                    rb_str = f"{rb_pct:.{max(0, digits-1)}f}%"
                else:
                    rb_str = f"{rb_pct:.3f}%"
                ax.text(x, y, f"{qubit_name}\n{rb_str}", color="white", **text_kw)
        else:
            label_text = qubit_name
            text_color = "white" if is_active else "black"
            ax.text(x, y, label_text, color=text_color, **text_kw)
    
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
                            label='Active Pair\n(control→target)')
    bell_state_label = plt.Line2D([0], [0], color='blue', linewidth=0, marker='D',
                                  markersize=8, markerfacecolor='blue',
                                  markeredgecolor='blue', markeredgewidth=1,
                                  label='Bell State Fidelity (%)')
    standard_rb_label = plt.Line2D([0], [0], color='blue', linewidth=0, marker='s',
                                   markersize=8, markerfacecolor='blue',
                                   markeredgecolor='blue', markeredgewidth=1,
                                   label='Standard RB Fidelity (%)')
    rb_label = plt.Line2D([0], [0], color='green', linewidth=0, marker='o',
                          markersize=10, markerfacecolor='green',
                          markeredgecolor='darkgreen', markeredgewidth=1,
                          label='1Q RB Fidelity (%)')
    _h = int(outdated_threshold_hours) if outdated_threshold_hours == int(outdated_threshold_hours) else outdated_threshold_hours
    stale_dash_label = (
        f'"-" = fidelity data\nolder than {_h} h or missing'
    )
    # Use a short line (dash) as legend handle instead of a patch, so it looks like a normal "-"
    stale_dash_handle = plt.Line2D([0, 1], [0, 0], color=STALE_DASH_COLOR, linewidth=2.5, label=stale_dash_label)
    ax.legend(handles=[active_patch, inactive_patch, pair_arrow,
                      bell_state_label, standard_rb_label, rb_label, stale_dash_handle],
             loc='upper left', fontsize=9, framealpha=0.9)
    
    # Add statistics text box in lower left to avoid overlap with legend
    stats_text = f'Total Qubits: {len(qubit_grids)}\n'
    stats_text += f'Active Qubits: {len(active_qubits)}\n'
    stats_text += f'Active Pairs: {len(active_pairs)}'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add average fidelities in top right
    fidelity_stats_lines = []
    if rb_values:
        vals_1q = np.array(list(rb_values.values())) * 100
        avg_1q, std_1q = np.mean(vals_1q), np.std(vals_1q)
        fidelity_stats_lines.append(f"Avg 1Q RB: {avg_1q:.2f} ± {std_1q:.2f}%")
    else:
        fidelity_stats_lines.append("Avg 1Q RB: N/A")
    if standard_rb_fidelities:
        vals_2q = np.array(list(standard_rb_fidelities.values())) * 100
        avg_2q, std_2q = np.mean(vals_2q), np.std(vals_2q)
        fidelity_stats_lines.append(f"Avg 2Q RB: {avg_2q:.2f} ± {std_2q:.2f}%")
    else:
        fidelity_stats_lines.append("Avg 2Q RB: N/A")
    if fidelities:
        vals_bell = np.array(list(fidelities.values())) * 100
        avg_bell, std_bell = np.mean(vals_bell), np.std(vals_bell)
        fidelity_stats_lines.append(f"Avg Bell State: {avg_bell:.2f} ± {std_bell:.2f}%")
    else:
        fidelity_stats_lines.append("Avg Bell State: N/A")
    fidelity_stats_text = "\n".join(fidelity_stats_lines)
    ax.text(0.98, 0.98, fidelity_stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
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
    fidelities, bell_updated_at = extract_bell_state_fidelities(machine)
    print(f"Found {len(fidelities)} pairs with Bell state fidelity data")
    
    print("Extracting Standard RB fidelities...")
    standard_rb_fidelities, standard_rb_updated_at = extract_standard_rb_fidelities(machine)
    print(f"Found {len(standard_rb_fidelities)} pairs with Standard RB fidelity data")
    
    print("Extracting single qubit RB values (averaged) for active qubits...")
    rb_values, rb_updated_at = extract_single_qubit_rb(machine, active_qubits)
    print(f"Found {len(rb_values)} active qubits with 1Q RB data")
    
    print("\nGenerating plot...")
    output_file = Path(__file__).parent / 'qubit_grid_plot.png'
    plot_qubit_grid(
        qubit_grids, active_qubits, active_pairs,
        fidelities, rb_values, standard_rb_fidelities,
        bell_updated_at=bell_updated_at,
        standard_rb_updated_at=standard_rb_updated_at,
        rb_updated_at=rb_updated_at,
        output_file=str(output_file),
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()