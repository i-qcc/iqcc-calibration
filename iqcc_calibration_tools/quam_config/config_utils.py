"""Utility functions for QUA config generation and manipulation."""

from typing import Dict, Any


def get_clean_config(machine) -> Dict[str, Any]:
    """Generate QUA config from machine and remove deprecated 'version' parameter.
    
    The quam library's generate_config() method adds a 'version' parameter
    that is deprecated in QUA 1.2.2+. This function removes it to avoid
    deprecation warnings.
    
    Args:
        machine: A Quam instance with a generate_config() method.
        
    Returns:
        dict: The QUA config dictionary without the 'version' key.
    """
    config = machine.generate_config()
    # Remove deprecated 'version' parameter if present
    if isinstance(config, dict) and "version" in config:
        config.pop("version")
    return config

