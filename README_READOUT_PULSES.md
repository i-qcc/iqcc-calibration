# Readout Pulses Summary

This document provides a comprehensive summary of all readout pulse classes available in `quam.components.pulses`.

## Overview

Readout pulses are special pulse types designed for quantum measurement operations. They differ from control pulses in that they:
- Have `operation = "measurement"` (vs. `"control"` for regular pulses)
- Include integration weights for signal processing
- Default digital marker is `"ON"` (vs. `None` for control pulses)

## Base Classes

### `BaseReadoutPulse` (Abstract)
**Location:** Base abstract class for all readout pulses

**Purpose:** Provides the foundation for readout pulse functionality, including:
- Integration weights management (iw1, iw2, iw3)
- Threshold support (`threshold`, `rus_exit_threshold`)
- Abstract method `integration_weights_function()` that must be implemented

**Key Features:**
- `operation: ClassVar[str] = "measurement"`
- `digital_marker: str = "ON"` (default)
- Abstract method for custom integration weights calculation
- Automatic config generation for integration weights

**When to Use:** Only when you need a completely custom integration weights function. Otherwise, use `ReadoutPulse`.

---

### `ReadoutPulse` (Abstract)
**Location:** Abstract base class for most readout pulses

**Purpose:** Provides a default implementation of integration weights suitable for most readout scenarios.

**Key Parameters:**
- `length` (int): Pulse length in samples
- `digital_marker` (str, list, optional): Digital marker, defaults to `"ON"`
- `integration_weights` (list[float] | list[tuple[float, int]]): 
  - List of floats (one per sample), OR
  - List of (weight, length) tuples
  - Default: `"#./default_integration_weights"` (uses `[(1, length)]`)
- `integration_weights_angle` (float): Rotation angle for integration weights in radians (default: 0)

**Key Features:**
- Default integration weights: `[(1, length)]` (uniform weights)
- Supports phase rotation via `integration_weights_angle`
- Converts float lists to (weight, length) tuples automatically
- Generates four integration weight sets: real, imag, minus_real, minus_imag

**When to Use:** As the base class for most custom readout pulse implementations.

---

## Concrete Readout Pulse Classes

### 1. `SquareReadoutPulse`
**Inheritance:** `ReadoutPulse` + `SquarePulse`

**Description:** A constant-amplitude square pulse for readout operations.

**Parameters:**
- `length` (int): Pulse length in samples
- `amplitude` (float): Constant amplitude in volts
- `axis_angle` (float, optional): IQ axis angle in radians
  - `None`: Single channel or I port of IQ channel
  - `0`: X axis for IQ channel
  - `π/2`: Y axis for IQ channel
- `digital_marker` (str, list, optional): Defaults to `"ON"`
- `integration_weights` (list[float] | list[tuple[float, int]], optional): Custom integration weights
- `integration_weights_angle` (float, optional): Rotation angle for weights

**Waveform:** Constant amplitude throughout the pulse duration.

**Use Cases:**
- Standard readout operations
- Simple, fast readout pulses
- When uniform amplitude is sufficient

**Example:**
```python
readout = SquareReadoutPulse(
    length=1000,  # 1000 samples
    amplitude=0.5,  # 0.5 V
    axis_angle=0  # X axis
)
```

---

### 2. `Square_zero_ReadoutPulse`
**Inheritance:** `ReadoutPulse` + `SquarePulse_zero`

**Description:** A square readout pulse with a zero-amplitude ringdown section at the end. Useful for allowing signal decay before integration.

**Parameters:**
- `length` (int): Total pulse length (square part + zero part) in samples
- `amplitude` (float): Amplitude of the square section in volts
- `zero_length` (int): Length of the zero-amplitude ringdown section in samples
- `axis_angle` (float, optional): IQ axis angle in radians
- `digital_marker` (str, list, optional): Defaults to `"ON"`
- `integration_weights` (list[float] | list[tuple[float, int]], optional): Custom integration weights
- `integration_weights_angle` (float, optional): Rotation angle for weights

**Waveform:** 
- Square section: Constant amplitude for `(length - zero_length)` samples
- Zero section: Zero amplitude for `zero_length` samples

**Use Cases:**
- Readout operations requiring ringdown time
- When signal needs to decay before integration
- Reducing artifacts from abrupt pulse termination

**Example:**
```python
readout = Square_zero_ReadoutPulse(
    length=1200,  # Total: 1200 samples
    amplitude=0.5,  # 0.5 V
    zero_length=200,  # 200 samples of zero at the end
    axis_angle=0
)
# Results in: 1000 samples at 0.5V, then 200 samples at 0V
```

---

### 3. `ConstantReadoutPulse` (Deprecated)
**Inheritance:** `SquareReadoutPulse`

**Description:** Deprecated alias for `SquareReadoutPulse`. 

**Status:** ⚠️ **Deprecated** - Use `SquareReadoutPulse` instead.

**Note:** This class issues a `DeprecationWarning` when instantiated. It is maintained for backward compatibility only.

---

## Class Hierarchy

```
Pulse (base)
  └── BaseReadoutPulse (abstract)
       └── ReadoutPulse (abstract)
            ├── SquareReadoutPulse
            │    └── ConstantReadoutPulse (deprecated)
            └── Square_zero_ReadoutPulse
```

## Common Parameters Across All Readout Pulses

All readout pulses share these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `length` | int | Required | Pulse length in samples |
| `digital_marker` | str \| list | `"ON"` | Digital marker configuration |
| `integration_weights` | list[float] \| list[tuple] | `[(1, length)]` | Integration weights |
| `integration_weights_angle` | float | `0` | Rotation angle for weights (radians) |
| `threshold` | float | `None` | Measurement threshold (from BaseReadoutPulse) |
| `rus_exit_threshold` | float | `None` | RUS exit threshold (from BaseReadoutPulse) |

## Integration Weights

All readout pulses automatically generate four integration weight sets:
- **iw1**: `cosine=real`, `sine=minus_imag`
- **iw2**: `cosine=imag`, `sine=real`
- **iw3**: `cosine=minus_imag`, `sine=minus_real`

These are calculated from the base integration weights with optional phase rotation.

## Choosing the Right Readout Pulse

| Use Case | Recommended Pulse |
|----------|-------------------|
| Simple constant readout | `SquareReadoutPulse` |
| Readout with ringdown | `Square_zero_ReadoutPulse` |
| Custom waveform readout | Inherit from `ReadoutPulse` + custom pulse |
| Custom integration weights | Inherit from `BaseReadoutPulse` |

## Notes

- All readout pulses have `operation = "measurement"` by default
- Digital marker defaults to `"ON"` for all readout pulses
- Integration weights can be specified per-sample (list of floats) or as (weight, length) tuples
- The `axis_angle` parameter allows rotation in the IQ plane for IQ channels
- Total pulse length for `Square_zero_ReadoutPulse` includes both square and zero sections
