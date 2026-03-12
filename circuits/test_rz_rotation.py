# %% [markdown]
# # RZ rotation angle & direction test
#
# Verifies that the RZ gate rotates by the correct angle in the correct
# direction.  The circuit for each angle θ is:
#
#     |0⟩ → H → RZ(θ) → H → measure
#
# Theory: P(|1⟩) = sin²(θ/2), which traces a smooth curve from 0 to 1 and
# back as θ goes from 0 to 2π.

# %%
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
from circuit_utils import Backend

# %% Connect to backend
backend = Backend("arbel")

# %% Build circuits

def build_rz_circuit(theta: float) -> QuantumCircuit:
    """  |0⟩ → H → RZ(θ) → H → measure  ⟹  P(|1⟩) = sin²(θ/2)  """
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(theta, 0)
    qc.h(0)
    return qc


angles = np.linspace(0, 2 * np.pi, 17)
circuits = [build_rz_circuit(theta) for theta in angles]
circuits = transpile(circuits, basis_gates=backend.native_gates, optimization_level=1)

# %% Run
results = backend.run(circuits, qubits=["qA1"], num_shots=1000)

# %% Print results
ideal = np.sin(angles / 2) ** 2

print("\nResults:")
for theta, p, ideal_p in zip(angles, results.probabilities, ideal):
    print(f"  θ = {theta:5.3f} rad ({np.degrees(theta):6.1f}°)  ->  "
          f"P(|1>) = {p:.4f}  (ideal {ideal_p:.4f})")

# %% Plot
fig, ax = plt.subplots(figsize=(10, 5))

theta_smooth = np.linspace(0, 2 * np.pi, 200)
ax.plot(theta_smooth, np.sin(theta_smooth / 2) ** 2,
        color="#10B981", linewidth=1.5, alpha=0.6, label="Ideal sin²(θ/2)")
ax.plot(angles, results.probabilities, "o", color="#2563EB", markersize=8,
        label="Measured P(|1>)")

ax.set_xlabel("RZ angle θ (rad)", fontsize=13)
ax.set_ylabel("P(|1>)", fontsize=13)
ax.set_title(
    f"RZ rotation test  |  qA1  |  {results.num_shots} shots",
    fontsize=14,
)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %%
