# %% [markdown]
# # X-gate parity test
#
# Runs circuits with an increasing number of X gates in a single batch job.
# Each circuit starts with one X gate to initialize |1⟩, followed by N
# additional X gates (powers of 2).  The expected state alternates between
# |0⟩ and |1⟩ depending on parity.

# %%
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
from circuit_utils import Backend

# %% Connect to backend
backend = Backend("arbel")

# %% Build circuits

def build_x_circuit(n_x: int) -> QuantumCircuit:
    """Single-qubit circuit: one X to initialize |1⟩, then *n_x* additional X gates."""
    qc = QuantumCircuit(1)
    qc.x(0)
    for _ in range(n_x):
        qc.x(0)
    return qc


n_x_gates = [2**k for k in range(1, 9)]
circuits = [build_x_circuit(n) for n in n_x_gates]
circuits = transpile(circuits, basis_gates=backend.native_gates, optimization_level=1)

# %% Run
results = backend.run(circuits, qubits=["qA1"], num_shots=1000)

# %% Print results
print("\nResults (1 init X + N additional X gates):")
for n, p in zip(n_x_gates, results.probabilities):
    total = 1 + n
    expected = 1.0 if (total % 2 == 1) else 0.0
    print(f"  1+{n:<3} = {total:<3} X gates  ->  P(|1>) = {p:.4f}  (expected {expected:.1f})")

# %% Plot
fig, ax = plt.subplots(figsize=(10, 5))

expected = [1.0 if ((1 + n) % 2 == 1) else 0.0 for n in n_x_gates]

ax.plot(n_x_gates, results.probabilities, "o-", color="#2563EB", markersize=8,
        linewidth=2, label="Measured P(|1>)")
ax.plot(n_x_gates, expected, "s--", color="#10B981", markersize=6, alpha=0.6,
        label="Ideal")

ax.set_xlabel("Additional X gates after initialization", fontsize=13)
ax.set_xscale("log", base=2)
ax.set_ylabel("P(|1>)", fontsize=13)
ax.set_title(
    f"X-gate parity test  |  qA1  |  {results.num_shots} shots",
    fontsize=14,
)
ax.set_xticks(n_x_gates)
ax.set_xticklabels(n_x_gates)
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %%
