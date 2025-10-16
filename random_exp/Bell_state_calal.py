# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def calculate_two_qubit_density_matrix(cphase_phase, q1_phase, q2_phase):
    """
    Calculate the density matrix of two qubits under the following sequence:
    1. Hadamard on q1
    2. Hadamard on q2  
    3. CPhase gate between q1-q2 with given phase
    4. Phase rotation on q1
    5. Phase rotation on q2
    6. Hadamard on q1
    
    Parameters:
    -----------
    cphase_phase : float
        Phase parameter for the CPhase gate (in radians)
    q1_phase : float
        Phase rotation parameter for qubit 1 (in radians)
    q2_phase : float
        Phase rotation parameter for qubit 2 (in radians)
    
    Returns:
    --------
    numpy.ndarray
        4x4 density matrix
    """
    
    # Define Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.array([[1, 0], [0, 1]], dtype=complex)
    
    # Hadamard gate
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    
    # Phase rotation gate
    def phase_rotation(phase):
        return np.array([[1, 0], [0, np.exp(1j * phase)]], dtype=complex)
    
    # CPhase gate (controlled-Z with phase)
    def cphase_gate(phase):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * phase)]
        ], dtype=complex)
    
    # Start with |00⟩ state
    psi_initial = np.array([1, 0, 0, 0], dtype=complex)
    
    # Apply gates in sequence
    # 1. Hadamard on q1
    U1 = np.kron(H, identity)
    psi = U1 @ psi_initial
    
    # 2. Hadamard on q2
    U2 = np.kron(identity, H)
    psi = U2 @ psi
    
    # 3. CPhase gate
    U3 = cphase_gate(cphase_phase)
    psi = U3 @ psi
    
    # 4. Phase rotation on q1
    U4 = np.kron(phase_rotation(q1_phase), identity)
    psi = U4 @ psi
    
    # 5. Phase rotation on q2
    U5 = np.kron(identity, phase_rotation(q2_phase))
    psi = U5 @ psi
    
    # 6. Hadamard on q1
    U6 = np.kron(H, identity)
    psi = U6 @ psi
    
    # Calculate density matrix
    rho = np.outer(psi, np.conj(psi))
    
    return rho

def plot_density_matrix(rho, title="Two-Qubit Density Matrix", figsize=(12, 5), vmin=-1, vmax=1):
    """
    Plot the density matrix in a nice visualization format.
    
    Parameters:
    -----------
    rho : numpy.ndarray
        4x4 density matrix
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    vmin, vmax : float
        Fixed color scale limits for consistent visualization
    """
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate color scale limits based on the data
    real_max = np.max(np.abs(np.real(rho)))
    imag_max = np.max(np.abs(np.imag(rho)))
    
    # Use the maximum of both real and imaginary parts for consistent scaling
    max_val = max(real_max, imag_max)
    max_val = 1
    
    # Use symmetric color scale around zero for both plots
    vmin, vmax = -max_val, max_val
    
    # Plot 1: Real part
    im1 = ax1.imshow(np.real(rho), cmap='RdBu_r', aspect='equal', 
                     vmin=vmin, vmax=vmax)
    ax1.set_title('Real Part')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Add colorbar for real part
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Real Value')
    
    # Add text annotations for real part
    for i in range(4):
        for j in range(4):
            text_color = 'white' if abs(np.real(rho[i, j])) > max_val/2 else 'black'
            text = ax1.text(j, i, f'{np.real(rho[i, j]):.3f}',
                          ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')
    
    # Plot 2: Imaginary part
    im2 = ax2.imshow(np.imag(rho), cmap='RdBu_r', aspect='equal',
                     vmin=vmin, vmax=vmax)
    ax2.set_title('Imaginary Part')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # Add colorbar for imaginary part
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Imaginary Value')
    
    # Add text annotations for imaginary part
    for i in range(4):
        for j in range(4):
            text_color = 'white' if abs(np.imag(rho[i, j])) > max_val/2 else 'black'
            text = ax2.text(j, i, f'{np.imag(rho[i, j]):.3f}',
                          ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')
    
    # Set axis labels
    basis_states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    for ax in [ax1, ax2]:
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(basis_states)
        ax.set_yticklabels(basis_states)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print some properties of the density matrix
    print(f"\nDensity Matrix Properties:")
    print(f"Trace: {np.trace(rho):.6f}")
    print(f"Is Hermitian: {np.allclose(rho, rho.conj().T)}")
    print(f"Is Positive Semi-definite: {np.all(np.linalg.eigvals(rho) >= -1e-10)}")
    print(f"Purity: {np.trace(rho @ rho):.6f}")
    print(f"Color scale range: [{vmin:.3f}, {vmax:.3f}]")
    print(f"Real part range: [{real_max:.3f}]")
    print(f"Imaginary part range: [{imag_max:.3f}]")

def plot_density_matrix_city(rho, title="Two-Qubit Density Matrix - City Plot"):
    """
    Create a city plot (bar chart) visualization of the density matrix.
    
    Parameters:
    -----------
    rho : numpy.ndarray
        4x4 density matrix
    title : str
        Title for the plot
    """
    
    fig = plt.figure(figsize=(12, 5))
    
    # Calculate color scale limits
    real_max = np.max(np.abs(np.real(rho)))
    imag_max = np.max(np.abs(np.imag(rho)))
    
    # Use the maximum of both real and imaginary parts for consistent scaling
    max_val = max(real_max, imag_max)
    vmin, vmax = -max_val, max_val
    
    # Create city plot for real part
    ax1 = fig.add_subplot(121, projection='3d')
    x, y = np.meshgrid(range(4), range(4))
    z_real = np.real(rho)
    
    # Create bars for each matrix element
    for i in range(4):
        for j in range(4):
            height = z_real[i, j]
            color = 'red' if height >= 0 else 'blue'
            alpha = abs(height) / max_val if max_val > 0 else 0
            ax1.bar3d(j, i, 0, 0.8, 0.8, height, color=color, alpha=alpha)
    
    ax1.set_title('Real Part (City Plot)')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.set_zlabel('Real Value')
    ax1.set_zlim(vmin, vmax)
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
    ax1.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
    
    # Add zero plane
    xx, yy = np.meshgrid(range(4), range(4))
    zz = np.zeros_like(xx)
    ax1.plot_surface(xx, yy, zz, alpha=0.3, color='gray', linewidth=0.5)
    
    # Create city plot for imaginary part
    ax2 = fig.add_subplot(122, projection='3d')
    z_imag = np.imag(rho)
    
    # Create bars for each matrix element
    for i in range(4):
        for j in range(4):
            height = z_imag[i, j]
            color = 'red' if height >= 0 else 'blue'
            alpha = abs(height) / max_val if max_val > 0 else 0
            ax2.bar3d(j, i, 0, 0.8, 0.8, height, color=color, alpha=alpha)
    
    ax2.set_title('Imaginary Part (City Plot)')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.set_zlabel('Imaginary Value')
    ax2.set_zlim(vmin, vmax)
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
    ax2.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
    
    # Add zero plane
    ax2.plot_surface(xx, yy, zz, alpha=0.3, color='gray', linewidth=0.5)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
# %%


cphase_phase = np.pi*1  # π/2 radians
q1_phase = 0.3*np.pi    # π/4 radians  
q2_phase = 0*np.pi      # π/6 radians

print("Calculating density matrix for two-qubit circuit...")
print(f"CPhase phase: {cphase_phase:.3f} rad")
print(f"Q1 phase rotation: {q1_phase:.3f} rad")
print(f"Q2 phase rotation: {q2_phase:.3f} rad")

# Calculate density matrix
rho = calculate_two_qubit_density_matrix(cphase_phase, q1_phase, q2_phase)

print(f"\nDensity Matrix:")
print(rho)

# Plot the density matrix
plot_density_matrix(rho, "Two-Qubit Circuit Density Matrix")

# Plot city plot visualization
plot_density_matrix_city(rho, "Two-Qubit Circuit - City Plot")

# %%
