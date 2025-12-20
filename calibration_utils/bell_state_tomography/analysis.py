"""
Analysis utilities for Bell state tomography.
"""
import numpy as np
import xarray as xr


def flatten(data):
    """
    Recursively flatten a nested tuple structure.
    
    Parameters:
    -----------
    data : tuple or other
        Data to flatten
    
    Returns:
    --------
    tuple
        Flattened tuple
    """
    if isinstance(data, tuple):
        if len(data) == 0:
            return ()
        else:
            return flatten(data[0]) + flatten(data[1:])
    else:
        return (data,)


def generate_pauli_basis(n_qubits):
    """
    Generate Pauli basis for n qubits.
    
    Parameters:
    -----------
    n_qubits : int
        Number of qubits
    
    Returns:
    --------
    list
        List of Pauli basis elements as tuples
    """
    pauli = np.array([0, 1, 2, 3])
    paulis = pauli
    for i in range(n_qubits - 1):
        new_paulis = []
        for ps in paulis:
            for p in pauli:
                new_paulis.append(flatten((ps, p)))
        paulis = new_paulis
    return paulis


def gen_inverse_hadamard(n_qubits):
    """
    Generate inverse Hadamard matrix for n qubits.
    
    Parameters:
    -----------
    n_qubits : int
        Number of qubits
    
    Returns:
    --------
    np.ndarray
        Inverse Hadamard matrix
    """
    H = np.array([[1, 1], [1, -1]]) / 2
    for _ in range(n_qubits - 1):
        H = np.kron(H, H)
    return np.linalg.inv(H)


def get_pauli_data(da):
    """
    Extract Pauli operator data from tomography dataset.
    
    Parameters:
    -----------
    da : xarray.DataArray
        Tomography data with tomo_axis coordinate
    
    Returns:
    --------
    xarray.Dataset
        Dataset containing Pauli operator values and appearances
    """
    pauli_basis = generate_pauli_basis(2)

    inverse_hadamard = gen_inverse_hadamard(2)

    # Create an xarray Dataset with dimensions and coordinates based on pauli_basis
    paulis_data = xr.Dataset(
        {
            "value": (["pauli_op"], np.zeros(len(pauli_basis))),
            "appearances": (["pauli_op"], np.zeros(len(pauli_basis), dtype=int))
        },
        coords={'pauli_op': [','.join(map(str, op)) for op in pauli_basis]}
    )

    for tomo_axis in da.coords['tomo_axis'].values:
        tomo_data = da.sel(tomo_axis=tomo_axis)
        pauli_data = inverse_hadamard @ tomo_data.data
        paulis = ["0,0", f"{tomo_axis[0]+1},0", f"0,{tomo_axis[1]+1}", f"{tomo_axis[0]+1},{tomo_axis[1]+1}"]
        for i, pauli in enumerate(paulis):
            paulis_data.value.loc[{'pauli_op': pauli}] += pauli_data[i]
            paulis_data.appearances.loc[{'pauli_op': pauli}] += 1

    paulis_data = xr.where(paulis_data.appearances != 0, paulis_data.value / paulis_data.appearances, paulis_data.value)

    return paulis_data


def get_density_matrix(paulis_data):
    """
    Convert Pauli operator data to density matrix.
    
    Parameters:
    -----------
    paulis_data : xarray.Dataset
        Dataset containing Pauli operator values
    
    Returns:
    --------
    np.ndarray
        4x4 density matrix (for 2 qubits)
    """
    # 2Q
    # Define the Pauli matrices
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    # Create a vector of the Pauli matrices
    pauli_matrices = [I, X, Y, Z]

    rho = np.zeros((4, 4))

    for i, pauli_i in enumerate(pauli_matrices):
        for j, pauli_j in enumerate(pauli_matrices):
            rho = rho + 0.25 * paulis_data.sel(pauli_op=f"{i},{j}").values * np.kron(pauli_i, pauli_j)

    return rho

