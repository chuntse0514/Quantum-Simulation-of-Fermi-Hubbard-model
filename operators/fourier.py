import numpy as np
from openfermion import FermionOperator

def fourier_transform_matrix(x_dimension, y_dimension):

    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    
    FT_matrix = np.zeros((n_spin_orbitals, n_spin_orbitals), dtype=complex)

    Nx = x_dimension
    Ny = y_dimension
    
    def index2tuple(index):
        return ((index // 2) % Nx, (index // 2) // Nx, index % 2)
        
    for row in range(n_spin_orbitals):
        for column in range(n_spin_orbitals):
            nx, ny, spin1 = index2tuple(row)
            mx, my, spin2 = index2tuple(column)

            if spin1 != spin2:
                continue

            FT_matrix[row, column] = np.exp(-1j * 2 * np.pi * mx * nx / Nx) * \
                                     np.exp(-1j * 2 * np.pi * my * ny / Ny)
            
    return FT_matrix / np.sqrt(n_sites)

def fourier_transform(hamiltonian, Nx, Ny):

    def index2tuple(index, spin=False):
        if spin:
            return ((index // 2) % Nx, (index // 2) // Nx, index % 2)
        else:
            return (index % Nx, index // Nx)
        
    def tuple2index(ix, iy, spin):
        return 2 * (ix + Nx * iy) + spin
        
    n_sites = Nx * Ny

    FT_hamiltonian = FermionOperator()
    for term, coeff in hamiltonian.terms.items():
        
        FT_term = FermionOperator.identity()
        for r, ladder in term:
            
            FT_basis = FermionOperator()
            rx, ry, spin = index2tuple(r, spin=True)

            for k in range(n_sites):
                kx, ky = index2tuple(k, spin=False)
                k_sigma = tuple2index(kx, ky, spin)
                FT_basis += FermionOperator((k_sigma, ladder), np.exp(-1j * 2 * np.pi * (kx * rx / Nx + ky * ry / Ny)) / np.sqrt(n_sites))

            FT_term *= FT_basis
        FT_hamiltonian += FT_term * coeff
    FT_hamiltonian.compress()

    return FT_hamiltonian