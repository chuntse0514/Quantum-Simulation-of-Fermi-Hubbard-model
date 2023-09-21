import numpy as np

def fourier_transform_matrix(x_dimension, y_dimension):

    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    
    FT_matrix = np.zeros((n_spin_orbitals, n_spin_orbitals), dtype=complex)

    Nx = x_dimension
    Ny = y_dimension
    
    def index2tuple(index):
        return (index % Nx, index // Nx, index % 2)
        
    for row in range(n_spin_orbitals):
        for column in range(n_spin_orbitals):
            nx, ny, spin1 = index2tuple(row)
            mx, my, spin2 = index2tuple(column)

            if spin1 != spin2:
                continue

            FT_matrix[row, column] = np.exp(-1j * 2 * np.pi * mx * nx / Nx) * \
                                     np.exp(-1j * 2 * np.pi * my * ny / Ny)
            
    return FT_matrix / np.sqrt(n_sites)