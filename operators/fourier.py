import numpy as np
from openfermion import FermionOperator, normal_ordered

def round_operator(op):

    new_op = FermionOperator()
    for term, coeff in op.terms.items():
        coeff = np.round(coeff, decimals=6)
        new_op += FermionOperator(term ,coeff)

    return new_op

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
        for n, ladder in term:
            
            FT_basis = FermionOperator()
            nx, ny, spin = index2tuple(n, spin=True)

            for m in range(n_sites):
                mx, my = index2tuple(m, spin=False)
                m_sigma = tuple2index(mx, my, spin)
                # creation operator
                if ladder:
                    FT_basis += FermionOperator((m_sigma, ladder), np.exp(-1j * 2 * np.pi * (mx * nx / Nx + my * ny / Ny)) / np.sqrt(n_sites))
                # annihilation operator
                else:
                    FT_basis += FermionOperator((m_sigma, ladder), np.exp(1j * 2 * np.pi * (mx * nx / Nx + my * ny / Ny)) / np.sqrt(n_sites))

            FT_term *= FT_basis
        FT_hamiltonian += FT_term * coeff
        FT_hamiltonian = normal_ordered(FT_hamiltonian)
    FT_hamiltonian.compress()

    return round_operator(FT_hamiltonian)

def inverse_fourier_transform(hamiltonian, Nx, Ny):

    def index2tuple(index, spin=False):
        if spin:
            return ((index // 2) % Nx, (index // 2) // Nx, index % 2)
        else:
            return (index % Nx, index // Nx)
        
    def tuple2index(ix, iy, spin):
        return 2 * (ix + Nx * iy) + spin
        
    n_sites = Nx * Ny

    IFT_hamiltonian = FermionOperator()
    for term, coeff in hamiltonian.terms.items():
        
        IFT_term = FermionOperator.identity()
        for m, ladder in term:
            
            IFT_basis = FermionOperator()
            mx, my, spin = index2tuple(m, spin=True)

            for n in range(n_sites):
                nx, ny = index2tuple(n, spin=False)
                n_sigma = tuple2index(nx, ny, spin)
                # creation operator
                if ladder:
                    IFT_basis += FermionOperator((n_sigma, ladder), np.exp(1j * 2 * np.pi * (mx * nx / Nx + my * ny / Ny)) / np.sqrt(n_sites))
                # annihilation operator
                else:
                    IFT_basis += FermionOperator((n_sigma, ladder), np.exp(-1j * 2 * np.pi * (mx * nx / Nx + my * ny / Ny)) / np.sqrt(n_sites))

            IFT_term *= IFT_basis
        IFT_hamiltonian += IFT_term * coeff
        IFT_hamiltonian = normal_ordered(IFT_hamiltonian)
    IFT_hamiltonian.compress()

    return round_operator(IFT_hamiltonian)