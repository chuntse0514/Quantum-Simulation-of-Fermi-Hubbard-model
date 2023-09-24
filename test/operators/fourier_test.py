from operators.fourier import (
    fourier_transform_matrix,
    fourier_transform
)
from operators.tools import (
    get_quadratic_term,
    get_interacting_term
)
from openfermion import fermi_hubbard

def test_fourier_transform(Nx, Ny):

    hamiltonian = fermi_hubbard(
        Nx, Ny, tunneling=1, coulomb=4
    )
    FT_hamiltonian = fourier_transform(
        hamiltonian, Nx, Ny
    )
    print('quadratic term:\n', get_quadratic_term(FT_hamiltonian))
    print('')
    print('interacting term:\n', get_interacting_term(FT_hamiltonian))

if __name__ == '__main__':
    test_fourier_transform(2, 2)