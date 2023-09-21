import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from openfermion import (
    FermionOperator,
    InteractionOperator,
    QubitOperator,
    QuadraticHamiltonian,
    fermi_hubbard,
    number_operator,
    get_interaction_operator,
    get_quadratic_hamiltonian,
)
from openfermion.circuits import (
    slater_determinant_preparation_circuit
)
from openfermion.linalg import (
    get_sparse_operator,
    jw_number_restrict_operator,
    jw_get_ground_state_at_particle_number,
)
from openfermion.transforms import (
    jordan_wigner,
    fourier_transform,
)
from openfermion.utils import (
    Grid,
)
import matplotlib.pyplot as plt
from .utils import (
    QubitOperator_to_qmlHamiltonian,
    PauliStringRotation
)
from functools import reduce

class DHA:
    def __init__(self,
                 pool: list[FermionOperator],
                 n_epoch: int,
                 n_electrons: int,
                 lr: float,
                 threshold1: float,
                 threshold2: float,
                 x_dimension: int,
                 y_dimension: int,
                 tunneling: int,
                 coulomb: float,
                 chemical_potential=0.0,
                 magnetic_field=0.0,
                 periodic=True,
                 spinless=False,
                 particle_hole_symmetry=False
                 ):
        
        self.pool = pool
        self.n_epoch = n_epoch
        self.lr = lr
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.n_qubits = x_dimension * y_dimension * 2
        self.n_electrons = n_electrons
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fermionHamiltonian = fourier_transform(
            hamiltonian=fermi_hubbard(
                x_dimension, y_dimension, tunneling, coulomb,
                chemical_potential, magnetic_field, periodic,
                spinless, particle_hole_symmetry),
            grid=Grid(dimensions=2, length=(x_dimension, y_dimension), scale=1.0),
            spinless=False
        )
        self.interactionHamiltonian = get_interaction_operator(
            self.fermionHamiltonian
        )
        self.quadraticHamiltonian = QuadraticHamiltonian(
            self.interactionHamiltonian.one_body_tensor
        )

    def get_circuit(self, tau=None, appendGates=None):

        if not appendGates:

            for i in range(self.n_qubits):
                qml.RY(self.params['theta'][i], wires=i)
                qml.RZ(self.params['phi'][i], wires=i)
            
            for i, gate in enumerate(self.selectedGates):
                gate(self.params['tau'][i])

            return qml.expval(self.qmlHamiltonian)

        else:
            for i in range(self.n_qubits):
                qml.RY(self.params['theta'][i].detach(), wires=i)
                qml.RZ(self.params['phi'][i].detach(), wires=i)
            
            for i, gate in enumerate(appendGates):
                gate(tau[i])

            return qml.expval(self.qmlHamiltonian)


if __name__ == '__main__':
    vqe = DHA(
        x_dimension=2,
        y_dimension=2,

    )