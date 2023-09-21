import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from openfermion import (
    FermionOperator,
    QubitOperator,
    number_operator,
    jordan_wigner,
    get_sparse_operator,
    get_quadratic_hamiltonian,
    jw_get_ground_state_at_particle_number,
)
from openfermion.hamiltonians.hubbard import (
    _hopping_term, 
    _coulomb_interaction_term,
    _right_neighbor,
    _bottom_neighbor
)
from openfermion.circuits import slater_determinant_preparation_circuit
from openfermion.utils.indexing import down_index, up_index
import matplotlib.pyplot as plt
from .utils import (
    QubitOperator_to_qmlHamiltonian,
    PauliStringRotation
)
from functools import reduce

def get_horizontal_hopping(x_dimension, y_dimension, periodic=True, spinless=False):
    
    # Initialize operator.
    n_sites = x_dimension * y_dimension
    horizontal_hopping = FermionOperator()

    for site in range(n_sites):

        # Get indices of right neighbors
        right_neighbor = _right_neighbor(site, x_dimension, y_dimension, periodic)

        # Avoid double-counting edges when one of the dimensions is 2
        # and the system is periodic
        if x_dimension == 2 and periodic and site % 2 == 1:
            right_neighbor = None

        # Add hopping terms with neighbors to the right.
        if right_neighbor is not None:

            if spinless:
                horizontal_hopping += _hopping_term(site, right_neighbor, 1)
            else:
                horizontal_hopping += _hopping_term(up_index(site),
                                                    up_index(right_neighbor), 1)
                horizontal_hopping += _hopping_term(down_index(site),
                                                    down_index(right_neighbor), 1)

    return horizontal_hopping
    
def get_vertical_hopping(x_dimension, y_dimension, periodic=True, spinless=False):

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    vertical_hopping = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):

        # Get indices of bottom neighbors
        bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension, periodic)

        # Avoid double-counting edges when one of the dimensions is 2
        # and the system is periodic
        if y_dimension == 2 and periodic and site >= x_dimension:
            bottom_neighbor = None

        # Add hopping terms with neighbors to the bottom.
        if bottom_neighbor is not None:

            if spinless:
                vertical_hopping += _hopping_term(site, bottom_neighbor, 1)
            else:
                vertical_hopping += _hopping_term(up_index(site),
                                                  up_index(bottom_neighbor), 1)
                vertical_hopping += _hopping_term(down_index(site),
                                                  down_index(bottom_neighbor), 1)

    return vertical_hopping

def get_coulomb_interaction(x_dimension, y_dimension, periodic=True, spinless=False, particle_hole_symmetry=False):
    
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    coulomb_interaction = FermionOperator()
    
    if spinless:
        for site in range(n_sites):

            right_neighbor = _right_neighbor(site, x_dimension, y_dimension, periodic)
            bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension, periodic)

            if x_dimension == 2 and periodic and site % 2 == 1:
                right_neighbor = None
            if y_dimension == 2 and periodic and site >= x_dimension:
                bottom_neighbor = None

            if right_neighbor is not None:
                coulomb_interaction += _coulomb_interaction_term(n_sites, site,
                                                                 right_neighbor, 1,
                                                                 particle_hole_symmetry)
            if bottom_neighbor is not None:
                coulomb_interaction += _coulomb_interaction_term(n_sites, site,
                                                                 bottom_neighbor, 1,
                                                                 particle_hole_symmetry)
    
    else:
        for site in range(n_sites):
            coulomb_interaction += _coulomb_interaction_term(n_spin_orbitals,
                                                             up_index(site),
                                                             down_index(site), 1,
                                                             particle_hole_symmetry)
    
    return coulomb_interaction

def get_chemical_potential(x_dimension, y_dimension, chemical_potential=0.0, 
                           magnetic_field=0.0, spinless=False):
    
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    chemical_potential_term = FermionOperator()

    for site in range(n_sites):
        
        if spinless:
            chemical_potential_term += number_operator(n_sites, site, -chemical_potential)
        else:
            chemical_potential_term += number_operator(n_spin_orbitals, up_index(site),
                                                       -chemical_potential - magnetic_field)
            chemical_potential_term += number_operator(n_spin_orbitals, down_index(site),
                                                       -chemical_potential + magnetic_field)
    
    return chemical_potential_term


class HVA:
    def __init__(self,
                 n_epoch: int,
                 lr: float,
                 threshold: float,
                 reps: int,
                 init_strategy: str,
                 x_dimension: int,
                 y_dimension: int,
                 tunneling: float,
                 coulomb: float,
                 chemical_potential=0.0,
                 magnetic_field=0.0,
                 periodic=True,
                 spinless=False,
                 particle_hole_symmetry=False
                 ):
        
        self.n_epoch = n_epoch
        self.lr = lr
        self.threshold = threshold
        self.reps = reps
        self.n_qubits = x_dimension * y_dimension * 2
        self.n_electrons = self.n_qubits // 2
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        _horizontal_hopping = get_horizontal_hopping(
            x_dimension, y_dimension, periodic, spinless
        )
        _vertical_hopping = get_vertical_hopping(
            x_dimension, y_dimension, periodic, spinless
        )
        _coulomb_interaction = get_coulomb_interaction(
            x_dimension, y_dimension, periodic, spinless, particle_hole_symmetry
        )
        _chemical_potential = get_chemical_potential(
            x_dimension, y_dimension, chemical_potential, magnetic_field, spinless
        )
        self.fermionHamiltonian = -tunneling * _horizontal_hopping + \
                                  -tunneling * _vertical_hopping + \
                                  coulomb * _coulomb_interaction
        self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(
            jordan_wigner(self.fermionHamiltonian)
        )
        self.nonInteractingHamiltonian = get_quadratic_hamiltonian(
            -tunneling * _horizontal_hopping + \
            -tunneling * _vertical_hopping
        )
        
        self.horizontal_hopping = jordan_wigner(_horizontal_hopping)
        self.vertical_hopping = jordan_wigner(_vertical_hopping)
        self.coulomb_interaction = jordan_wigner(_coulomb_interaction)
        self.chemical_potential = jordan_wigner(_chemical_potential)
        self.horizontal_hopping.compress()
        self.vertical_hopping.compress()
        self.coulomb_interaction.compress()
        self.chemical_potential.compress()
        
        if init_strategy == 'Identity':
            self.params = nn.ParameterDict({
                'theta_U': nn.Parameter(torch.zeros(reps), requires_grad=True),
                'theta_v': nn.Parameter(torch.zeros(reps), requires_grad=True),
                'theta_h': nn.Parameter(torch.zeros(reps), requires_grad=True),
            }).to(self.device)
        elif init_strategy == 'Random-uniform':
            self.params = nn.ParameterDict({
                'theta_U': nn.Parameter((torch.rand(reps) * 2 - 1) * np.pi, requires_grad=True),
                'theta_v': nn.Parameter((torch.rand(reps) * 2 - 1) * np.pi, requires_grad=True),
                'theta_h': nn.Parameter((torch.rand(reps) * 2 - 1) * np.pi, requires_grad=True),
            }).to(self.device)

        # self.ground_state_energy, self.ground_state_wf = jw_get_ground_state_at_particle_number(
        #     sparse_operator=get_sparse_operator(self.fermionHamiltonian),
        #     particle_number=self.n_electrons
        # )
        # self.ground_state_wf = torch.complex(
        #     torch.Tensor(self.ground_state_wf.real).double(), torch.Tensor(self.ground_state_wf.imag).double()
        # ).to(self.device)
        self.loss_history = []
        self.filename = f'./images/HVA-{x_dimension}x{y_dimension},t={tunneling}, U={coulomb}, layers={reps}, init={init_strategy}.png'

    def Trotterize_operator(self, theta, operator: QubitOperator):

        for pauliString, _ in operator.terms.items():
    
            if not pauliString:
                continue

            String = reduce(lambda a, b: a+b, [pauliString[i][1] for i in range(len(pauliString))])
            Indicies = [pauliString[i][0] for i in range(len(pauliString))]
            Pk = (String, Indicies)
            PauliStringRotation(theta, Pk)
    
    def prepare_nonInteracting_groundState(self):
        
        initially_occupied_orbitals = list(range(self.n_electrons))
        _, transformation_matrix, _ = self.nonInteractingHamiltonian.diagonalizing_bogoliubov_transform()
        transformation_matrix = transformation_matrix[initially_occupied_orbitals]
        circuit_description = slater_determinant_preparation_circuit(transformation_matrix)

        for q in initially_occupied_orbitals:
            qml.PauliX(wires=q)
        
        for parallel_ops in circuit_description:
            for op in parallel_ops:
                if op == 'pht':
                    qml.PauliX(wires=self.n_qubits-1)
                else:
                    i, j, theta, phi = op
                    qml.SingleExcitation(2 * theta, wires=[i, j])
                    qml.RZ(phi, wires=j)

    def circuit(self, theta_U, theta_v, theta_h):
    
        self.prepare_nonInteracting_groundState()

        self.Trotterize_operator(theta_U[0] / 2, self.coulomb_interaction)
        for rep in range(self.reps):
            self.Trotterize_operator(theta_v[rep], self.vertical_hopping)
            self.Trotterize_operator(theta_h[rep], self.horizontal_hopping)
            if rep != self.reps-1:
                self.Trotterize_operator((theta_U[rep] + theta_U[rep+1]) / 2,
                                         self.coulomb_interaction)
            else:
                self.Trotterize_operator(theta_U[rep], self.coulomb_interaction)

        return qml.expval(self.qmlHamiltonian)

    def state(self):
        
        self.prepare_nonInteracting_groundState()
        self.Trotterize_operator(self.params['theta_U'][0] / 2, self.coulomb_interaction)
        for rep in range(self.reps):
            self.Trotterize_operator(self.params['theta_v'][rep], self.vertical_hopping)
            self.Trotterize_operator(self.params['theta_h'][rep], self.horizontal_hopping)
            if rep != self.reps-1:
                self.Trotterize_operator((self.params['theta_U'][rep] + self.params['theta_U'][rep+1]) / 2,
                                         self.coulomb_interaction)
            else:
                self.Trotterize_operator(self.params['theta_U'][rep], self.coulomb_interaction)

        return qml.state()


    def run(self):
        
        plt.ion()
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        dev = qml.device('default.qubit.torch', wires=self.n_qubits)
        opt = optim.NAdam(params=self.params.values(), lr=self.lr)
        circuit = self.circuit
        circuit2 = self.state
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')
        model2 = qml.QNode(circuit2, dev, interface='torch', diff_method='backprop')
        
        nums_frequency = {
            'theta_U': {(i,): len(self.coulomb_interaction.terms)-1 for i in range(self.reps)},
            'theta_v': {(i,): len(self.vertical_hopping.terms) for i in range(self.reps)},
            'theta_h': {(i,): len(self.horizontal_hopping.terms) for i in range(self.reps)}
        }

        print(qml.draw(model)(*(self.params['theta_U'], self.params['theta_v'], self.params['theta_h'])))

        return 

        for i_epoch in range(self.n_epoch):
            
            opt.zero_grad()
            loss = model(self.params['theta_U'], self.params['theta_v'], self.params['theta_h'])
            state = model2()
            loss.backward()
            opt.step()

            self.loss_history.append(loss.item())

            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1}, energy: {loss.item()}, fidelity: {(torch.inner(state.conj(), self.ground_state_wf).abs() ** 2).item()}')
                print(self.params['theta_U'])
                print(self.params['theta_v'])
                print(self.params['theta_h'])
            
            ax.clear()
            ax.plot(np.arange(i_epoch+1)+1, self.loss_history, marker='X', color='r', label='HVA')
            ax.plot(np.arange(i_epoch+1)+1, np.full(i_epoch+1, self.ground_state_energy), ls='-', color='g', label='ED')
            ax.set_xlabel('epochs')
            ax.set_ylabel('energy')
            ax.legend()
            ax.grid()

            plt.pause(0.01)
            plt.savefig(self.filename)

        plt.ioff()
        plt.show()
    
if __name__ == '__main__':
    vqe = HVA(
        n_epoch=200,
        lr=1e-3,
        threshold=1e-3,
        reps=1,
        init_strategy='Identity',
        x_dimension=2,
        y_dimension=2,
        tunneling=1,
        coulomb=0.1,
    )

    vqe.run()
