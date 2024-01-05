import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from openfermion import (
    FermionOperator,
    QubitOperator,
    fermi_hubbard,
    jordan_wigner,
    number_operator,
    get_sparse_operator,
    givens_decomposition_square,
)
from openfermion.utils.indexing import down_index, up_index
import matplotlib.pyplot as plt
from .utils import (
    QubitOperator_to_qmlHamiltonian,
    PauliStringRotation,
    get_hva_commuting_hopping_terms
)
from linalg.exact_diagonalization import jw_get_ground_state_for_3x3
from operators.fourier import fourier_transform_matrix, fourier_transform
from operators.tools import get_quadratic_term, get_interacting_term
from functools import reduce
import os
import pickle

def get_particle_number_operator(x_dimension, y_dimension, spinless=False):

    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    total_particle_operator = FermionOperator()

    for site in range(n_sites):

        if spinless:
            total_particle_operator += number_operator(n_sites, site, 1)
        else:
            total_particle_operator += number_operator(n_spin_orbitals, up_index(site), 1)
            total_particle_operator += number_operator(n_spin_orbitals, down_index(site), 1)

    return total_particle_operator

def get_total_spin(x_dimension, y_dimension, spin_type):
    
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    total_spin = FermionOperator()

    for site in range(n_sites):
        if spin_type == 'spin-up':
            total_spin += number_operator(n_spin_orbitals, up_index(site), 1)
        elif spin_type == 'spin-down':
            total_spin += number_operator(n_spin_orbitals, down_index(site), 1)
        else:
            raise ValueError('spin_type must be either spin-up or spin-down')

    return total_spin

def get_spin_operators(n_sites, spin_type):

    Sx = FermionOperator()
    Sy = FermionOperator()
    Sz = FermionOperator()

    for site in range(n_sites):
        i_up = up_index(site)
        i_down = down_index(site)

        Sx += FermionOperator(f'{i_up}^ {i_down}', 0.5) + FermionOperator(f'{i_down}^ {i_up}', 0.5)
        Sy += FermionOperator(f'{i_up}^ {i_down}', -0.5j) - FermionOperator(f'{i_down}^ {i_up}', -0.5j)
        Sz += FermionOperator(f'{i_up}^ {i_up}', 0.5) - FermionOperator(f'{i_down}^ {i_down}', 0.5)

    if spin_type == 'Sx':
        return Sx
    elif spin_type == 'Sy':
        return Sy
    elif spin_type == 'Sz':
        return Sz
    elif spin_type == 'S^2':
        return Sx * Sx + Sy * Sy + Sz * Sz

def Trotterize_generator(theta, generator: QubitOperator):

    if isinstance(generator, FermionOperator):
        generator = jordan_wigner(generator)
    
    for pauliString, coeff in generator.terms.items():
        if not pauliString:
            continue
        String = reduce(lambda a, b: a+b, [pauliString[i][1] for i in range(len(pauliString))])
        Indicies = [pauliString[i][0] for i in range(len(pauliString))]
        Pk = (String, Indicies)
        PauliStringRotation(2 * theta * coeff.real, Pk)

def get_non_interacting_ground_state_index(quadratic_hamiltonian: FermionOperator, n_qubits, n_spin_up, n_spin_down):

    spin_up_energies = {x: 0 for x in range(0, n_qubits, 2)}
    spin_down_energies = {x: 0 for x in range(1, n_qubits, 2)}
    
    for term, coeff in quadratic_hamiltonian.terms.items():
        index = term[0][0]
        if index % 2 == 0:
            spin_up_energies[index] = coeff
        else:
            spin_down_energies[index] = coeff

    spin_up_indices = sorted(spin_up_energies, key=spin_up_energies.get)[:n_spin_up]
    spin_down_indices = sorted(spin_down_energies, key=spin_down_energies.get)[:n_spin_down]

    print('spin up orbital energies:', spin_up_energies)
    print('spin down orbital energies: ', spin_down_energies)

    return spin_up_indices, spin_down_indices

class HVA:
    def __init__(self,
                 n_epoch: int,
                 reps: int,
                 lr: float,
                 threshold: float,
                 x_dimension: int,
                 y_dimension: int,
                 n_electrons: int,
                 n_spin_up: int,
                 n_spin_down: int,
                 tunneling: float,
                 coulomb: float,
                 periodic=True,
                 spinless=False,
                 particle_hole_symmetry=False,
                 load_model=False
                 ):
        
        self.n_epoch = n_epoch
        self.reps = reps
        self.lr = lr
        self.threshold = threshold
        self.n_sites = x_dimension * y_dimension
        self.n_qubits = x_dimension * y_dimension * 2
        self.n_electrons = n_electrons
        self.n_spin_up = n_spin_up
        self.n_spin_down = n_spin_down
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fermionHamiltonian = fermi_hubbard(x_dimension,
                                                y_dimension,
                                                tunneling,
                                                coulomb,
                                                periodic=periodic,
                                                spinless=spinless,
                                                particle_hole_symmetry=particle_hole_symmetry)
        self.qubitHamiltonian = jordan_wigner(self.fermionHamiltonian)
        self.quadratic_term = get_quadratic_term(self.fermionHamiltonian)
        self.interacting_term = get_interacting_term(self.fermionHamiltonian)
        self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(self.fermionHamiltonian)
        self.fermionOperators = {
            'hopping': get_quadratic_term(self.fermionHamiltonian),
            'coulomb': get_interacting_term(self.fermionHamiltonian),
            'particle number': get_particle_number_operator(x_dimension, y_dimension, spinless),
            'spin up': get_total_spin(x_dimension, y_dimension, 'spin-up'),
            'spin down': get_total_spin(x_dimension, y_dimension, 'spin-down'),
            'Sx': get_spin_operators(self.n_sites, spin_type='Sx'),
            'Sy': get_spin_operators(self.n_sites, spin_type='Sy'),
            'Sz': get_spin_operators(self.n_sites, spin_type='Sz'),
            'S^2': get_spin_operators(self.n_sites, spin_type='S^2')
        }
        _h, _v = get_hva_commuting_hopping_terms(x_dimension, y_dimension, periodic)
        self.Nh = len(_h)
        self.Nv = len(_v)
        self.hvaGenerators = {
            'horizontal': [jordan_wigner(generator) for generator in _h],
            'vertical': [jordan_wigner(generator) for generator in _v],
            'coulomb': jordan_wigner(self.fermionOperators['coulomb'])
        }
        self.qmlOperators = {
            'spin up': QubitOperator_to_qmlHamiltonian(self.fermionOperators['spin up']),
            'spin down': QubitOperator_to_qmlHamiltonian(self.fermionOperators['spin down']),
            'Sx': QubitOperator_to_qmlHamiltonian(self.fermionOperators['Sx']),
            'Sy': QubitOperator_to_qmlHamiltonian(self.fermionOperators['Sy']),
            'Sz': QubitOperator_to_qmlHamiltonian(self.fermionOperators['Sz']),
            'S^2': QubitOperator_to_qmlHamiltonian(self.fermionOperators['S^2'])
        }
        self.FT_transformation_matrix = fourier_transform_matrix(x_dimension, y_dimension)
        self.decomposition, self.diagonal = givens_decomposition_square(self.FT_transformation_matrix)
        self.circuit_description = list(reversed(self.decomposition))
        self.k_quadratic_term = fourier_transform(self.quadratic_term, x_dimension, y_dimension)
        self.spin_up_indices, self.spin_down_indices = get_non_interacting_ground_state_index(
                                                            self.k_quadratic_term,
                                                            self.n_qubits,
                                                            n_spin_up, n_spin_down
                                                       )
        print('spin up indices: ', self.spin_up_indices, '  ', 'spin down indices: ', self.spin_down_indices, '\n')
        

        self.img_filepath = f'./images/HVA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}, reps={reps}).png'
        self.wf_filepath = f'./results/ground_state_results/Hubbard-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}).pkl'
        self.result_filepath = f'./results/vqe_results/HVA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}, reps={reps}).pkl'
        self.model_filepath = f'./results/saved_model/HVA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}, reps={reps}).pkl'
        self.ground_state_energy, self.ground_state_wfs = self.get_ground_state()
        
        if load_model:
            self.load_model()
        else:
            self.params = nn.ParameterDict({
                'theta_U': nn.Parameter(torch.zeros(reps+1), requires_grad=True),
                'theta_v': nn.Parameter(torch.zeros(reps * self.Nv), requires_grad=True),
                'theta_h': nn.Parameter(torch.zeros(reps * self.Nh), requires_grad=True),
            }).to(self.device)

            self.results = {
                'loss': [],
                'Sz': [],
                'S^2': [],
                'fidelity': [],
            }

    def get_ground_state(self):
        
        # Check if the file exists. 
        # If the file exists, load the ground state energy and wavefunction
        if os.path.exists(self.wf_filepath):
            with open(self.wf_filepath, 'rb') as file:
                loaded_state_dict = pickle.load(file)
                ground_state_energy = loaded_state_dict['energy']
                ground_state_wfs = loaded_state_dict['wave function']
        
        # If the file does not exist, calculate the ground state energy and wavefunction, 
        # and then save them as a file
        else:
            ground_state_energy, ground_state_wfs = jw_get_ground_state_for_3x3(
                                                        sparse_operator=get_sparse_operator(self.fermionHamiltonian),
                                                        particle_number=self.n_electrons,
                                                        spin_up=self.n_spin_up,
                                                        spin_down=self.n_spin_down
                                                    )
            state_dict = {
                'energy': ground_state_energy,
                'wave function': ground_state_wfs
            }
            with open(self.wf_filepath, 'wb') as file:
                pickle.dump(state_dict, file)

        return ground_state_energy, ground_state_wfs
    
    def save_model(self):
        
        state_dict = {
            'params': self.params,
        }

        with open(self.model_filepath, 'wb') as file:
            pickle.dump(state_dict, file)

        with open(self.result_filepath , 'wb') as file:
            pickle.dump(self.results, file)

    def load_model(self):
        
        if not os.path.exists(self.model_filepath):
            raise ValueError('Please check if the file ' + self.model_filepath + 'exists!')
        if not os.path.exists(self.result_filepath):
            raise ValueError('Please check if the file ' + self.result_filepath + 'exists!')
        
        with open(self.model_filepath, 'rb') as file:
            state_dict = pickle.load(file)
            self.params = state_dict['params'].to(self.device)
        
        with open(self.result_filepath, 'rb') as file:
            self.results = pickle.load(file)        
    
    def circuit(self, theta_U, theta_h, theta_v, mode='train'):
        
        # prepare non-interacting ground state
        for q in self.spin_up_indices + self.spin_down_indices:
            qml.PauliX(wires=q)

        for i in range(self.n_qubits):
            qml.RZ(np.angle(self.diagonal[i]), wires=i)
            
        for parallel_ops in self.circuit_description:
            for op in parallel_ops:
                if op == 'pht':
                    qml.PauliX(wires=self.n_qubits-1)
                else:
                    i, j, theta, phi = op
                    qml.SingleExcitation(2 * theta, wires=[i, j])
                    qml.RZ(phi, wires=j)
        
        # ansatz circuit
        for rep in range(self.reps):
            Trotterize_generator(theta_U[rep], self.hvaGenerators['coulomb'])
            for i in range(self.Nv):
                Trotterize_generator(theta_v[rep * self.Nv + i], self.hvaGenerators['vertical'][i])
            for i in range(self.Nh):
                Trotterize_generator(theta_h[rep * self.Nh + i], self.hvaGenerators['horizontal'][i])
        Trotterize_generator(theta_U[self.reps], self.hvaGenerators['coulomb'])

        if mode == 'train':
            return qml.expval(self.qmlHamiltonian), qml.expval(self.qmlOperators['Sz']), qml.expval(self.qmlOperators['S^2'])
        elif mode == 'state':
            return qml.state()
        
    def calculate_fidelity(self, ground_state_wfs, state):

        projected_state = np.zeros_like(state)
        for ground_state_wf in ground_state_wfs:
            coeff = ground_state_wf.conj() @ state
            projected_state += coeff * ground_state_wf
        projected_state = projected_state / np.linalg.norm(projected_state, ord=2)
        return np.abs(state.conj() @ projected_state) ** 2

    def run(self):
        
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        dev = qml.device('default.qubit.torch', wires=self.n_qubits)
        opt = optim.Adam(params=self.params.values(), lr=self.lr)
        circuit = self.circuit
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')

        i_epoch = len(self.results['loss'])
        
        while i_epoch < self.n_epoch:
            
            state = model(self.params['theta_U'], self.params['theta_h'], self.params['theta_v'], mode='state')
            state = state.detach().cpu().numpy()
            fidelity = self.calculate_fidelity(self.ground_state_wfs, state)

            opt.zero_grad()
            loss, Sz, S_square = model(self.params['theta_U'], self.params['theta_h'], self.params['theta_v'], mode='train')
            loss.backward()
            opt.step()

            self.results['loss'].append(loss.item())
            self.results['Sz'].append(Sz.item())
            self.results['S^2'].append(S_square.item())
            self.results['fidelity'].append(fidelity.item())

            grad_vector = torch.cat((self.params['theta_U'].grad, self.params['theta_h'].grad, self.params['theta_v'].grad))
            grad_norm = torch.linalg.vector_norm(grad_vector).item()
            print(f"iter: {len(self.results['loss'])} | loss: {loss.item(): 6f} | norm: {grad_norm: 6f} | fidelity: {fidelity.item(): 6f} | Sz: {Sz.item(): 6f} | S^2: {S_square.item(): 6f}")
            
            ax1.clear()
            ax1.plot(np.arange(i_epoch+1)+1, self.results['loss'], marker='X', color='r', label='HVA')
            ax1.plot(np.arange(i_epoch+1)+1, np.full(i_epoch+1, self.ground_state_energy), ls='-', color='g', label='ED')
            ax1.set_xlabel('epochs')
            ax1.set_ylabel('energy')
            ax1.legend()
            ax1.grid()

            ax2.clear()
            ax2.plot(np.arange(i_epoch+1)+1, self.results['fidelity'], marker='X', ls=':', color='coral')
            ax2.set_xlabel('epochs')
            ax2.set_ylabel('fidelity')
            ax2.grid()

            plt.savefig(self.img_filepath)

            # if grad_norm < self.threshold:
            #     break

            if (i_epoch+1) % 10 == 0:
                self.save_model()

            i_epoch += 1

        self.save_model()
    
if __name__ == '__main__':
    vqe = HVA(
        n_epoch=800,
        reps=10,
        lr=1e-2,
        threshold=1e-2,
        x_dimension=3,
        y_dimension=3,
        n_electrons=9,
        n_spin_up=5,
        n_spin_down=4,
        tunneling=1,
        coulomb=6,
        periodic=True,
        spinless=False,
        particle_hole_symmetry=False,
        load_model=False
    )
    
    vqe.run()

    
