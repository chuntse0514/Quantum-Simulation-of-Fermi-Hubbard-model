import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csc_matrix
from openfermion import (
    QubitOperator,
    FermionOperator,
    fermi_hubbard,
    jordan_wigner,
    get_sparse_operator,
    givens_decomposition_square,
    jw_get_ground_state_at_particle_number,
    number_operator,
    up_index, down_index
)
import matplotlib.pyplot as plt
from .utils import (
    QubitOperator_to_qmlHamiltonian,
    PauliStringRotation
)
from linalg.exact_diagonalization import jw_get_ground_state
from operators.fourier import fourier_transform_matrix, fourier_transform
from operators.tools import get_interacting_term, get_quadratic_term
from operators.pool import hubbard_interaction_pool_simplified
from functools import reduce, partial
import gc
import os
import pickle
import time

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

def get_total_spin(n_sites, spin_type):
    
    total_spin = FermionOperator()
    n_spin_orbitals = n_sites * 2

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

def print_list(op_list):
    for op in op_list:
        print(str(op).replace('\n', ' '))

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

class DHA:
    def __init__(self,
                 n_epoch: int,
                 threshold1: float,
                 threshold2: float,
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
        
        self.fermionOperatorPool = hubbard_interaction_pool_simplified(x_dimension, y_dimension)
        self.qubitOperatorPool = [jordan_wigner(generator) for generator in self.fermionOperatorPool]
        self.gateOperatorPool = [partial(Trotterize_generator, generator=generator) for generator in self.qubitOperatorPool]

        self.n_epoch = n_epoch
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.n_sites = x_dimension * y_dimension
        self.n_qubits = x_dimension * y_dimension * 2
        self.n_electrons = n_electrons
        self.n_spin_up = n_spin_up
        self.n_spin_down = n_spin_down
        
        self.ratio = 0.1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

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
            'spin up': get_total_spin(self.n_sites, spin_type='spin-up'),
            'spin down': get_total_spin(self.n_sites, spin_type='spin-down'),
            'Sx': get_spin_operators(self.n_sites, spin_type='Sx'),
            'Sy': get_spin_operators(self.n_sites, spin_type='Sy'),
            'Sz': get_spin_operators(self.n_sites, spin_type='Sz'),
            'S^2': get_spin_operators(self.n_sites, spin_type='S^2')
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
        self.k_interacting_term = fourier_transform(self.interacting_term, x_dimension, y_dimension)
        self.spin_up_indices, self.spin_down_indices = get_non_interacting_ground_state_index(
                                                            self.k_quadratic_term,
                                                            self.n_qubits,
                                                            n_spin_up, n_spin_down
                                                       )
        print('spin up indices: ', self.spin_up_indices, '  ', 'spin down indices: ', self.spin_down_indices, '\n')
        self.img_filepath = f'./images/DHA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}).png'
        self.wf_filepath = f'./results/ground_state_results/Hubbard-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}).pkl'
        self.result_filepath = f'./results/vqe_results/DHA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}).pkl'
        self.model_filepath = f'./results/saved_model/DHA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}).pkl'
        self.ground_state_energy, self.ground_state_wf = self.get_ground_state()
        
        if load_model:
            self.load_model()
        else:
            self.params = nn.ParameterDict({
                'e': nn.Parameter(torch.zeros(len(self.gateOperatorPool)), requires_grad=True),
                't': nn.Parameter(torch.Tensor([]), requires_grad=True)
            }).to(self.device)
            self.selected_gates = []
            self.results = {
                'epoch loss': [],
                'iteration loss': [],
                'Sz': [],
                'S^2': [],
                'fidelity': [],
                'n_params': [],
                'selected operators': []
            }

    def get_ground_state(self):
        
        # Check if the file exists. 
        # If the file exists, load the ground state energy and wavefunction
        if os.path.exists(self.wf_filepath):
            with open(self.wf_filepath, 'rb') as file:
                loaded_state_dict = pickle.load(file)
                ground_state_energy = loaded_state_dict['energy']
                ground_state_wf = loaded_state_dict['wave function']
        
        # If the file does not exist, calculate the ground state energy and wavefunction, 
        # and then save them as a file
        else:
            ground_state_energy, ground_state_wf = jw_get_ground_state(
                                                        sparse_operator=get_sparse_operator(self.fermionHamiltonian),
                                                        particle_number=self.n_electrons,
                                                        spin_up=self.n_spin_up,
                                                        spin_down=self.n_spin_down
                                                    )
            state_dict = {
                'energy': ground_state_energy,
                'wave function': ground_state_wf
            }
            with open(self.wf_filepath, 'wb') as file:
                pickle.dump(state_dict, file)

        return ground_state_energy, ground_state_wf

    def get_ground_state_properties(self):
        
        # up = get_sparse_operator(self.fermionOperators['spin up'], n_qubits=self.n_qubits)
        # down = get_sparse_operator(self.fermionOperators['spin down'], n_qubits=self.n_qubits)
        # Sz = get_sparse_operator(self.fermionOperators['Sz'], n_qubits=self.n_qubits)
        # S_square = get_sparse_operator(self.fermionOperators['S^2'], n_qubits=self.n_qubits)

        # up_spin_value = (self.ground_state_wf.conj() @ up @ self.ground_state_wf).real
        # down_spin_value = (self.ground_state_wf.conj() @ down @ self.ground_state_wf).real
        # Sz_value = (self.ground_state_wf.conj() @ Sz @ self.ground_state_wf).real
        # S_square_value = (self.ground_state_wf.conj() @ S_square @ self.ground_state_wf).real
        
        print('ground state energy: ', self.ground_state_energy)
        print('particle number: ', self.n_electrons)
        # print('spin up:', np.round(up_spin_value, decimals=6))
        # print('spin down:', np.round(down_spin_value, decimals=6))
        # print('Sz: ', np.round(Sz_value, decimals=6))
        # print('S^2: ', np.round(S_square_value, decimals=6))
        print('')

    def save_model(self):
        
        state_dict = {
            'params': self.params,
            'circuit': self.selected_gates
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
            self.selected_gates = state_dict['circuit']
        
        with open(self.result_filepath, 'rb') as file:
            self.results = pickle.load(file)
    
    def select_operator(self):
        
        if self.n_qubits < 20:
            dev = qml.device('default.qubit.torch', wires=self.n_qubits)
            diff_method = 'backprop'
        else:
            dev = qml.device('lightning.gpu', wires=self.n_qubits)
            diff_method = 'adjoint'
        model = qml.QNode(self.circuit, dev, interface='torch', diff_method=diff_method)
        loss = model(mode='eval')
        loss.backward()
        grads = self.params['e'].grad.cpu().numpy()
        grads = np.abs(grads)
        self.params['e'].grad.zero_()
        
        max_grad = np.max(grads)
        self.Ng = np.sum((grads >= max_grad * self.ratio) * (grads >= self.threshold1))
        selected_indices = np.argsort(grads)[::-1][:self.Ng].tolist()
        selected_operator = [self.fermionOperatorPool[index] for index in selected_indices]
        selected_gates = [self.gateOperatorPool[index] for index in selected_indices]
        max_grads = [grads[index] for index in selected_indices]

        del model
        torch.cuda.empty_cache()
        gc.collect()

        return selected_operator, selected_gates, max_grads

    def circuit(self, mode='train'):
        
        # prepare non-interacting ground state in k-space
        for q in self.spin_up_indices + self.spin_down_indices:
            qml.PauliX(wires=q)
        
        # circuit selected by algorithm
        if mode == 'train' or mode == 'state':
            for i, gate in enumerate(self.selected_gates):
                gate(self.params['t'][i])
            
        elif mode == 'eval':
            for i, gate in enumerate(self.selected_gates):
                gate(self.params['t'][i])

            for i, gate in enumerate(self.gateOperatorPool):
                gate(self.params['e'][i])

        # apply the Fourier Transform back to the real space
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
        
        if mode == 'train':
            return qml.expval(self.qmlHamiltonian), qml.expval(self.qmlOperators['Sz']), qml.expval(self.qmlOperators['S^2'])
        elif mode == 'state':
            return qml.state()
        elif mode == 'eval':
            return qml.expval(self.qmlHamiltonian)

    def run(self):
        
        self.get_ground_state_properties()

        start_time = time.time()
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        if self.n_qubits < 20:
            dev = qml.device('default.qubit.torch', wires=self.n_qubits)
            diff_method = 'backprop'
        else:
            dev = qml.device('lightning.gpu', wires=self.n_qubits)
            diff_method = 'adjoint'

        i_epoch = len(self.results['epoch loss'])

        while i_epoch < self.n_epoch:

            selected_operators, selected_gates, max_grads = self.select_operator()
            if len(max_grads) == 0:
                print('\nconvergence criterion has satisfied, break the loop!')
                break
            
            self.selected_gates += selected_gates
            self.params['t'] = torch.cat((self.params['t'], nn.Parameter(torch.zeros(self.Ng), requires_grad=True).to(self.device)))
            self.results['selected operators'] += selected_operators
            self.results['n_params'].append(len(self.results['selected operators']))
            lr = torch.linalg.vector_norm(torch.Tensor(max_grads)).item() / np.sqrt(self.Ng) * 0.05
            opt = optim.Adam(params=self.params.values(), lr=lr)
            print('learning rate = ', lr)

            print('Find operators')
            print_list(selected_operators)
            print('with max gradients')
            print(max_grads)
            print('')
            
            while True:
                
                model = qml.QNode(self.circuit, dev, interface='torch', diff_method=None)
                state = model(mode='state')
                if self.n_qubits < 20:
                    state = state.detach().cpu().numpy()
                fidelity = np.abs(state.conj() @ self.ground_state_wf) ** 2
                
                # Delete the unused state, release the memory of GPU device
                del state, model
                torch.cuda.empty_cache()
                gc.collect()

                model = qml.QNode(self.circuit, dev, interface='torch', diff_method=diff_method)
                opt.zero_grad()
                loss, Sz, S_square = model(mode='train')
                loss.backward()
                opt.step()

                self.results['iteration loss'].append(loss.item())
                self.results['Sz'].append(Sz.item())
                self.results['S^2'].append(S_square.item())
                self.results['fidelity'].append(fidelity.item())

                grad_norm = torch.linalg.vector_norm(self.params['t'].grad).item()

                print(f"iter: {len(self.results['iteration loss'])} | loss: {loss.item(): 6f} | norm: {grad_norm: 6f} | fidelity: {fidelity.item(): 6f} | Sz: {Sz.item(): 6f} | S^2: {S_square.item(): 6f}")

                del loss, Sz, S_square, model
                torch.cuda.empty_cache()
                gc.collect()

                if grad_norm < self.threshold2:
                    break
            
            self.results['epoch loss'].append(self.results['iteration loss'][-1])
            i_epoch += 1
            print('')

            #############################
            
            self.save_model()

            ax1.clear()
            length = len(self.results['iteration loss'])
            ax1.plot(np.arange(length)+1, self.results['iteration loss'], color='coral', marker='X', ls='--', label='DHA')
            ax1.plot(np.arange(length)+1, np.full(length, self.ground_state_energy), color='violet', label='ED')
            ax1.set_xlabel('iteration')
            ax1.set_ylabel('energy')
            ax1.legend()
            ax1.grid()
            
            ax2.clear()
            length = len(self.results['epoch loss'])
            ax2.plot(np.arange(length)+1, self.results['epoch loss'], color='yellowgreen', marker='X', ls='--', label='DHA')
            ax2.plot(np.arange(length)+1, np.full(length, self.ground_state_energy), color='violet', label='ED')
            ax2.set_xlabel('epoch')
            ax2.set_ylabel('energy')
            ax2.legend()
            ax2.grid()

            plt.savefig(self.img_filepath)
        
        end_time = time.time()

        print('total run time: ', end_time - start_time)
        

if __name__ == '__main__':
    vqe = DHA(
        n_epoch=100,
        threshold1=1e-2,
        threshold2=1e-2,
        x_dimension=2,
        y_dimension=4,
        n_electrons=8,
        n_spin_up=4,
        n_spin_down=4,
        tunneling=1,
        coulomb=2,
        load_model=False
    )
    # vqe.get_ground_state_properties()
    vqe.run()