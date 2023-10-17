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
)
from openfermion.circuits import slater_determinant_preparation_circuit
from openfermion.utils.indexing import down_index, up_index
import matplotlib.pyplot as plt
from .utils import (
    QubitOperator_to_qmlHamiltonian,
    PauliStringRotation,
    get_hva_commuting_hopping_terms
)
from linalg.exact_diagonalization import jw_get_ground_state
from operators.fourier import fourier_transform_matrix
from operators.tools import get_quadratic_term, get_interacting_term
from functools import reduce

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

class HVA:
    def __init__(self,
                 n_epoch: int,
                 lr: float,
                 threshold: float,
                 reps: int,
                 x_dimension: int,
                 y_dimension: int,
                 tunneling: float,
                 coulomb: float,
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fermionHamiltonian = fermi_hubbard(
                                    x_dimension, y_dimension, tunneling, coulomb,
                                    periodic=periodic, spinless=spinless, 
                                    particle_hole_symmetry=particle_hole_symmetry
                                  )
        self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(self.fermionHamiltonian)
        self.fermionOperators = {
            'hopping': get_quadratic_term(self.fermionHamiltonian),
            'coulomb': get_interacting_term(self.fermionHamiltonian),
            'particle number': get_particle_number_operator(x_dimension, y_dimension, spinless),
            'up spin': get_total_spin(x_dimension, y_dimension, 'spin-up'),
            'down spin': get_total_spin(x_dimension, y_dimension, 'spin-down')
        }
        _h, _v = get_hva_commuting_hopping_terms(x_dimension, y_dimension, periodic)
        self.hvaGenerators = {
            'horizontal': [jordan_wigner(generator) for generator in _h],
            'vertical': [jordan_wigner(generator) for generator in _v],
            'coulomb': jordan_wigner(self.fermionOperators['coulomb'])
        }
        self.qmlOperators = {
            'particle number': QubitOperator_to_qmlHamiltonian(self.fermionOperators['particle number']),
            'up spin': QubitOperator_to_qmlHamiltonian(self.fermionOperators['up spin']),
            'down spin': QubitOperator_to_qmlHamiltonian(self.fermionOperators['down spin'])
        }
        self.FT_transformation_matrix = fourier_transform_matrix(x_dimension, y_dimension)

        self.Nh = len(_h)
        self.Nv = len(_v)
        self.params = nn.ParameterDict({
            'theta_U': nn.Parameter(torch.zeros(reps+1), requires_grad=True),
            'theta_v': nn.Parameter(torch.zeros(reps * self.Nv), requires_grad=True),
            'theta_h': nn.Parameter(torch.zeros(reps * self.Nh), requires_grad=True),
        }).to(self.device)

        self.loss_history = []
        self.fidelity_history = []
        self.filename = f'./images/HVA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, layers={reps}, lr={lr}).png'
        self.wf_filepath = f'./models/ground_state_results/Hubbard-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}).pt'
        self.ground_state_energy, self.ground_state_wf = self.get_ground_state()

    def get_ground_state(self, show_properties=False):
        
        import os
        import pickle
        
        # Check if the file exists. 
        # If the file exists, load the ground state energy and wavefunction
        if os.path.exists(self.wf_filepath):
            with open(self.wf_filepath, 'rb') as file:
                loaded_tensor_dict = pickle.load(file)
                ground_state_energy = loaded_tensor_dict['energy']
                ground_state_wf = loaded_tensor_dict['wave function']
        
        # If the file does not exist, calculate the ground state energy and wavefunction, 
        # and then save them as a file
        else:
            ground_state_energy, ground_state_wf = jw_get_ground_state(
                                                        sparse_operator=get_sparse_operator(self.fermionHamiltonian),
                                                        particle_number=self.n_electrons,
                                                        spin_up=self.n_electrons // 2,
                                                        spin_down=self.n_electrons // 2
                                                    )
            ground_state_energy = torch.Tensor([ground_state_energy]).double()
            ground_state_wf = torch.complex(
                                    torch.Tensor(ground_state_wf.real).double(), torch.Tensor(ground_state_wf.imag).double()
                              )
            tensor_dict = {
                'energy': ground_state_energy,
                'wave function': ground_state_wf
            }
            with open(self.wf_filepath, 'wb') as file:
                pickle.dump(tensor_dict, file)

        # Todo: implement the properties of the hubbard model at the ground state
        if show_properties:
            pass

        return ground_state_energy.item(), ground_state_wf.to(self.device)

    def Trotterize_operator(self, theta, operator: QubitOperator):

        for pauliString, _ in operator.terms.items():
    
            if not pauliString:
                continue

            String = reduce(lambda a, b: a+b, [pauliString[i][1] for i in range(len(pauliString))])
            Indicies = [pauliString[i][0] for i in range(len(pauliString))]
            Pk = (String, Indicies)
            PauliStringRotation(2 * theta, Pk)
    
    def prepare_nonInteracting_groundState(self):
        
        initially_occupied_orbitals = list(range(self.n_electrons))
        transformation_matrix = self.FT_transformation_matrix[initially_occupied_orbitals]
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
    
    def circuit(self, theta_U, theta_h, theta_v):
        
        self.prepare_nonInteracting_groundState()
        
        for rep in range(self.reps):
            self.Trotterize_operator(theta_U[rep], self.hvaGenerators['coulomb'])
            for i in range(self.Nv):
                self.Trotterize_operator(theta_v[rep * self.Nv + i], self.hvaGenerators['vertical'][i])
            for i in range(self.Nh):
                self.Trotterize_operator(theta_h[rep * self.Nh + i], self.hvaGenerators['horizontal'][i])
        self.Trotterize_operator(theta_U[self.reps], self.hvaGenerators['coulomb'])

        return [qml.expval(self.qmlHamiltonian), 
                qml.expval(self.qmlOperators['up spin']), 
                qml.expval(self.qmlOperators['down spin']), 
                qml.expval(self.qmlOperators['particle number'])]

    def state(self, theta_U, theta_h, theta_v):
        
        self.prepare_nonInteracting_groundState()
        
        for rep in range(self.reps):
            self.Trotterize_operator(theta_U[rep], self.hvaGenerators['coulomb'])
            for i in range(self.Nv):
                self.Trotterize_operator(theta_v[rep * self.Nv + i], self.hvaGenerators['vertical'][i])
            for i in range(self.Nh):
                self.Trotterize_operator(theta_h[rep * self.Nh + i], self.hvaGenerators['horizontal'][i])
        self.Trotterize_operator(theta_U[self.reps], self.hvaGenerators['coulomb'])

        return qml.state()


    def run(self):
        
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        dev = qml.device('default.qubit.torch', wires=self.n_qubits)
        opt = optim.Adam(params=self.params.values(), lr=self.lr)
        circuit = self.circuit
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')

        state_eval_circuit = self.state
        model2 = qml.QNode(state_eval_circuit, dev, interface='torch', diff_method='backprop')

        def closure():
            opt.zero_grad()
            loss, up_spin, down_spin, num_particle = model(*self.params.values())
            loss.backward()
            return loss
        
        for i_epoch in range(self.n_epoch):
            
            opt.zero_grad()
            loss, up_spin, down_spin, num_particle = model(*self.params.values())
            state = model2(*self.params.values())
            fidelity = torch.inner(state.conj(), self.ground_state_wf).abs() ** 2
            loss.backward()
            opt.step()

            # opt.step(closure)
            # loss, up_spin, down_spin, num_particle = model(*self.params.values())
            # state = model2(*self.params.values())
            # fidelity = torch.inner(state.conj(), self.ground_state_wf).abs() ** 2

            self.loss_history.append(loss.item())
            self.fidelity_history.append(fidelity.item())

            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1} | energy: {loss.item(): 6f} | fidelity: {fidelity.item(): 4f} | up: {up_spin.item(): 2f} | down: {down_spin.item(): 2f} | particle: {num_particle.item(): 2f}')
            
            ax1.clear()
            ax1.plot(np.arange(i_epoch+1)+1, self.loss_history, marker='X', color='r', label='HVA')
            ax1.plot(np.arange(i_epoch+1)+1, np.full(i_epoch+1, self.ground_state_energy), ls='-', color='g', label='ED')
            ax1.set_xlabel('epochs')
            ax1.set_ylabel('energy')
            ax1.legend()
            ax1.grid()

            ax2.clear()
            ax2.plot(np.arange(i_epoch+1)+1, self.fidelity_history, marker='X', ls=':', color='coral')
            ax2.set_xlabel('epochs')
            ax2.set_ylabel('fidelity')
            ax2.grid()

            plt.savefig(self.filename)
    
if __name__ == '__main__':
    vqe = HVA(
        n_epoch=200,
        lr=1e-2,
        threshold=1e-3,
        reps=10,
        x_dimension=2,
        y_dimension=3,
        tunneling=1,
        coulomb=0.1,
    )
    
    vqe.run()

    # from openfermion import get_sparse_operator

    # gnd_energy, gnd_wave_fn = vqe.get_ground_state()
    # particle = get_sparse_operator(vqe.fermionOperators['particle number']).todense()
    # up = get_sparse_operator(vqe.fermionOperators['up spin'], n_qubits=12).todense()
    # down = get_sparse_operator(vqe.fermionOperators['down spin'], n_qubits=12).todense()

    # particle = torch.complex(torch.Tensor(particle.real).double(), torch.Tensor(particle.imag).double()).to(vqe.device)
    # up = torch.complex(torch.Tensor(up.real).double(), torch.Tensor(up.imag).double()).to(vqe.device)
    # down = torch.complex(torch.Tensor(down.real).double(), torch.Tensor(down.imag).double()).to(vqe.device)
    # print(gnd_energy)
    # print(gnd_wave_fn.conj() @ particle @ gnd_wave_fn)
    # print(gnd_wave_fn.conj() @ up @ gnd_wave_fn)
    # print(gnd_wave_fn.conj() @ down @ gnd_wave_fn)

    
