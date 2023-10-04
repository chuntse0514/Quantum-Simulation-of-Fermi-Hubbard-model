import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from openfermion import (
    QubitOperator,
    FermionOperator,
    fermi_hubbard,
    jordan_wigner,
    get_sparse_operator,
    jw_get_ground_state_at_particle_number,
    slater_determinant_preparation_circuit
)
import matplotlib.pyplot as plt
from .utils import (
    QubitOperator_to_qmlHamiltonian,
    PauliStringRotation
)
from operators.fourier import fourier_transform_matrix , inverse_fourier_transform
from operators.tools import get_interacting_term, get_quadratic_term
from operators.pool import hubbard_interation_pool_modified
from functools import reduce, partial
from collections import OrderedDict

def Trotterize_generator(theta, generator: QubitOperator):

    if isinstance(generator, FermionOperator):
        generator = jordan_wigner(generator)

    sorted_generator_terms = OrderedDict(sorted(generator.terms.items(), key=lambda item: len(item[0])))
    
    for pauliString, coeff in sorted_generator_terms.items():
        if not pauliString:
            continue
        String = reduce(lambda a, b: a+b, [pauliString[i][1] for i in range(len(pauliString))])
        Indicies = [pauliString[i][0] for i in range(len(pauliString))]
        Pk = (String, Indicies)
        PauliStringRotation(theta * coeff.real, Pk)

def print_dict(op_dict):
    for channel, elements in op_dict.items():
        print(channel, ':')
        if not elements:
            continue
        if isinstance(elements[0], FermionOperator):
            for element in elements:
                print(element)
        else:
            print(elements)

        print('')
        

class DHA:
    def __init__(self,
                 n_epoch: int,
                 lr: float,
                 reps: int,
                 threshold: float,
                 x_dimension: int,
                 y_dimension: int,
                 tunneling: int,
                 coulomb: float,
                 periodic=True,
                 spinless=False,
                 particle_hole_symmetry=False
                 ):
        
        self.fermionOperatorPool = hubbard_interation_pool_modified(x_dimension, y_dimension)
        self.fermionOperatorPool = {
            channel: inverse_fourier_transform(operator, x_dimension, y_dimension)
            for channel, operator in self.fermionOperatorPool.items()
        }
        self.qubitOperatorPool = {
            channel: jordan_wigner(generator)
            for channel, generator in self.fermionOperatorPool.items() 
        }
        for q_op in self.qubitOperatorPool.values():
            q_op.compress()

        self.n_epoch = n_epoch
        self.lr = lr
        self.threshold = threshold
        self.reps = reps
        self.n_qubits = x_dimension * y_dimension * 2
        self.n_electrons = x_dimension * y_dimension
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
        self.FT_transformation_matrix = fourier_transform_matrix(x_dimension, y_dimension)

        self.params = nn.ParameterList([nn.Parameter(torch.normal(0, 0.03, size=(5,))) for _ in range(reps)]).to(self.device)
        self.filename = f'./images/DHA_revised-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, lr={lr}).png'
        self.wf_filepath = f'./models/ground_state_results/Hubbard-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}).pt'
        self.ground_state_energy, self.ground_state_wf = self.get_ground_state()
        self.loss_history = []

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
            ground_state_energy, ground_state_wf = jw_get_ground_state_at_particle_number(
                                                        sparse_operator=get_sparse_operator(self.fermionHamiltonian),
                                                        particle_number=self.n_electrons
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

    def circuit(self):

        self.prepare_nonInteracting_groundState()

        for rep in range(self.reps):
            for i, generator in enumerate(self.qubitOperatorPool.values()):
                Trotterize_generator(self.params[rep][i], generator)

        return qml.expval(self.qmlHamiltonian)

    def run(self):

        plt.ion()
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        dev = qml.device('default.qubit.torch', wires=self.n_qubits)
        model = qml.QNode(self.circuit, dev, interface='torch', diff_method='backprop')
        opt = optim.Adam(params=self.params, lr=self.lr)

        for i_epoch in range(self.n_epoch):
            
            opt.zero_grad()
            loss = model()
            loss.backward()
            opt.step()

            self.loss_history.append(loss.item())
            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1} | energy: {loss.item(): 6f}')

            ##############################
            
            # ax1.clear()
            # length = len(self.loss_history['iteration'])
            # ax1.plot(np.arange(length)+1, self.loss_history['iteration'], color='coral', marker='X', ls='', label='DHA')
            # ax1.plot(np.arange(length)+1, np.full(length, self.ground_state_energy), color='violet', label='ED')
            # ax1.set_xlabel('iteration')
            # ax1.set_ylabel('energy')
            # ax1.legend()
            # ax1.grid()
            
            # ax2.clear()
            # length = len(self.loss_history['epoch'])
            # ax2.plot(np.arange(length)+1, self.loss_history['epoch'], color='yellowgreen', marker='X', ls='', label='DHA')
            # ax2.plot(np.arange(length)+1, np.full(length, self.ground_state_energy), color='violet', label='ED')
            # ax2.set_xlabel('epoch')
            # ax2.set_ylabel('energy')
            # ax2.legend()
            # ax2.grid()

            # plt.pause(0.01)
            # plt.savefig(self.filename)
        
        plt.ioff()
        plt.show()
        

if __name__ == '__main__':
    vqe = DHA(
        n_epoch=100,
        lr=1e-2,
        reps=1,
        threshold=1e-2,
        x_dimension=2,
        y_dimension=2,
        tunneling=1,
        coulomb=2
    )

    vqe.run()