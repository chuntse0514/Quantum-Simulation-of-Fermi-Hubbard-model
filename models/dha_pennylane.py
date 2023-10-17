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
    reverse_jordan_wigner,
    get_sparse_operator,
    jw_get_ground_state_at_particle_number,
)
import matplotlib.pyplot as plt
from .utils import (
    QubitOperator_to_qmlHamiltonian,
    PauliStringRotation
)
from operators.fourier import fourier_transform
from operators.tools import get_interacting_term, get_quadratic_term
from operators.pool import hubbard_interaction_pool
from functools import reduce, partial

def Trotterize_generator(theta, generator: QubitOperator):
    
    for pauliString, coeff in generator.terms.items():
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
                 threshold1: float,
                 threshold2: float,
                 x_dimension: int,
                 y_dimension: int,
                 tunneling: int,
                 coulomb: float,
                 periodic=True,
                 spinless=False,
                 particle_hole_symmetry=False
                 ):
        
        self.fermionOperatorPool = hubbard_interaction_pool(x_dimension, y_dimension)
        self.qubitOperatorPool = {
            channel: [jordan_wigner(generator) for generator in generators]
            for channel, generators in self.fermionOperatorPool.items() 
        }
        self.gatesOperatorPool = {
            channel: [partial(Trotterize_generator, generator=generator) for generator in generators]
            for channel, generators in self.qubitOperatorPool.items()
        }

        self.n_epoch = n_epoch
        self.lr = lr
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.ratio = 0.3
        self.n_qubits = x_dimension * y_dimension * 2
        self.n_electrons = x_dimension * y_dimension
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.realSpaceHamiltonian = fermi_hubbard(x_dimension,
                                                  y_dimension,
                                                  tunneling,
                                                  coulomb,
                                                  periodic=periodic,
                                                  spinless=spinless,
                                                  particle_hole_symmetry=particle_hole_symmetry)
        self.momentumSpaceHamiltonian = fourier_transform(self.realSpaceHamiltonian, 
                                                          x_dimension, 
                                                          y_dimension)
        self.fermionHamiltonian = self.momentumSpaceHamiltonian
        self.qubitHamiltonian = jordan_wigner(self.momentumSpaceHamiltonian)
        self.quadratic_term = get_quadratic_term(self.momentumSpaceHamiltonian)
        self.interacting_term = get_interacting_term(self.momentumSpaceHamiltonian)
        self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(
                                    self.momentumSpaceHamiltonian)

        self.eval_params = nn.ParameterDict({
            channel: nn.Parameter(torch.zeros(len(generators)), requires_grad=True)
            for channel, generators in self.fermionOperatorPool.items()
        }).to(self.device)
        self.loss_history = {
            'epoch': [],
            'iteration': []
        }
        self.append_history = {}
        self.filename = f'./images/DHA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, lr={lr}).png'
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


    def circuit(self, mode=None):

        if mode == 'eval-grad':
            for i in range(self.n_electrons):
                qml.PauliX(wires=i)
            for channel, gates in self.gatesOperatorPool.items():
                for i, gate in enumerate(gates):
                    gate(self.eval_params[channel][i])

            return qml.expval(self.qmlHamiltonian)

        elif mode == 'train':
            for i in range(self.n_electrons):
                qml.PauliX(wires=i)
            for channel, gates in self.selected_gates.items():
                for i, gate in enumerate(gates):
                    gate(self.train_params[channel][i])

            return qml.expval(self.qmlHamiltonian)

    def select_gates(self):
        
        dev = qml.device('default.qubit.torch', wires=self.n_qubits)
        model = qml.QNode(self.circuit, dev, interface='torch', diff_method='backprop')
        loss = model(mode='eval-grad')
        loss.backward()
        channel_max_grads = [torch.max(params.grad) for params in self.eval_params.values()]
        total_max_grad = max(channel_max_grads).cpu().numpy()
        
        self.selected_gates = {}
        self.selected_operators = {}
        self.selected_indices = {}
        self.max_grads = {}
        for channel, params in self.eval_params.items():
            selected_indices = np.arange(len(params))[params.grad.cpu().numpy() > total_max_grad * self.ratio]
            self.selected_gates[channel] = [self.gatesOperatorPool[channel][index] for index in selected_indices]
            self.selected_operators[channel] = [self.fermionOperatorPool[channel][index] for index in selected_indices]
            self.selected_indices[channel] = selected_indices.tolist()
            self.max_grads[channel] = params.grad.cpu().numpy()[selected_indices].tolist()

        self.train_params = nn.ParameterDict({
            channel: nn.Parameter(torch.zeros(len(gates)), requires_grad=True)
            for channel, gates in self.selected_gates.items()
        }).to(self.device)

        # zero out the accumulated gradient of the parameters
        for params in self.eval_params.values():
            params.grad.data.zero_()

        return total_max_grad

    def run(self):

        plt.ion()
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        for i_epoch in range(self.n_epoch):

            dev = qml.device('default.qubit.torch', wires=self.n_qubits)
            model = qml.QNode(self.circuit, dev, interface='torch', diff_method='backprop')
            
            total_max_grad = self.select_gates()
            print('Find gates')
            print_dict(self.selected_operators) 
            print('with gradients') 
            print_dict(self.max_grads)

            if total_max_grad < self.threshold1:
                break
            
            opt = optim.Adam(params=self.train_params.values(), lr=self.lr)
            while True:
                opt.zero_grad()
                loss = model(mode='train')
                loss.backward()
                opt.step()
                self.loss_history['iteration'].append(loss.item())

                grad_vec = torch.Tensor([]).to(self.device)
                for params in self.train_params.values():
                    grad_vec = torch.cat((grad_vec, params.grad), dim=0)
                grad_norm = torch.linalg.vector_norm(grad_vec).item()

                print(loss.item(), grad_norm)

                if grad_norm < self.threshold2:
                    break
            self.loss_history['epoch'].append(loss.item())

            # Todo: Transform the Hamiltonian according to the parameters
            for channel, indices in list(self.selected_indices.items())[::-1]:
                for index, param in zip(indices, self.train_params[channel]):
                    generator = self.qubitOperatorPool[channel][index]
                    for pauliString, coeff in list(generator.terms.items())[::-1]:
                        if not pauliString:
                            continue

                        theta = param.item() * coeff.real
                        pauli = QubitOperator(pauliString)
                        first_term = np.sin(theta) * (-1j / 2) * (self.qubitHamiltonian * pauli - pauli * self.qubitHamiltonian)
                        second_term = 1 / 2 * (1 - np.cos(theta)) * (pauli * self.qubitHamiltonian * pauli - self.qubitHamiltonian)
                        self.qubitHamiltonian += first_term + second_term
                        self.qubitHamiltonian.compress()

            self.fermionHamiltonian = reverse_jordan_wigner(self.qubitHamiltonian)
            self.fermionHamiltonian.compress(1e-6)
            self.qubitHamiltonian.compress(1e-6)

            self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(self.qubitHamiltonian)

            print('----transformed hamiltonian----')
            print(self.fermionHamiltonian)
            print('')

            ##############################
            
            ax1.clear()
            length = len(self.loss_history['iteration'])
            ax1.plot(np.arange(length)+1, self.loss_history['iteration'], color='coral', marker='X', ls='', label='DHA')
            ax1.plot(np.arange(length)+1, np.full(length, self.ground_state_energy), color='violet', label='ED')
            ax1.set_xlabel('iteration')
            ax1.set_ylabel('energy')
            ax1.legend()
            ax1.grid()
            
            ax2.clear()
            length = len(self.loss_history['epoch'])
            ax2.plot(np.arange(length)+1, self.loss_history['epoch'], color='yellowgreen', marker='X', ls='', label='DHA')
            ax2.plot(np.arange(length)+1, np.full(length, self.ground_state_energy), color='violet', label='ED')
            ax2.set_xlabel('epoch')
            ax2.set_ylabel('energy')
            ax2.legend()
            ax2.grid()

            plt.pause(0.01)
            plt.savefig(self.filename)
        
        plt.ioff()
        plt.show()
        

if __name__ == '__main__':
    vqe = DHA(
        n_epoch=5,
        lr=1e-2,
        threshold1=1e-2,
        threshold2=1e-2,
        x_dimension=3,
        y_dimension=3,
        tunneling=1,
        coulomb=2
    )

    print(vqe.quadratic_term)
    vqe.run()
