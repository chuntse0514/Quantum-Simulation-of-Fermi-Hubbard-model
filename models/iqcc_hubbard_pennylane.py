import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from openfermion import (
    FermionOperator, 
    InteractionOperator, 
    QubitOperator,
    count_qubits,
    jordan_wigner,
    get_interaction_operator,
    get_sparse_operator,
    get_ground_state
)
import matplotlib.pyplot as plt
from .utils import (
    QubitOperator_to_qmlHamiltonian,
    PauliStringRotation
)
from functools import partial

class IQCC:
    def __init__(self,
                 fermion_hamiltonian: FermionOperator | InteractionOperator,
                 n_epoch: int,
                 lr: float,
                 threshold: float
                 ):
        
        self.n_epoch = n_epoch
        self.lr = lr
        self.threshold = threshold
        self.Ng = 0
        self.ratio = 0.1
        self.n_qubits = count_qubits(fermion_hamiltonian)
        self.n_electrons = self.n_qubits // 2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if isinstance(fermion_hamiltonian, FermionOperator):
            fermion_hamiltonian = get_interaction_operator(fermion_hamiltonian)
        self.fermionHamiltonian = fermion_hamiltonian
        self.currentHamiltonian = jordan_wigner(self.fermionHamiltonian)
        self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(self.currentHamiltonian)

        self.params = nn.ParameterDict({
            'theta': nn.Parameter(torch.Tensor([np.pi] * self.n_electrons + [0] * (self.n_qubits - self.n_electrons)), requires_grad=True),
            'phi': nn.Parameter(torch.zeros(self.n_qubits), requires_grad=True),
            'tau': nn.Parameter(torch.zeros(self.Ng), requires_grad=True)
        }).to(self.device)

        self.loss_history = {
            'iteration': [],
            'epoch': []
        }
        self.selectedGates = []
        self.ground_state_energy, self.ground_state_wf = get_ground_state(get_sparse_operator(fermion_hamiltonian))

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
        
    def partition_hamiltonian(self):

        partitioned_hamiltonian = {}

        for pauliString, coeff in self.currentHamiltonian.terms.items():

            flip_indicies = []

            for index, pauli in pauliString:
                if pauli == 'X' or pauli == 'Y':
                    flip_indicies.append(index)
            
            flip_indicies = tuple(flip_indicies)

            if partitioned_hamiltonian.get(flip_indicies) is None:
                partitioned_hamiltonian[flip_indicies] = coeff * QubitOperator(pauliString)
            else:
                partitioned_hamiltonian[flip_indicies] += coeff * QubitOperator(pauliString)
        
        return partitioned_hamiltonian

    def select_operator(self):

        partitioned_hamiltonian = self.partition_hamiltonian()
        DIS_gates = []
        DIS_strings = []

        for flip_indicies in partitioned_hamiltonian.keys():

            if len(flip_indicies) == 0:
                continue

            Pk = ('Y' + 'X' * (len(flip_indicies)-1), flip_indicies)
            gate = partial(PauliStringRotation, pauliString=Pk)
            DIS_gates.append(gate)

            Pk_string = ''
            for pauli, index in zip(*Pk):
                Pk_string += pauli + str(index) + ' '
            DIS_strings.append(Pk_string[:-1])

        dev = qml.device('default.qubit.torch', wires=self.n_qubits)
        tau = nn.Parameter(torch.zeros(len(DIS_gates)), requires_grad=True)
        circuit = self.get_circuit
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')
        loss = model(tau, DIS_gates)
        loss.backward()
        grads = tau.grad.detach().cpu().numpy()
        grads = np.abs(grads)

        max_grad = np.max(grads)
        if max_grad * self.ratio > self.threshold:
            self.Ng = np.sum(grads > max_grad * self.ratio)
        else:
            self.Ng = np.sum(grads > self.threshold)
        
        maxIndicies = np.argsort(grads)[::-1][:self.Ng]
        maxGrads = grads[maxIndicies]
        maxGates = [DIS_gates[index] for index in maxIndicies]
        maxOperators = [DIS_strings[index] for index in maxIndicies]

        return maxGates, maxOperators, maxGrads
    
    def run(self):

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        dev = qml.device('default.qubit.torch', wires=self.n_qubits)

        for i_epoch in range(self.n_epoch):

            maxGates, maxOperators, maxGrads = self.select_operator()

            print(f'=== Found operators: {maxOperators} \n with gradients: {maxGrads} ====')

            if len(maxGrads) == 0:
                break

            self.selectedGates = maxGates
            self.params['tau'] = nn.Parameter(torch.zeros(self.Ng), requires_grad=True).to(self.device)

            circuit = self.get_circuit
            model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')
            opt = optim.Adam(self.params.values(), lr=self.lr)

            while True:
                opt.zero_grad()
                loss = model()
                loss.backward()
                opt.step()
                self.loss_history['iteration'].append(loss.item())
                grad_vec = torch.cat((self.params['theta'].grad, self.params['phi'].grad, self.params['tau'].grad))
                grad_norm = torch.linalg.vector_norm(grad_vec)
                if grad_norm < self.threshold:
                    break
                print(loss.item(), grad_norm)
            
            self.loss_history['epoch'].append(loss.item())
            self.selectedGates = []

            maxOperators = [QubitOperator(op) for op in maxOperators]
            for P_k, tau_k in zip(maxOperators[::-1], self.params['tau'].detach().cpu().numpy()[::-1]):
                first_term = np.sin(tau_k) * (-1j / 2) * (self.currentHamiltonian * P_k - P_k * self.currentHamiltonian)
                second_term = 1 / 2 * (1 - np.cos(tau_k)) * (P_k * self.currentHamiltonian * P_k - self.currentHamiltonian)
                self.currentHamiltonian += first_term + second_term
            self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(self.currentHamiltonian)
            
            print(f'epoch: {i_epoch+1}, total energy: {loss.item()}')

            length = len(self.loss_history['iteration'])
            ax1.clear()
            ax1.plot(np.arange(length)+1, self.loss_history['iteration'], marker='x', color='r', label='iqcc')
            ax1.plot(np.arange(length)+1, np.full(length, self.ground_state_energy), linestyle='-', color='g', label='FCI')
            ax1.set_xlabel('iterations')
            ax1.set_ylabel('energy')
            ax1.legend()
            ax1.grid()
            plt.pause(0.01)

            length = len(self.loss_history['epoch'])
            ax2.clear()
            ax2.plot(np.arange(length)+1, self.loss_history['epoch'], marker='^', color='b', label='iqcc')
            ax2.plot(np.arange(length)+1, np.full(length, self.ground_state_energy), linestyle='-', color='g', label='ED')
            ax2.set_xlabel('epochs')
            ax2.set_ylabel('energy')
            ax2.legend()
            ax2.grid()

            plt.savefig('output.png')
            plt.pause(0.01)
    
if __name__ == '__main__':
    from openfermion import fermi_hubbard
    hamiltonian = fermi_hubbard(
        x_dimension=2,
        y_dimension=2,
        tunneling=1,
        coulomb=4,
        periodic=True,
        spinless=False
    )
    vqe = IQCC(
        fermion_hamiltonian=hamiltonian,
        n_epoch=100,
        lr=1e-2,
        threshold=5e-3
    )
    vqe.run()