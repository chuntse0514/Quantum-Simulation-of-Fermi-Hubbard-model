import qiskit 
from qiskit import QuantumCircuit, QuantumRegister
import torch
import torch.nn as nn
from torch.optim import Adam
import openfermion as of
import numpy as np
from openfermion import (
    MolecularData, get_sparse_operator, jordan_wigner
)
from .utils import (
    QubitOperator_to_SparsePauliOp,
)
from qiskit.primitives import Estimator
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
import matplotlib.pyplot as plt
import time

class AdaptVQE:

    def __init__(self,
                 molecule: MolecularData,
                 pool: dict,
                 maxIter: int,
                 lr,
                 threshold,
                 ) -> None:
        self.molecule = molecule
        pool = [jordan_wigner(op) for op in pool]
        self.pool = [QubitOperator_to_SparsePauliOp(op, num_qubits=molecule.n_qubits) for op in pool]                      # convert the pool element to SparsePauliOp
        self.maxIter = maxIter
        self.lr = lr
        self.threshold = threshold

        self.n_qubits = molecule.n_qubits
        self.n_electrons = molecule.n_electrons
        self.n_orbitals = molecule.n_orbitals

        self.fermionicHamiltonian = molecule.get_molecular_hamiltonian()                              # fermionic hamiltonian
        self.sparseHamiltonian = get_sparse_operator(self.fermionicHamiltonian, self.n_qubits)        # sparse fermionic hamiltonian
        self.qubitHamiltonian = jordan_wigner(self.fermionicHamiltonian)                              # qubit hamiltonian
        self.pauliHamiltonian = QubitOperator_to_SparsePauliOp(self.qubitHamiltonian, self.n_qubits)
        
        self.estimator = Estimator()
        self.circuit = QuantumCircuit(self.n_qubits)
        self.params = np.array([])
        self.operator_id = []
        self.loss_history = []
        self.modelPath = './'
    
    def construct_qnn(self, observables) -> TorchConnector:
        qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=self.estimator,
            observables=observables,
            weight_params=self.circuit.parameters,
        )

        return TorchConnector(qnn, self.params)

    def calculatePoolElementGradient(self):
        commutators = [self. pauliHamiltonian @ op - op @ self.pauliHamiltonian for op in self.pool]
        qnn = self.construct_qnn(commutators)
        return qnn(torch.Tensor([])).detach().numpy()

    def selectOperator(self):
        gradients = self.calculatePoolElementGradient()
        # print(gradients)
        max_grad_index, max_grad = max(
            enumerate(gradients), key=lambda element: np.abs(element[1])
        )
        return max_grad_index, max_grad

    def updateAnsatz(self, theta: Parameter, operator: SparsePauliOp):
        
        appendCircuit = QuantumCircuit(self.n_qubits)

        for (pauliString, coeff) in operator.to_list():
            
            coeff = (coeff * 2j).real

            # Store the index that the pauli string acts non-trivially 
            # in order to apply the CNOT gate later.
            nonTrivialIndex = []
            
            # Apply the basis rotation unitary
            for index, pauli in enumerate(pauliString):
                if pauli == 'I':
                    continue
                elif pauli == 'X':
                    appendCircuit.ry(-np.pi/2, qubit=index)
                elif pauli == 'Y':
                    appendCircuit.rx(np.pi/2, qubit=index)
                
                nonTrivialIndex.append(index)

            # Apply the CNOT gates to compute the parity
            for i, q_index in enumerate(nonTrivialIndex[:-1]):
                
                next_q_index = nonTrivialIndex[i+1]
                appendCircuit.cx(q_index, next_q_index)
            
            # Apply the parameterized rotation-z gate
            last_q_index = nonTrivialIndex[-1]
            appendCircuit.rz(theta * coeff, last_q_index)

            # Apply the CNOT gates to uncompute the parity
            reversedNonTrivialIndex = list(reversed(nonTrivialIndex))
            for i, q_index in enumerate(reversedNonTrivialIndex[:-1]):
                
                next_q_index = reversedNonTrivialIndex[i+1]
                appendCircuit.cx(next_q_index, q_index)
            
            # Apply the basis rotation unitary
            for index, pauli in enumerate(pauliString):
                if pauli == 'X':
                    appendCircuit.ry(np.pi/2, qubit=index)
                elif pauli == 'Y':
                    appendCircuit.rx(-np.pi/2, qubit=index)
            
            appendCircuit.barrier()
        
        self.circuit = self.circuit.compose(appendCircuit)

    def run(self):

        time1 = time.time()

        # prepare Hartree-Fock state
        for i in range(self.n_electrons):
            self.circuit.x(i)
        
        print(f'==== Start to train Adapt-VQE for {self.molecule.name} ====')
        for i in range(self.maxIter):

            # select the operator with largest gradient
            max_grad_index, max_grad = self.selectOperator()
            isAppend = False
            try:
                if self.operator_id[-1] != max_grad_index:
                    self.operator_id.append(max_grad_index)
                    isAppend = True
            except:
                self.operator_id.append(max_grad_index)
                isAppend = True

            print(f'==== Found maximium gradient {np.abs(max_grad)} at index {max_grad_index} ====')

            # termiate the Adapt-VQE if the largest gradient is less than the threshold
            if np.abs(max_grad) < self.threshold:
                print(f'==== Adapt-VQE has terminated at iteration {i+1} since the convergence criterion is satisfied. ====')
                self.maxIter = i
                break
            
            # update the ansatz circuit according to the operator we selected,
            # and initialize the new parameter to 0.
            if isAppend:
                newTheta = Parameter(f'θ_{i}')
                operator = self.pool[max_grad_index]
                self.params = np.append(self.params, np.array([0]))
                self.updateAnsatz(newTheta, operator)

            # update the model parameter
            if isAppend:
                model = self.construct_qnn(self.pauliHamiltonian)
                optimizer = Adam(model.parameters(), lr=self.lr * np.exp(-0.1 * i), betas=(0.7, 0.999), weight_decay=5e-4)
            optimizer.zero_grad()
            loss = model(torch.Tensor([]))
            loss.backward()
            optimizer.step()
            for p in model.parameters():
                self.params = p.detach().numpy()
            self.loss_history.append(loss.detach().item())

            if (i + 1) % 5 == 0:
                print(f'iteration: {i+1}, total energy: {self.loss_history[-1]}')

        time2 = time.time()
        print(f'total evalutation time: {time2-time1}s')
        self.plot_training_result()

    def plot_training_result(self):
        
        plt.title('Energy-Iteration plot')
        plt.ylabel('Energy')
        plt.xlabel('Iteration')
        plt.xlim([1, self.maxIter+1])
        plt.plot(np.arange(1, self.maxIter+1), self.loss_history, color="royalblue", label='Adapt-VQE')
        plt.axhline(self.molecule.ccsd_energy, color='yellowgreen', label='CCSD')
        plt.axhline(self.molecule.fci_energy, color='tomato', label='FCI')
        plt.legend(loc="best")
        plt.grid()

        plt.show()

    def saveModel(self):
        pass

    def loadModel(self):
        pass

from molecules import *
from operatorPools.fermion_pool import Fermionic_Pool
import matplotlib.pyplot as plt

def test_vqe():
    molecule = H2(r=0.8)
    pool = Fermionic_Pool(molecule.n_electrons, molecule.n_orbitals)
    pool = pool.spin_complemented_gsd_excitation()
    vqe = AdaptVQE(molecule, pool, maxIter=100, lr=1e-2, threshold=1e-3)
    vqe.run()

test_vqe()