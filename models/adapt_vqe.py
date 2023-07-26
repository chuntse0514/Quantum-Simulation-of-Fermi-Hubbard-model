import qiskit 
from qiskit import QuantumCircuit, QuantumRegister
import torch
import torch.nn as nn
from torch.optim import Adam
import openfermion as of
import numpy as np
from openfermion import (
    MolecularData, FermionOperator, jordan_wigner
)
from .utils import (
    QubitOperator_to_SparsePauliOp,
    processPauliString,
    exponentialPauliString
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
                 pool: list[FermionOperator],
                 maxIter: int,
                 lr: float,
                 threshold: float,
                 ):
        self.molecule = molecule
        self.pool = [QubitOperator_to_SparsePauliOp(jordan_wigner(op),                                                    # convert the pool element to SparsePauliOp
                                                    num_qubits=molecule.n_qubits) 
                                                    for op in pool]                                                       
        self.maxIter = maxIter
        self.lr = lr
        self.threshold = threshold

        self.n_qubits = molecule.n_qubits
        self.n_electrons = molecule.n_electrons
        self.n_orbitals = molecule.n_orbitals

        self.fermionicHamiltonian = molecule.get_molecular_hamiltonian()                                                   # fermionic hamiltonian                          
        self.qubitHamiltonian = QubitOperator_to_SparsePauliOp(jordan_wigner(self.fermionicHamiltonian), self.n_qubits)    # qubit hamiltonian
        
        self.estimator = Estimator()
        self.circuit = QuantumCircuit(self.n_qubits)
        self.params = np.array([])
        self.operator_id = []
        self.loss_history = []
        self.modelPath = './save_model/'
    
    def construct_qnn(self, observables) -> TorchConnector:
        qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=self.estimator,
            observables=observables,
            weight_params=self.circuit.parameters,
        )

        return TorchConnector(qnn, self.params)

    def calculatePoolElementGradient(self):
        commutators = [self.qubitHamiltonian @ op - op @ self.qubitHamiltonian for op in self.pool]
        qnn = self.construct_qnn(commutators)
        return qnn(torch.Tensor([])).detach().numpy()

    def selectOperator(self):
        gradients = self.calculatePoolElementGradient()
        # print(np.round(gradients, decimals=3))
        max_grad_index, max_grad = max(
            enumerate(gradients), key=lambda element: np.abs(element[1])
        )
        return max_grad_index, max_grad

    def updateAnsatz(self, theta: Parameter, operator: SparsePauliOp):
        """
        Args:
            theta (qiskit.circuit.Parameter): the parameter of the appended gate
            operator (qiskit.quantum_info.SparsePauliOp): the anti-Hermitian generator 
                                                         that we are going to exponentiate 
        """
        appendCircuit = QuantumCircuit(self.n_qubits)

        pauliStrings, coeffs = processPauliString(operator)

        for pauliString, coeff in zip(pauliStrings, coeffs):

            appendCircuit.compose(exponentialPauliString(theta, pauliString, coeff), qubits=pauliString[1], inplace=True)
        
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

                newTheta = Parameter(f'$θ_{i}$')
                operator = self.pool[max_grad_index]
                self.params = np.append(self.params, np.array([0]))
                self.updateAnsatz(newTheta, operator)

                # update the model parameter
                model = self.construct_qnn(self.qubitHamiltonian)
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
        self.circuit.draw('mpl')
        plt.savefig(f'./models/output/circuit-{self.molecule.name}.png')

        self.plot_training_result()

    def plot_training_result(self):
        
        fig = plt.figure()
        plt.title('Energy-Iteration plot')
        plt.ylabel('Energy')
        plt.xlabel('Iteration')
        plt.xlim([1, self.maxIter+1])
        plt.plot(np.arange(1, self.maxIter+1), self.loss_history, color="royalblue", label='Adapt-VQE')
        plt.axhline(self.molecule.ccsd_energy, color='yellowgreen', label='CCSD')
        plt.axhline(self.molecule.fci_energy, color='tomato', label='FCI')
        plt.legend(loc="best")
        plt.grid()

        plt.savefig(f'./models/output/adaptvqe-{self.molecule.name}.png')

    def saveModel(self):
        pass

    def loadModel(self):
        pass

if __name__ == '__main__':

    from molecules import *
    from operators.pool import spin_complemented_pool
    
    molecule = H2(r=0.8)
    pool = spin_complemented_pool(molecule.n_electrons, molecule.n_orbitals)
    vqe = AdaptVQE(molecule, pool, maxIter=100, lr=1e-2, threshold=1e-3)
    vqe.run()
