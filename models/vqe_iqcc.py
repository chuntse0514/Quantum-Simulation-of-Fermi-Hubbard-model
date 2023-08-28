from qiskit import QuantumCircuit, QuantumRegister
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import openfermion as of
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

class IQCC:
    def __init__(self,
                 molecule: MolecularData,
                 maxIter: int,
                 lr: float,
                 threshold: float
                 ):
        
        self.molecule = molecule
        self.maxIter = maxIter
        self.lr = lr
        self.threshold = threshold
        self.Ng = 3
        self.n_qubits = molecule.n_qubits
        self.n_electrons = molecule.n_electrons

        self.fermionicHamiltonian = molecule.get_molecular_hamiltonian()
        self.qubitHamiltonian = QubitOperator_to_SparsePauliOp(jordan_wigner(self.fermionicHamiltonian), self.n_qubits)
        self.currentHamiltonian = self.qubitHamiltonian

        self.estimator = Estimator()
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = np.array([np.pi for _ in range(self.n_electrons)] + [0 for _ in range(self.n_qubits - self.n_electrons)])
        self.phi = np.array([0 for _ in range(self.n_qubits)])
        for i in range(self.n_qubits):
            self.circuit.ry(Parameter(f'theta_{i}'), qubit=i)
            self.circuit.rz(Parameter(f'phi_{i}'), qubit=i)
        self.loss_history = []
        self.num_generator = []

    def get_qnn1(self, observables: list[SparsePauliOp]):
        qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=self.estimator,
            observables=observables,
            weight_params=self.circuit.parameters,
        )

        return TorchConnector(qnn, initial_weights=np.concatenate((vqe.phi, vqe.theta), axis=0))
    
    def get_qnn2(self, circuit: QuantumCircuit):
        qnn = EstimatorQNN(
            circuit=circuit,
            estimator=self.estimator,
            observables=self.currentHamiltonian,
            weight_params=circuit.parameters,
        )

        num_generators = self.num_generator[-1]
        return TorchConnector(qnn, initial_weights=np.concatenate((np.zeros(num_generators), vqe.phi, vqe.theta), axis=0))
    
    def partition_hamiltonian(self):
        
        paritionedHamiltonian = {}

        for label, coeff in self.currentHamiltonian.to_list():
            
            flip_indicies = []

            for i, pauli in enumerate(label[::-1]):
                if pauli == 'X' or pauli == 'Y':
                    flip_indicies.append(i)

            flip_indicies = tuple(flip_indicies)

            if paritionedHamiltonian.get(flip_indicies) is None:
                paritionedHamiltonian[flip_indicies] = SparsePauliOp.from_list([(label, coeff)])
            else:
                paritionedHamiltonian[flip_indicies] += SparsePauliOp.from_list([(label, coeff)])

        return paritionedHamiltonian
    
    def select_operators(self):

        partitionedHamiltonian = self.partition_hamiltonian()
        DIS = []
        commutators = []

        for flip_indicies in partitionedHamiltonian.keys():
            
            if len(flip_indicies) == 0:
                continue
            
            pauliString = 'Y' + 'X' * (len(flip_indicies) - 1)
            Pk = SparsePauliOp.from_sparse_list([(pauliString, flip_indicies, 1)], num_qubits=self.n_qubits)
            DIS.append(Pk)

            Sk = partitionedHamiltonian[flip_indicies]
            commutators.append(-1j / 2 * (Sk @ Pk - Pk @ Sk))
        
        model = self.get_qnn1(commutators)
        gradients = model(torch.Tensor([])).detach().numpy()
        gradients = np.abs(gradients)

        numIndicies = self.Ng if self.Ng < len(DIS) else len(DIS)
        self.num_generator.append(numIndicies)
        maxIndicies = np.argsort(gradients)[::-1][:numIndicies]
        maxGradients = gradients[maxIndicies]
        maxOperators = [DIS[i] for i in maxIndicies]

        return maxOperators, maxGradients
    
    def run(self):

        for i in range(self.maxIter):

            ops, grads = self.select_operators()

            print(f'Found operators: {[op.to_list()[0][0][::-1] for op in ops]} with gradients: {grads}')

            if grads[0] < self.threshold:
                break
            
            appendCircuit = QuantumCircuit(self.n_qubits)
            
            for k, op in enumerate(ops):

                pauliString = processPauliString(op)[0][0]

                _tau = Parameter(f'_tau_{k}')
                appendCircuit.compose(exponentialPauliString(_tau, pauliString, 1), qubits=pauliString[1], inplace=True)

            circuit = self.circuit.compose(appendCircuit)
            model = self.get_qnn2(circuit)

            opt = optim.Adam(params=model.parameters(), lr=self.lr)
            while True:
                opt.zero_grad()
                loss = model(torch.Tensor([]))
                loss.backward()
                opt.step()
                for p in model.parameters():
                    grads_norm = torch.linalg.vector_norm(p.grad)
                if grads_norm < self.threshold:
                    break
            
            for p in model.parameters():
                Ng = self.num_generator[-1]
                tau = p.detach().numpy()[:Ng]
                self.phi = p.detach().numpy()[Ng: Ng+self.n_qubits]
                self.theta = p.detach().numpy()[Ng+self.n_qubits:]
            
            for P_k, tau_k in zip(ops[::-1], tau[::-1]):
                first_term = np.sin(tau_k) * (-1j / 2) * (self.currentHamiltonian @ P_k - P_k @ self.currentHamiltonian)
                second_term = 1 / 2 * (1 - np.cos(tau_k)) * (P_k @ self.currentHamiltonian @ P_k - self.currentHamiltonian)
                self.currentHamiltonian += first_term + second_term
                self.currentHamiltonian = self.currentHamiltonian.simplify()

            print(f'epoch: {i+1}, total energy: {loss.item()}')

           

if __name__ == '__main__':
    
    from molecules import *
    
    molecule = H4(r=0.8)
    vqe = IQCC(molecule, maxIter=100, lr=1e-2, threshold=1e-3)
    vqe.run()

