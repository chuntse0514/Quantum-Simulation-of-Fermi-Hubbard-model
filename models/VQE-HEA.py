from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import openfermion as of
from openfermion import (
    MolecularData, get_sparse_operator, jordan_wigner
)
from .utils import (
    QubitOperator_to_SparsePauliOp,
    prepare_HF_state,
)
from qiskit.primitives import Estimator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import EfficientSU2
from qiskit.utils import algorithm_globals
from molecules import *
import matplotlib.pyplot as plt
import time

class VQE:
    
    def __init__(self,
                 molecule: MolecularData,
                 maxIter: int,
                 lr,
                 epsilon
                 ):
        self.molecule = molecule
        self.maxIter = maxIter
        self.lr = lr
        self.epsilon = epsilon

        self.n_qubits = molecule.n_qubits
        self.n_electrons = molecule.n_electrons
        self.n_orbitals = molecule.n_orbitals

        self.fermionicHamiltonian = molecule.get_molecular_hamiltonian()
        self.sparseHamiltonian = get_sparse_operator(self.fermionicHamiltonian)
        self.qubitHamiltonian = jordan_wigner(self.fermionicHamiltonian)
        self.pauliHamiltonian = QubitOperator_to_SparsePauliOp(self.qubitHamiltonian, self.n_qubits)

        hf_circ = prepare_HF_state(self.n_electrons, self.n_qubits)
        self.circuit = EfficientSU2(
            num_qubits=self.n_qubits,
            su2_gates=['rz', 'ry', 'rz'],
            entanglement='circular',
            insert_barriers=True,
            reps=5,
        )
        self.circuit = hf_circ.compose(self.circuit)
        self.params = self.circuit.parameters
        self.estimator = Estimator()
        self.loss_history = []
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    
    def construct_qnn(self):
        qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=self.estimator,
            observables=self.pauliHamiltonian,
            weight_params=self.params
        )
        initial_weights = algorithm_globals.random.random(qnn.num_weights)
        return TorchConnector(qnn, initial_weights=initial_weights).to(self.device)
    
    def run(self):
        
        model = self.construct_qnn()
        optimizer = Adam(model.parameters(), lr=self.lr, betas=(0.7, 0.999), weight_decay=5e-4)
        time1 = time.time()

        for iter in range(self.maxIter):
            
            optimizer.zero_grad()
            loss = model(torch.Tensor([]))
            self.loss_history.append(loss.detach().item())
            loss.backward()
            optimizer.step()
            for p in model.parameters():
                grad_norm = torch.linalg.vector_norm(p.grad)
                if grad_norm.item() < self.epsilon:
                    self.maxIter = iter + 1
                    time2 = time.time()
                    print(f'total evalutation time: {time2-time1}s')
                    self.plot_training_result()
                    return
                p.grad.zero_()

            if (iter + 1) % 5 == 0:
                print(f'iteration: {iter+1}, total energy: {self.loss_history[-1]}')
        
        time2 = time.time()
        print(f'total evalutation time: {time2-time1}s')
        self.plot_training_result()

    def plot_training_result(self):

        plt.title('Energy-Iteration plot')
        plt.ylabel('Energy')
        plt.xlabel('Iteration')
        plt.xlim([1, self.maxIter+1])
        plt.plot(np.arange(1, self.maxIter+1), self.loss_history, color="royalblue", label='HEA')
        plt.axhline(self.molecule.fci_energy, color='tomato', label='FCI')
        plt.axhline(self.molecule.ccsd_energy, color='yellowgreen', label='CCSD')
        plt.legend(loc="best")
        plt.grid()

        plt.show()
        
    
def test_VQE():
    molecule = H2(r=0.8)
    vqe = VQE(molecule, maxIter=100, lr=1e-2, epsilon=0.002)
    vqe.run()

test_VQE()