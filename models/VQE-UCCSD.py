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
from molecules import H2
import matplotlib.pyplot as plt
import qiskit_nature
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
qiskit_nature.settings.use_pauli_sum_op = False

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

        hf_state = HartreeFock(
            num_spatial_orbitals=self.n_orbitals,
            num_particles=(self.n_electrons//2, self.n_electrons//2),
            qubit_mapper=JordanWignerMapper()
        )

        self.circuit = UCCSD(
            num_spatial_orbitals=self.n_orbitals,
            num_particles=(self.n_electrons//2, self.n_electrons//2),
            qubit_mapper=JordanWignerMapper(),
            reps=1,
            initial_state=hf_state,
        )
        self.circuit.decompose().draw('mpl')
        plt.show()
        self.params = self.circuit.parameters
        self.estimator = Estimator()
        self.loss_history = []

    
    def construct_qnn(self):
        qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=self.estimator,
            observables=self.pauliHamiltonian,
            weight_params=self.params
        )
        initial_weights = algorithm_globals.random.random(qnn.num_weights)
        return TorchConnector(qnn, initial_weights=initial_weights)
    
    def run(self):
        
        model = self.construct_qnn()
        optimizer = Adam(model.parameters(), lr=self.lr, betas=(0.7, 0.999), weight_decay=5e-4)

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
                    self.plot_training_result()
                    return
                p.grad.zero_()

            if (iter + 1) % 5 == 0:
                print(f'iteration: {iter+1}, total energy: {self.loss_history[-1]}')

        self.plot_training_result()

    def plot_training_result(self):

        plt.title('Energy-Iteration plot')
        plt.ylabel('Energy')
        plt.xlabel('Iteration')
        plt.xlim([1, self.maxIter+1])
        plt.plot(np.arange(1, self.maxIter+1), self.loss_history, color="royalblue", label='UCCSD')
        plt.axhline(self.molecule.fci_energy, color='tomato', label='FCI')
        plt.axhline(self.molecule.ccsd_energy, color='yellowgreen', label='CCSD')
        plt.legend(loc="best")
        plt.grid()

        plt.show()
        
    
def test_VQE():
    molecule = H2(r=0.8)
    vqe = VQE(molecule, maxIter=100, lr=1e-2, epsilon=0.02)
    vqe.run()

test_VQE()