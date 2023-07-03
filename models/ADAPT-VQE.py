import qiskit 
from qiskit import QuantumCircuit, QuantumRegister
import torch
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

class AdaptVQE:

    def __init__(self,
                 molecule: MolecularData,
                 pool: dict,
                 maxIter: int,
                 lr,
                 epsilon,
                 ) -> None:
        self.molecule = molecule
        pool = [jordan_wigner(op) for op in pool]
        self.pool = [QubitOperator_to_SparsePauliOp(op) for op in pool]                      # convert the pool element to SparsePauliOp
        self.maxIter = maxIter
        self.lr = lr
        self.epsilon = epsilon

        self.n_electrons = molecule.n_electrons
        self.n_qubits = molecule.n_qubits

        # openfermion hamiltonians
        self.hamiltonain = molecule.get_molecular_hamiltonain()                              # fermionic hamiltonian
        self.sparseHamiltonian = get_sparse_operator(self.hamiltonian, self.n_qubits)        # sparse fermionic hamiltonian
        self.qubitHamiltonian = jordan_wigner(self.hamiltonian)                              # qubit hamiltonian

        # qiskit hamiltonians
        self.pauliHamiltonian = QubitOperator_to_SparsePauliOp(self.qubitHamiltonian, self.n_qubits)
        

        self.estimator = Estimator()

        hf_circ = prepare_HF_state(self.n_electrons, self.n_qubits)
        self.circuit = QuantumCircuit(self.n_qubits).compose(hf_circ)
        self.params = []
        self.operator_id = []
        self.num_iter = 0
        self.modelPath = './'
    
    def construct_qnn(self, observables) -> TorchConnector:
        qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=self.estimator,
            observables=observables,
            weight_params=self.params,
        )

        return TorchConnector(qnn, self.params)


    def calculatePoolElementGradient(self, op):
        # The usage of the Estimator is refer to the link below
        # https://qiskit.org/documentation/apidoc/primitives.html

        commutator = self.pauliHamiltonian.dot(op) - op.dot(self.pauliHamiltonian)

    def selectOperator(self):
        pass

    def updateAnsatz(self):
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def train(self):
        pass


