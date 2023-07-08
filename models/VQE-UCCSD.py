import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import time

from qiskit.primitives import Estimator
from qiskit.utils import algorithm_globals
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

import qiskit_nature
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
qiskit_nature.settings.use_pauli_sum_op = False

class VQE_UCCSD:
    
    def __init__(self,
                 molecule: ElectronicStructureProblem,
                 maxIter: int,
                 lr,
                 epsilon
                 ):
        self.molecule = molecule
        self.maxIter = maxIter
        self.lr = lr
        self.epsilon = epsilon

        self.n_qubits = molecule.num_spatial_orbitals
        self.n_particles = molecule.num_particles        # (n_alpha, n_beta)
        self.n_electrons = sum(molecule.num_particles)
        self.n_orbitals = molecule.num_spatial_orbitals
        self.nuclear_repulsion_energy = molecule.hamiltonian.nuclear_repulsion_energy

        self.fermionicHamiltonian = molecule.hamiltonian.second_q_op()
        self.pauliHamiltonian = JordanWignerMapper().map(self.fermionicHamiltonian)

        hf_state = HartreeFock(
            num_spatial_orbitals=self.n_orbitals,
            num_particles=self.n_particles,
            qubit_mapper=JordanWignerMapper()
        )

        self.circuit = UCCSD(
            num_spatial_orbitals=self.n_orbitals,
            num_particles=self.n_particles,
            qubit_mapper=JordanWignerMapper(),
            reps=1,
            initial_state=hf_state,
        )
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
        
        self.exact_diagonalization()
        model = self.construct_qnn()
        optimizer = Adam(model.parameters(), lr=self.lr, betas=(0.7, 0.999), weight_decay=5e-4)
        time1 = time.time()

        for iter in range(self.maxIter):
            
            optimizer.zero_grad()
            loss = model(torch.Tensor([]))
            self.loss_history.append(loss.detach().item() + self.nuclear_repulsion_energy)
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
    
    def exact_diagonalization(self):
        solver = GroundStateEigensolver(
                    JordanWignerMapper(),
                    NumPyMinimumEigensolver(),
                )
        result = solver.solve(self.molecule)
        self.ed_enrgy = result.groundenergy + self.nuclear_repulsion_energy


    def plot_training_result(self):

        plt.title('Energy-Iteration plot')
        plt.ylabel('Energy')
        plt.xlabel('Iteration')
        plt.xlim([1, self.maxIter+1])
        plt.plot(np.arange(1, self.maxIter+1), self.loss_history, color="royalblue", label='UCCSD')
        plt.axhline(self.ed_enrgy, color='tomato', label='Exact Diagonalization')
        plt.legend(loc="best")
        plt.grid()

        plt.show()
        
def H2(r) -> ElectronicStructureProblem:
    h2 = f'H 0.0 0.0 0.0; H 0.0 0.0 {r}'
    charge = 0
    spin = 0
    driver = PySCFDriver(h2, 
                        charge=0,
                        spin=0,
                        basis='sto3g')
    h2 = driver.run()
    return h2

def HeH_Ion(r) -> ElectronicStructureProblem:
    helonium = f'He 0.0 0.0 0.0; H 0.0 0.0 {r}'
    driver = PySCFDriver(helonium.format(r), 
                         charge=1,
                         spin=0,
                         basis='sto3g')
    helonium = driver.run()
    return helonium

def H4(r) -> ElectronicStructureProblem:
    h4 = f'H 0.0 0.0 {0}; H 0.0 0.0 {r}; H 0.0 0.0 {2*r}; H 0.0 0.0 {3*r};'
    driver = PySCFDriver(h4, 
                        charge=0,
                        spin=0,
                        basis='sto3g')
    h4 = driver.run()
    return h4

def test_vqe():
    molecule = H2(r=0.8)
    vqe = VQE_UCCSD(molecule, maxIter=100, lr=1e-2, epsilon=1e-3)
    vqe.run()

test_vqe()
    