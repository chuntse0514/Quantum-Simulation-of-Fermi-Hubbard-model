from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator, Sampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

import torch
import torch.nn as nn

import time
import numpy as np
import matplotlib.pyplot as plt


class VQD:
    def __init__(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        maxIter: int,
        k: int,
        beta: float,
        lr: float,
        threshold: float
    ) -> None:
        
        self.estimator = Estimator()
        self.sampler = Sampler()
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self.maxIter = maxIter
        self.k = k
        self.beta = beta
        self.lr = lr
        self.threshold = threshold

        self.loss_history = [[] for _ in range(k)]
        self.energy_history = [[] for _ in range(k)]
        self.params = np.zeros((k, circuit.num_parameters))
    
    def construct_estimator_qnn(self):

        qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=self.estimator,
            observables=self.hamiltonian,
            weight_params=self.circuit.parameters
        )

        init_params = np.zeros(self.circuit.num_parameters)

        return TorchConnector(qnn, initial_weights=init_params)
    
    def construct_sampler_qnn(self, index):

        params1 = self.params[index]
        circuit1 = self.circuit.inverse().bind_parameters(params1)
        circuit2 = self.circuit
        composed_circuit = circuit2.compose(circuit1)

        qnn = SamplerQNN(
            circuit=composed_circuit,
            sampler=self.sampler,
            weight_params=composed_circuit.parameters
        )

        init_params = np.zeros(composed_circuit.num_parameters)

        return TorchConnector(qnn, initial_weights=init_params)
    
    def run(self):

        time1 = time.time()

        print('\n\n ==== Start to train VQD ====')

        for j in range(self.k):
            
            estimator_qnn = self.construct_estimator_qnn()
            sampler_qnns = [self.construct_sampler_qnn(k) for k in range(j)]

            # overwrite the parameters of sampler qnn by estimator qnn to ensure 
            # that thet parameters share the same memory and can be optimized together
            for sampler_qnn in sampler_qnns:
                sampler_qnn._weights = estimator_qnn._weights

            optimizer = torch.optim.Adam(params=estimator_qnn.parameters(), lr=self.lr)
            n_iter = 0

            print(f'==== The {j}-th state is being trained ... ====')

            while n_iter < self.maxIter:
                
                n_iter += 1
                energy = estimator_qnn(torch.Tensor([]))
                loss = energy.clone()
                for sampler_qnn in sampler_qnns:
                    # The fidelity is the first element of the output of sampler_qnn
                    loss += self.beta * sampler_qnn(torch.Tensor([]))[0].clone()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.loss_history[j].append(loss.detach().item())
                self.energy_history[j].append(energy.detach().item())

                if n_iter % 5 == 0:
                    print(f'iteration: {n_iter}, total loss: {loss.detach().item()}, total energy: {energy.detach().item()}')
                
                for p in estimator_qnn.parameters():
                    self.params[j] = p.detach().numpy()
                    grad_norm = torch.linalg.vector_norm(p.grad, ord=2)
                
                if grad_norm < self.threshold:
                    print(f'==== The {j}-th state has converged at energy {energy.item()} ====')
                    break

        time2 = time.time()
        print(f'total evalutation time: {time2-time1}s')
        self.plot_training_result()

    def plot_training_result(self):

        fig = plt.figure()
        plt.title('Energy-Iteration plot')
        plt.ylabel('Energy')
        plt.xlabel('Iteration')
        plt.xlim([1, self.maxIter+1])
        colors = [(0.1, 0.5, x / len(self.energy_history)) for x in range(len(self.energy_history))]
        for i in range(self.k):
            plt.plot(np.arange(1, len(self.energy_history[i])+1), self.energy_history[i], color=colors[i], label=f'{i}-th state')

        plt.legend(loc="best")
        plt.grid()

        plt.savefig('./models/output/vqd-out.png')

if __name__ == '__main__':
    
    from molecules import *
    from operators.pool import spin_complemented_pool
    from models.adapt_vqe import AdaptVQE
    molecule = H2(r=0.8)
    pool = spin_complemented_pool(molecule.n_electrons, molecule.n_orbitals)

    adaptvqe = AdaptVQE(molecule, pool, maxIter=100, lr=1e-2, threshold=1e-3)
    adaptvqe.run()

    circuit = adaptvqe.circuit
    hamiltonian = adaptvqe.qubitHamiltonian

    vqd = VQD(
        circuit=circuit,
        hamiltonian=hamiltonian,
        maxIter=100,
        k=3,
        beta=3,
        lr=1e-2,
        threshold=1e-3
    )

    vqd.run()