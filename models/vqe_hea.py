import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from openfermion import (
    MolecularData, jordan_wigner
)
from .utils import QubitOperator_to_qmlHamiltonian
import matplotlib.pyplot as plt
import time

class VQE:

    def __init__(self,
                 molecule: MolecularData,
                 n_epoch: int,
                 reps: int,
                 lr: float,
                 threshold: float):
        
        self.molecule = molecule
        self.n_epoch = n_epoch
        self.reps = reps
        self.lr = lr
        self.threshold = threshold

        self.n_qubits = molecule.n_qubits
        self.n_electrons = molecule.n_electrons
        self.n_orbitals = molecule.n_orbitals

        self.fermionHamiltonian = molecule.get_molecular_hamiltonian()
        self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(
            jordan_wigner(self.fermionHamiltonian)
        )

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.params = nn.ParameterList([
            nn.Parameter((2 * torch.rand((reps+1, self.n_qubits, 3)) - 1) * np.pi, requires_grad=True)
        ]).to(self.device)
        self.loss_history = []

    def circuit(self):

        for rep in range(self.reps):
            for q in range(self.n_qubits):
                qml.RX(self.params[0][rep, q, 0], wires=q)
                qml.RY(self.params[0][rep, q, 1], wires=q)
                qml.RZ(self.params[0][rep, q, 2], wires=q)
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q+1)%self.n_qubits])
        for q in range(self.n_qubits):
            qml.RX(self.params[0][self.reps-1, q, 0], wires=q)
            qml.RY(self.params[0][self.reps-1, q, 1], wires=q)
            qml.RZ(self.params[0][self.reps-1, q, 2], wires=q)

        return qml.expval(self.qmlHamiltonian)
    
    def run(self):

        dev = qml.device('default.qubit.torch', wires=self.n_qubits)
        circuit = self.circuit
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')
        opt = optim.Adam(self.params, lr=self.lr)

        start_time = time.time()

        plt.ion()
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 1, 1)

        for i_epoch in range(self.n_epoch):
            
            opt.zero_grad()
            loss = model()
            loss.backward()
            opt.step()
            self.loss_history.append(loss.item())

            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1}, total energy: {loss.item()}')

            ax1.clear()
            ax1.plot(np.arange(i_epoch+1)+1, self.loss_history, color='deepskyblue', marker='X', label='hea')
            ax1.plot(np.arange(i_epoch+1)+1, np.full(i_epoch+1, self.molecule.fci_energy), color='hotpink', ls='-', label='FCI')
            ax1.set_xlabel('epoch')
            ax1.set_ylabel('energy')
            ax1.legend()
            ax1.grid()
            plt.pause(0.01)

            grad_norm = torch.linalg.vector_norm(self.params[0].grad)
            if grad_norm < self.threshold:
                print(f'gradient norm is less than threshold {self.threshold}, break the loop!')
                break

        plt.ioff()
        plt.show()
        
        end_time = time.time()
        print(f'total evaluation time: {end_time-start_time}s')
             
if __name__ == '__main__':

    from molecules import *

    molecule = H2(r=0.8)
    vqe = VQE(molecule, n_epoch=100, reps=5, lr=1e-1, threshold=0.002)
    vqe.run()