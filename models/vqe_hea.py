import pennylane as qml
import jax
import jax.numpy as jnp
import optax
import numpy as np
import json
import os
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
                 threshold: float,
                 load_model: bool = False):
        
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

        self.model_filepath = f'./results/saved_model/HEA-{molecule.name}_reps{reps}.npz'
        self.result_filepath = f'./results/vqe_results/HEA-{molecule.name}_reps{reps}.json'
        self.img_filepath = f'./images/HEA-{molecule.name}_reps{reps}.pdf'

        if load_model:
            self.load_model()
        else:
            key = jax.random.PRNGKey(0)
            self.params = (jax.random.uniform(key, (reps+1, self.n_qubits, 3)) * 2 - 1) * np.pi
            self.loss_history = []

    def save_model(self):
        np.savez(self.model_filepath, params=self.params)
        with open(self.result_filepath, 'w') as f:
            json.dump({'loss_history': self.loss_history}, f)

    def load_model(self):
        if os.path.exists(self.model_filepath):
            self.params = jnp.array(np.load(self.model_filepath)['params'])
        if os.path.exists(self.result_filepath):
            with open(self.result_filepath, 'r') as f:
                self.loss_history = json.load(f)['loss_history']

    def circuit(self, params):

        for rep in range(self.reps):
            for q in range(self.n_qubits):
                qml.RX(params[rep, q, 0], wires=q)
                qml.RY(params[rep, q, 1], wires=q)
                qml.RZ(params[rep, q, 2], wires=q)
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q+1)%self.n_qubits])
        
        # Last layer
        rep = self.reps # Using reps index for the last set of gates
        for q in range(self.n_qubits):
            qml.RX(params[rep-1, q, 0], wires=q)
            qml.RY(params[rep-1, q, 1], wires=q)
            qml.RZ(params[rep-1, q, 2], wires=q)

        return qml.expval(self.qmlHamiltonian)
    
    def run(self):

        dev = qml.device('default.qubit', wires=self.n_qubits)
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(self.params)

        # Define QNode once since the circuit structure is fixed
        @qml.qnode(dev, interface='jax')
        def loss_fn(p):
            return self.circuit(p)

        def train_step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, grads

        start_time = time.time()

        plt.ion()
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 1, 1)

        for i_epoch in range(self.n_epoch):
            
            self.params, opt_state, loss, grads = train_step(self.params, opt_state)
            self.loss_history.append(float(loss))

            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1}, total energy: {float(loss)}')

            ax1.clear()
            ax1.plot(np.arange(i_epoch+1)+1, self.loss_history, color='deepskyblue', marker='X', label='hea')
            ax1.plot(np.arange(i_epoch+1)+1, np.full(i_epoch+1, self.molecule.fci_energy), color='hotpink', ls='-', label='FCI')
            ax1.set_xlabel('epoch')
            ax1.set_ylabel('energy')
            ax1.legend()
            ax1.grid()
            plt.pause(0.01)

            grad_norm = jnp.linalg.norm(grads)
            if grad_norm < self.threshold:
                print(f'gradient norm is less than threshold {self.threshold}, break the loop!')
                break

        plt.ioff()
        plt.savefig(self.img_filepath)
        self.save_model()
        
        end_time = time.time()
        print(f'total evaluation time: {end_time-start_time}s')
             
if __name__ == '__main__':

    from molecules import H2

    molecule = H2(r=0.8)
    vqe = VQE(molecule, n_epoch=100, reps=5, lr=1e-1, threshold=0.002)
    vqe.run()
             
if __name__ == '__main__':

    from molecules import *

    molecule = H2(r=0.8)
    vqe = VQE(molecule, n_epoch=100, reps=5, lr=1e-1, threshold=0.002)
    vqe.run()