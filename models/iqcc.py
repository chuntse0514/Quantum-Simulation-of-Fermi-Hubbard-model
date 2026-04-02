import pennylane as qml
import jax
import jax.numpy as jnp
import optax
import numpy as np
import json
import os
from openfermion import (
    MolecularData, jordan_wigner, QubitOperator
)
import matplotlib.pyplot as plt
from .utils import (
    QubitOperator_to_qmlHamiltonian,
    PauliStringRotation
)
from functools import partial
import time

class IQCC:
    def __init__(self,
                 molecule: MolecularData,
                 n_epoch: int,
                 lr: float,
                 threshold: float,
                 load_model: bool = False
                 ):

        self.molecule = molecule
        self.n_epoch = n_epoch
        self.lr = lr
        self.threshold = threshold
        self.Ng = 8
        self.ratio = 0.1
        self.n_qubits = molecule.n_qubits
        self.n_electrons = molecule.n_electrons

        self.fermionHamiltonian = molecule.get_molecular_hamiltonian()
        self.currentHamiltonian = jordan_wigner(self.fermionHamiltonian)
        self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(self.currentHamiltonian)

        self.model_filepath = f'./results/saved_model/IQCC-{molecule.name}.npz'
        self.result_filepath = f'./results/vqe_results/IQCC-{molecule.name}.json'
        self.img_filepath = f'./results/vqe_results/IQCC-{molecule.name}.pdf'

        if load_model:
            self.load_model()
        else:
            self.params = {
                'theta': jnp.array([np.pi for _ in range(self.n_electrons)] + [0.0 for _ in range(self.n_qubits - self.n_electrons)]),
                'phi': jnp.zeros(self.n_qubits),
                'tau': jnp.zeros(self.Ng)
            }

            self.loss_history = {
                'iteration': [],
                'epoch': []
            }
        self.selectedGates = []

    def save_model(self):
        # We need to save currentHamiltonian too as it changes
        np.savez(self.model_filepath, **self.params, currentHamiltonian=str(self.currentHamiltonian))
        with open(self.result_filepath, 'w') as f:
            json.dump(self.loss_history, f)

    def load_model(self):
        if os.path.exists(self.model_filepath):
            loaded_data = np.load(self.model_filepath)
            self.params = {key: jnp.array(loaded_data[key]) for key in loaded_data.files if key != 'currentHamiltonian'}
            # Note: reconstructing QubitOperator from string is possible but complex,
            # for now we'll just store it as string in the file.
        if os.path.exists(self.result_filepath):
            with open(self.result_filepath, 'r') as f:
                self.loss_history = json.load(f)
        
    def get_circuit(self, params, tau=None, appendGates=None):
        
        if appendGates is None:
            
            for i in range(self.n_qubits):
                qml.RY(params['theta'][i], wires=i)
                qml.RZ(params['phi'][i], wires=i)
            
            for i, gate in enumerate(self.selectedGates):
                gate(params['tau'][i])

            return qml.expval(self.qmlHamiltonian)
        
        else:
            for i in range(self.n_qubits):
                qml.RY(params['theta'][i], wires=i)
                qml.RZ(params['phi'][i], wires=i)
            
            for i, gate in enumerate(appendGates):
                gate(tau[i])

            return qml.expval(self.qmlHamiltonian)

    def partition_hamiltonian(self):
        
        partitioned_hamiltonian = {}

        for pauliString, coeff in self.currentHamiltonian.terms.items():

            flip_indicies = []

            for index, pauli in pauliString:
                if pauli == 'X' or pauli == 'Y':
                    flip_indicies.append(index)
            
            flip_indicies = tuple(flip_indicies)

            if partitioned_hamiltonian.get(flip_indicies) is None:
                partitioned_hamiltonian[flip_indicies] = coeff * QubitOperator(pauliString)
            else:
                partitioned_hamiltonian[flip_indicies] += coeff * QubitOperator(pauliString)
        
        return partitioned_hamiltonian
    
    def select_operator(self):

        partitioned_hamiltonian = self.partition_hamiltonian()
        DIS_gates = []
        DIS_strings = []

        for flip_indicies in partitioned_hamiltonian.keys():

            if len(flip_indicies) == 0:
                continue

            Pk = ('Y' + 'X' * (len(flip_indicies)-1), flip_indicies)
            gate = partial(PauliStringRotation, pauliString=Pk)
            DIS_gates.append(gate)

            Pk_string = ''
            for pauli, index in zip(*Pk):
                Pk_string += pauli + str(index) + ' '
            DIS_strings.append(Pk_string[:-1])

        dev = qml.device('default.qubit', wires=self.n_qubits)
        
        @qml.qnode(dev, interface='jax')
        def model(tau_val):
            return self.get_circuit(self.params, tau=tau_val, appendGates=DIS_gates)
        
        tau_init = jnp.zeros(len(DIS_gates))
        grads = jax.grad(model)(tau_init)
        grads = np.abs(np.array(grads))

        max_grad = np.max(grads)
        if max_grad * self.ratio > self.threshold:
            self.Ng = np.sum(grads > max_grad * self.ratio)
        else:
            self.Ng = np.sum(grads > self.threshold)
        
        maxIndicies = np.argsort(grads)[::-1][:self.Ng]
        maxGrads = grads[maxIndicies]
        maxGates = [DIS_gates[index] for index in maxIndicies]
        maxOperators = [DIS_strings[index] for index in maxIndicies]

        return maxGates, maxOperators, maxGrads

    def run(self):

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        dev = qml.device('default.qubit', wires=self.n_qubits)

        for i_epoch in range(self.n_epoch):

            maxGates, maxOperators, maxGrads = self.select_operator()

            print(f'=== Found operators: {maxOperators} \n with gradients: {maxGrads} ====')

            if len(maxGrads) == 0:
                break

            self.selectedGates = maxGates
            self.params['tau'] = jnp.zeros(self.Ng)

            optimizer = optax.adam(self.lr)
            opt_state = optimizer.init(self.params)

            # Re-define QNode once per epoch because qmlHamiltonian has been updated
            @qml.qnode(dev, interface='jax')
            def loss_fn(p):
                return self.get_circuit(p)

            def train_step(params, opt_state):
                loss, grads = jax.value_and_grad(loss_fn)(params)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss, grads

            while True:
                self.params, opt_state, loss, grads = train_step(self.params, opt_state)
                self.loss_history['iteration'].append(float(loss))
                
                grad_norm = float(jnp.linalg.norm(jnp.concatenate([jax.tree_util.tree_flatten(grads)[0][i].flatten() for i in range(len(jax.tree_util.tree_flatten(grads)[0]))])))
                if grad_norm < self.threshold:
                    break
                print(float(loss), grad_norm)
            
            self.loss_history['epoch'].append(float(loss))
            tau_final = np.array(self.params['tau'])
            self.selectedGates = []

            maxOperators_ops = [QubitOperator(op) for op in maxOperators]
            for P_k, tau_k in zip(maxOperators_ops[::-1], tau_final[::-1]):
                first_term = np.sin(tau_k) * (-1j / 2) * (self.currentHamiltonian * P_k - P_k * self.currentHamiltonian)
                second_term = 1 / 2 * (1 - np.cos(tau_k)) * (P_k * self.currentHamiltonian * P_k - self.currentHamiltonian)
                self.currentHamiltonian += first_term + second_term
            self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(self.currentHamiltonian)
            
            print(f'epoch: {i_epoch+1}, total energy: {float(loss)}')

            length = len(self.loss_history['iteration'])
            ax1.clear()
            ax1.plot(np.arange(length)+1, self.loss_history['iteration'], marker='x', color='r', label='iqcc')
            ax1.plot(np.arange(length)+1, np.full(length, self.molecule.fci_energy), linestyle='-', color='g', label='FCI')
            ax1.set_xlabel('iterations')
            ax1.set_ylabel('energy')
            ax1.legend()
            ax1.grid()
            plt.pause(0.01)

            length = len(self.loss_history['epoch'])
            ax2.clear()
            ax2.plot(np.arange(length)+1, self.loss_history['epoch'], marker='^', color='b', label='iqcc')
            ax2.plot(np.arange(length)+1, np.full(length, self.molecule.fci_energy), linestyle='-', color='g', label='FCI')
            ax2.set_xlabel('epochs')
            ax2.set_ylabel('energy')
            ax2.legend()
            ax2.grid()

            plt.savefig(self.img_filepath)
            plt.pause(0.01)
            self.save_model()


if __name__ == '__main__':
    from molecules import *
    molecule = LiH(r=0.8)
    vqe = IQCC(
        molecule, n_epoch=5, lr=1e-2, threshold=1e-2
    )
    vqe.run()