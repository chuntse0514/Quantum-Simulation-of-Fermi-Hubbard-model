import os
import time
import json
from functools import reduce, partial

import numpy as np
import jax
import jax.numpy as jnp
import pennylane as qml
import optax
import matplotlib.pyplot as plt
from openfermion import (
    QubitOperator, FermionOperator, fermi_hubbard, jordan_wigner,
    get_sparse_operator, givens_decomposition_square,
    number_operator, up_index, down_index
)

from .utils import QubitOperator_to_qmlHamiltonian, PauliStringRotation
from linalg.exact_diagonalization import jw_get_ground_state
from operators.fourier import fourier_transform_matrix, fourier_transform
from operators.tools import get_interacting_term, get_quadratic_term
from operators.pool import hubbard_interaction_pool_simplified

# Global JAX/XLA configuration
jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def get_number_operator(n_sites, spin_type='total', spinless=False):
    n_orbitals = n_sites if spinless else 2 * n_sites
    op = FermionOperator()
    for i in range(n_sites):
        if spinless:
            op += number_operator(n_sites, i, 1.0)
        else:
            if spin_type in ['total', 'up']:
                op += number_operator(n_orbitals, up_index(i), 1.0)
            if spin_type in ['total', 'down']:
                op += number_operator(n_orbitals, down_index(i), 1.0)
    return op

def get_spin_operators(n_sites, spin_type):
    Sx, Sy, Sz = FermionOperator(), FermionOperator(), FermionOperator()
    n_orbitals = n_sites * 2
    for i in range(n_sites):
        u, d = up_index(i), down_index(i)
        Sx += FermionOperator(f'{u}^ {d}', 0.5) + FermionOperator(f'{d}^ {u}', 0.5)
        Sy += FermionOperator(f'{u}^ {d}', -0.5j) - FermionOperator(f'{d}^ {u}', -0.5j)
        Sz += FermionOperator(f'{u}^ {u}', 0.5) - FermionOperator(f'{d}^ {d}', 0.5)
    
    if spin_type == 'Sx': return Sx
    if spin_type == 'Sy': return Sy
    if spin_type == 'Sz': return Sz
    if spin_type == 'S^2': return Sx*Sx + Sy*Sy + Sz*Sz

def Trotterize_generator(theta, generator: QubitOperator):

    if isinstance(generator, FermionOperator):
        generator = jordan_wigner(generator)
    
    for pauliString, coeff in generator.terms.items():
        if not pauliString:
            continue
        String = reduce(lambda a, b: a+b, [pauliString[i][1] for i in range(len(pauliString))])
        Indicies = [pauliString[i][0] for i in range(len(pauliString))]
        Pk = (String, Indicies)
        PauliStringRotation(2 * theta * coeff.real, Pk)

def print_list(op_list):
    for op in op_list:
        print(str(op).replace('\n', ' '))

def get_non_interacting_ground_state_index(quadratic_hamiltonian: FermionOperator, n_qubits, n_spin_up, n_spin_down):

    spin_up_energies = {x: 0.0 for x in range(0, n_qubits, 2)}
    spin_down_energies = {x: 0.0 for x in range(1, n_qubits, 2)}
    
    for term, coeff in quadratic_hamiltonian.terms.items():
        index = term[0][0]
        if index % 2 == 0:
            spin_up_energies[index] = float(coeff.real)
        else:
            spin_down_energies[index] = float(coeff.real)

    spin_up_indices = sorted(spin_up_energies, key=spin_up_energies.get)[:n_spin_up]
    spin_down_indices = sorted(spin_down_energies, key=spin_down_energies.get)[:n_spin_down]

    # Format for beautiful printing
    spin_up_print = {k: round(v, 6) for k, v in spin_up_energies.items()}
    spin_down_print = {k: round(v, 6) for k, v in spin_down_energies.items()}

    print('spin up orbital energies:', spin_up_print)
    print('spin down orbital energies: ', spin_down_print)

    return spin_up_indices, spin_down_indices

class ADAPT:
    def __init__(self, n_epoch, threshold1, threshold2, x_dimension, y_dimension, 
                 n_electrons, n_spin_up, n_spin_down, tunneling, coulomb,
                 periodic=True, spinless=False, particle_hole_symmetry=False,
                 load_model=False, device='default.qubit', use_qng=False, qng_reg=1e-4):
        
        self.n_epoch, self.threshold1, self.threshold2 = n_epoch, threshold1, threshold2
        self.x_dim, self.y_dimension = x_dimension, y_dimension
        self.n_sites, self.n_qubits = x_dimension * y_dimension, x_dimension * y_dimension * 2
        self.n_electrons, self.n_spin_up, self.n_spin_down = n_electrons, n_spin_up, n_spin_down
        self.device_name, self.use_qng, self.qng_reg = device, use_qng, qng_reg
        self.ratio = 0.1

        # Operator pools
        self.fermionOperatorPool = hubbard_interaction_pool_simplified(x_dimension, y_dimension)
        self.qubitOperatorPool = [jordan_wigner(gen) for gen in self.fermionOperatorPool]
        self.gateOperatorPool = [partial(Trotterize_generator, generator=gen) for gen in self.qubitOperatorPool]

        # Hamiltonian and Observables
        self.fermionHamiltonian = fermi_hubbard(x_dimension, y_dimension, tunneling, coulomb,
                                                periodic=periodic, spinless=spinless,
                                                particle_hole_symmetry=particle_hole_symmetry)
        self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(self.fermionHamiltonian)
        
        self.fermionOperators = {
            'Sz': get_spin_operators(self.n_sites, 'Sz'),
            'S^2': get_spin_operators(self.n_sites, 'S^2')
        }
        self.qmlOperators = {k: QubitOperator_to_qmlHamiltonian(v) for k, v in self.fermionOperators.items()}

        # Fourier Transform setup
        self.FT_matrix = fourier_transform_matrix(x_dimension, y_dimension)
        self.decomposition, self.diagonal = givens_decomposition_square(self.FT_matrix)
        self.circuit_description = list(reversed(self.decomposition))
        
        k_quad = fourier_transform(get_quadratic_term(self.fermionHamiltonian), x_dimension, y_dimension)
        self.spin_up_indices, self.spin_down_indices = get_non_interacting_ground_state_index(
            k_quad, self.n_qubits, n_spin_up, n_spin_down
        )

        # File path management
        base_name = f"ADAPT-{x_dimension}x{y_dimension}_t{tunneling}_U{coulomb}_n{n_electrons}_up{n_spin_up}_down{n_spin_down}"
        hubbard_name = f"Hubbard-{x_dimension}x{y_dimension}_t{tunneling}_U{coulomb}_n{n_electrons}"
        
        self.img_filepath = f'./images/{base_name}.pdf'
        self.wf_filepath = f'./results/ground_state_results/{hubbard_name}.npz'
        self.result_filepath = f'./results/vqe_results/{base_name}.json'
        self.model_filepath = f'./results/saved_model/{base_name}.npz'
        self.log_filepath = f'./results/vqe_logs/{base_name}.log'

        self.ground_state_energy, self.ground_state_wfs = self.get_ground_state()
        
        if not load_model and os.path.exists(self.log_filepath):
            os.remove(self.log_filepath)
        
        if load_model:
            self.load_model()
        else:
            self.params = {'e': jnp.zeros(len(self.gateOperatorPool)), 't': jnp.array([])}
            self.selected_gates, self.selected_indices = [], []
            self.results = {k: [] for k in ['epoch loss', 'iteration loss', 'Sz', 'S^2', 'fidelity', 'n_params', 'selected operators']}

    def get_ground_state(self):
        
        # Check if the file exists. 
        # If the file exists, load the ground state energy and wavefunction
        if os.path.exists(self.wf_filepath):
            loaded_data = np.load(self.wf_filepath)
            ground_state_energy = float(loaded_data['energy'])
            ground_state_wfs = list(loaded_data['wave_function'])
        
        # If the file does not exist, calculate the ground state energy and wavefunction, 
        # and then save them as a file
        else:
            ground_state_energy, ground_state_wfs = jw_get_ground_state(
                                                        sparse_operator=get_sparse_operator(self.fermionHamiltonian),
                                                        particle_number=self.n_electrons,
                                                        spin_up=self.n_spin_up,
                                                        spin_down=self.n_spin_down
                                                    )
            np.savez(self.wf_filepath, energy=ground_state_energy, wave_function=np.array(ground_state_wfs))

        return ground_state_energy, ground_state_wfs

    def get_ground_state_properties(self):
        print('ground state energy: ', self.ground_state_energy)
        print('particle number: ', self.n_electrons)
        print('degeneracy: ', len(self.ground_state_wfs))
        print('')

    def calculate_fidelity(self, state):
        if len(self.ground_state_wfs) == 1:
            return np.abs(state.conj() @ self.ground_state_wfs[0]) ** 2
        
        projected_state = np.zeros_like(state)
        for ground_state_wf in self.ground_state_wfs:
            coeff = ground_state_wf.conj() @ state
            projected_state += coeff * ground_state_wf
        
        norm = np.linalg.norm(projected_state, ord=2)
        if norm < 1e-10:
            return 0.0
        
        projected_state = projected_state / norm
        return np.abs(state.conj() @ projected_state) ** 2

    def save_model(self):
        
        np.savez(self.model_filepath, **self.params, circuit_indices=np.array(self.selected_indices))

        # Convert FermionOperators to strings for JSON serialization
        save_results = self.results.copy()
        save_results['selected operators'] = [str(op) for op in self.results['selected operators']]

        with open(self.result_filepath , 'w') as file:
            json.dump(save_results, file)

    def load_model(self):
        
        if not os.path.exists(self.model_filepath):
            raise ValueError('Please check if the file ' + self.model_filepath + 'exists!')
        if not os.path.exists(self.result_filepath):
            raise ValueError('Please check if the file ' + self.result_filepath + 'exists!')
        
        loaded_data = np.load(self.model_filepath)
        self.params = {key: jnp.array(loaded_data[key]) for key in loaded_data.files if key != 'circuit_indices'}
        self.selected_indices = loaded_data['circuit_indices'].tolist()
        self.selected_gates = [self.gateOperatorPool[idx] for idx in self.selected_indices]
        
        with open(self.result_filepath, 'r') as file:
            self.results = json.load(file)
    
    def select_operator(self):

        dev = qml.device(self.device_name, wires=self.n_qubits)
        diff_method = 'adjoint' if self.device_name == 'lightning.gpu' else 'backprop'

        @qml.qnode(dev, interface='jax', diff_method=diff_method)
        def model(params_e, params_t):
            return self.circuit(params_e, params_t, mode='eval')

        # Compute gradients only with respect to 'e' parameters (argnum=0)
        grads_e_val = jax.grad(model, argnums=0)(self.params['e'], self.params['t'])
        grads_e = np.abs(np.array(grads_e_val))

        max_grad = np.max(grads_e)
        self.Ng = np.sum((grads_e >= max_grad * self.ratio) * (grads_e >= self.threshold1))
        selected_indices = np.argsort(grads_e)[::-1][:self.Ng].tolist()
        selected_operator = [self.fermionOperatorPool[index] for index in selected_indices]
        selected_gates = [self.gateOperatorPool[index] for index in selected_indices]
        max_grads = [grads_e[index] for index in selected_indices]

        return selected_operator, selected_gates, max_grads, selected_indices

    def circuit(self, params_e, params_t, mode='train'):

        # prepare non-interacting ground state in k-space
        for q in self.spin_up_indices + self.spin_down_indices:
            qml.PauliX(wires=q)

        # circuit selected by algorithm
        if mode in ['train', 'state', 'all', 'eval']:
            for i, gate in enumerate(self.selected_gates):
                gate(params_t[i])

        if mode == 'eval':
            for i, gate in enumerate(self.gateOperatorPool):
                gate(params_e[i])

        # apply the Fourier Transform back to the real space
        for i in range(self.n_qubits):
            qml.RZ(np.angle(self.diagonal[i]), wires=i)

        for parallel_ops in self.circuit_description:
            for op in parallel_ops:
                if op == 'pht':
                    qml.PauliX(wires=self.n_qubits-1)
                else:
                    i, j, theta, phi = op
                    qml.SingleExcitation(2 * theta, wires=[i, j])
                    qml.RZ(phi, wires=j)

        if mode == 'train':
            return qml.expval(self.qmlHamiltonian), qml.expval(self.qmlOperators['Sz']), qml.expval(self.qmlOperators['S^2'])
        elif mode == 'state':
            return qml.state()
        elif mode == 'eval':
            return qml.expval(self.qmlHamiltonian)
        elif mode == 'all':
            return qml.expval(self.qmlHamiltonian), qml.expval(self.qmlOperators['Sz']), qml.expval(self.qmlOperators['S^2']), qml.state()

    def _setup_qnodes(self, dev, diff_method):
        """Helper to setup QNodes based on the differentiation method."""
        if diff_method == 'backprop':
            @qml.qnode(dev, interface='jax')
            def full_qnode(p_e, p_t):
                return self.circuit(p_e, p_t, mode='all')

            # Use general metric_tensor for robustness
            mt_fn = qml.metric_tensor(full_qnode, approx="block-diag", argnums=1) if self.use_qng else None

            def step_fn(params):
                def loss_fn(p_t):
                    res = full_qnode(params['e'], p_t)
                    return res[0], (res[1], res[2], res[3])
                (loss, (Sz, S_square, state)), grads_t = jax.value_and_grad(loss_fn, has_aux=True)(params['t'])
                return loss, Sz, S_square, state, grads_t, mt_fn
        else:
            @qml.qnode(dev, interface='jax', diff_method='adjoint')
            def train_qnode(p_e, p_t):
                return self.circuit(p_e, p_t, mode='train')
            @qml.qnode(dev, interface='jax', diff_method=None)
            def state_qnode(p_e, p_t):
                return self.circuit(p_e, p_t, mode='state')
            
            # For lightning.gpu, we need a QNode without diff_method='adjoint' for the metric tensor
            @qml.qnode(dev, interface='jax')
            def mt_qnode(p_e, p_t):
                return self.circuit(p_e, p_t, mode='train')

            mt_fn = qml.metric_tensor(mt_qnode, approx="block-diag", argnums=1) if self.use_qng else None

            def step_fn(params):
                state = state_qnode(params['e'], params['t'])
                def loss_fn(p_t):
                    res = train_qnode(params['e'], p_t)
                    return res[0], (res[1], res[2])
                (loss, (Sz, S_square)), grads_t = jax.value_and_grad(loss_fn, has_aux=True)(params['t'])
                return loss, Sz, S_square, state, grads_t, mt_fn
        return step_fn

    def run(self):
        self.get_ground_state_properties()
        start_time = time.time()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        dev = qml.device(self.device_name, wires=self.n_qubits)
        diff_method = 'adjoint' if self.device_name == 'lightning.gpu' else 'backprop'

        while len(self.results['epoch loss']) < self.n_epoch:
            i_epoch = len(self.results['epoch loss'])
            print(f"\n{'='*20} Epoch {i_epoch + 1} {'='*20}")

            print(">>> Phase 1: Operator Selection")
            sel_ops, sel_gates, max_grads, sel_indices = self.select_operator()
            if not max_grads: break

            self.selected_gates += sel_gates
            self.selected_indices += sel_indices
            self.params['t'] = jnp.concatenate([self.params['t'], jnp.zeros(self.Ng)])
            self.results['selected operators'] += sel_ops
            self.results['n_params'].append(len(self.results['selected operators']))

            lr = float(np.linalg.norm(max_grads) / np.sqrt(self.Ng) * 0.05)
            optimizer = optax.adam(lr)
            opt_state = optimizer.init(self.params)

            print(f">>> Found {len(sel_ops)} new operators. Total parameters: {len(self.params['t'])}")
            print(f">>> Learning rate: {lr:.6f} | Use QNG: {self.use_qng}")
            print(">>> Phase 2: Parameter Optimization")

            step_fn = self._setup_qnodes(dev, diff_method)

            while True:
                loss, Sz, S_square, state, grads_t, mt_fn = step_fn(self.params)
                fidelity = self.calculate_fidelity(state)

                grads = {'e': jnp.zeros_like(self.params['e']), 't': grads_t}
                if self.use_qng:
                    mt = mt_fn(self.params['e'], self.params['t'])
                    grads['t'] = jnp.linalg.solve(mt + self.qng_reg * jnp.eye(mt.shape[0]), grads['t'])

                updates, opt_state = optimizer.update(grads, opt_state)
                self.params = optax.apply_updates(self.params, updates)

                # Logging
                iter_idx = len(self.results['iteration loss']) + 1
                log_str = (f"Iter: {iter_idx:4d} | Energy: {float(loss):.8f} | "
                           f"Grad Norm: {float(jnp.linalg.norm(grads['t'])):.6f} | "
                           f"Fidelity: {float(fidelity):.6f} | Sz: {float(Sz):.6f} | S^2: {float(S_square):.6f}")
                print(log_str)
                with open(self.log_filepath, 'a') as f: f.write(log_str + '\n')

                self.results['iteration loss'].append(float(loss))
                self.results['Sz'].append(float(Sz))
                self.results['S^2'].append(float(S_square))
                self.results['fidelity'].append(float(fidelity))

                if float(jnp.linalg.norm(grads['t'])) < self.threshold2: break

            self.results['epoch loss'].append(self.results['iteration loss'][-1])
            self.save_model()

            # Plotting
            for ax, data, label, color in zip([ax1, ax2], 
                                              [self.results['iteration loss'], self.results['epoch loss']], 
                                              ['iteration', 'epoch'], ['coral', 'yellowgreen']):
                ax.clear()
                ax.plot(np.arange(len(data))+1, data, color=color, marker='X', ls='--', label='ADAPT')
                ax.axhline(self.ground_state_energy, color='violet', label='ED')
                ax.set_xlabel(label); ax.set_ylabel('energy'); ax.legend(); ax.grid()
            plt.savefig(self.img_filepath)

        print('Total run time: ', time.time() - start_time)
        

if __name__ == '__main__':
    vqe = ADAPT(
        n_epoch=100,
        threshold1=1e-2,
        threshold2=1e-2,
        x_dimension=2,
        y_dimension=4,
        n_electrons=8,
        n_spin_up=4,
        n_spin_down=4,
        tunneling=1,
        coulomb=2,
        load_model=False,
        device='lightning.gpu',  # Uncomment to use cuQuantum and Adjoint Differentiation for 16+ qubits
        use_qng=True,
    )
    # vqe.get_ground_state_properties()
    vqe.run()