import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import pennylane as qml
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import optax
import numpy as np
from scipy.sparse import csc_matrix
from openfermion import (
    QubitOperator,
    FermionOperator,
    fermi_hubbard,
    jordan_wigner,
    get_sparse_operator,
    givens_decomposition_square,
    jw_get_ground_state_at_particle_number,
    number_operator,
    up_index, down_index
)
import matplotlib.pyplot as plt
from .utils import (
    QubitOperator_to_qmlHamiltonian,
    PauliStringRotation
)
from linalg.exact_diagonalization import jw_get_ground_state
from operators.fourier import fourier_transform_matrix, fourier_transform
from operators.tools import get_interacting_term, get_quadratic_term
from operators.pool import hubbard_interaction_pool_simplified
from functools import reduce, partial
import os
import json
import time


def get_particle_number_operator(x_dimension, y_dimension, spinless=False):

    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    total_particle_operator = FermionOperator()

    for site in range(n_sites):

        if spinless:
            total_particle_operator += number_operator(n_sites, site, 1)
        else:
            total_particle_operator += number_operator(n_spin_orbitals, up_index(site), 1)
            total_particle_operator += number_operator(n_spin_orbitals, down_index(site), 1)

    return total_particle_operator

def get_total_spin(n_sites, spin_type):
    
    total_spin = FermionOperator()
    n_spin_orbitals = n_sites * 2

    for site in range(n_sites):
        if spin_type == 'spin-up':
            total_spin += number_operator(n_spin_orbitals, up_index(site), 1)
        elif spin_type == 'spin-down':
            total_spin += number_operator(n_spin_orbitals, down_index(site), 1)
        else:
            raise ValueError('spin_type must be either spin-up or spin-down')

    return total_spin

def get_spin_operators(n_sites, spin_type):

    Sx = FermionOperator()
    Sy = FermionOperator()
    Sz = FermionOperator()

    for site in range(n_sites):
        i_up = up_index(site)
        i_down = down_index(site)

        Sx += FermionOperator(f'{i_up}^ {i_down}', 0.5) + FermionOperator(f'{i_down}^ {i_up}', 0.5)
        Sy += FermionOperator(f'{i_up}^ {i_down}', -0.5j) - FermionOperator(f'{i_down}^ {i_up}', -0.5j)
        Sz += FermionOperator(f'{i_up}^ {i_up}', 0.5) - FermionOperator(f'{i_down}^ {i_down}', 0.5)

    if spin_type == 'Sx':
        return Sx
    elif spin_type == 'Sy':
        return Sy
    elif spin_type == 'Sz':
        return Sz
    elif spin_type == 'S^2':
        return Sx * Sx + Sy * Sy + Sz * Sz

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
    def __init__(self,
                 n_epoch: int,
                 threshold1: float,
                 threshold2: float,
                 x_dimension: int,
                 y_dimension: int,
                 n_electrons: int,
                 n_spin_up: int,
                 n_spin_down: int,
                 tunneling: float,
                 coulomb: float,
                 periodic=True,
                 spinless=False,
                 particle_hole_symmetry=False,
                 load_model=False
                 ):
        
        self.fermionOperatorPool = hubbard_interaction_pool_simplified(x_dimension, y_dimension)
        self.qubitOperatorPool = [jordan_wigner(generator) for generator in self.fermionOperatorPool]
        self.gateOperatorPool = [partial(Trotterize_generator, generator=generator) for generator in self.qubitOperatorPool]

        self.n_epoch = n_epoch
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.n_sites = x_dimension * y_dimension
        self.n_qubits = x_dimension * y_dimension * 2
        self.n_electrons = n_electrons
        self.n_spin_up = n_spin_up
        self.n_spin_down = n_spin_down
        
        self.ratio = 0.1
        # In JAX, we usually don't need to specify device manually like in Torch
        # unless we want to use specific ones.

        self.fermionHamiltonian = fermi_hubbard(x_dimension,
                                                y_dimension,
                                                tunneling,
                                                coulomb,
                                                periodic=periodic,
                                                spinless=spinless,
                                                particle_hole_symmetry=particle_hole_symmetry)
        self.qubitHamiltonian = jordan_wigner(self.fermionHamiltonian)
        self.quadratic_term = get_quadratic_term(self.fermionHamiltonian)
        self.interacting_term = get_interacting_term(self.fermionHamiltonian)
        self.qmlHamiltonian = QubitOperator_to_qmlHamiltonian(self.fermionHamiltonian)
        self.fermionOperators = {
            'spin up': get_total_spin(self.n_sites, spin_type='spin-up'),
            'spin down': get_total_spin(self.n_sites, spin_type='spin-down'),
            'Sx': get_spin_operators(self.n_sites, spin_type='Sx'),
            'Sy': get_spin_operators(self.n_sites, spin_type='Sy'),
            'Sz': get_spin_operators(self.n_sites, spin_type='Sz'),
            'S^2': get_spin_operators(self.n_sites, spin_type='S^2')
        }
        self.qmlOperators = {
            'spin up': QubitOperator_to_qmlHamiltonian(self.fermionOperators['spin up']),
            'spin down': QubitOperator_to_qmlHamiltonian(self.fermionOperators['spin down']),
            'Sx': QubitOperator_to_qmlHamiltonian(self.fermionOperators['Sx']),
            'Sy': QubitOperator_to_qmlHamiltonian(self.fermionOperators['Sy']),
            'Sz': QubitOperator_to_qmlHamiltonian(self.fermionOperators['Sz']),
            'S^2': QubitOperator_to_qmlHamiltonian(self.fermionOperators['S^2'])
        }
        self.FT_transformation_matrix = fourier_transform_matrix(x_dimension, y_dimension)
        self.decomposition, self.diagonal = givens_decomposition_square(self.FT_transformation_matrix)
        self.circuit_description = list(reversed(self.decomposition))
        self.k_quadratic_term = fourier_transform(self.quadratic_term, x_dimension, y_dimension)
        self.k_interacting_term = fourier_transform(self.interacting_term, x_dimension, y_dimension)
        self.spin_up_indices, self.spin_down_indices = get_non_interacting_ground_state_index(
                                                            self.k_quadratic_term,
                                                            self.n_qubits,
                                                            n_spin_up, n_spin_down
                                                       )
        print('spin up indices: ', self.spin_up_indices, '  ', 'spin down indices: ', self.spin_down_indices, '\n')
        self.img_filepath = f'./images/ADAPT-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}).pdf'
        self.wf_filepath = f'./results/ground_state_results/Hubbard-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}).npz'
        self.result_filepath = f'./results/vqe_results/ADAPT-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}).json'
        self.model_filepath = f'./results/saved_model/ADAPT-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}).npz'
        self.log_filepath = f'./results/vqe_logs/ADAPT-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}).log'
        self.ground_state_energy, self.ground_state_wfs = self.get_ground_state()
        
        if not load_model:
            if os.path.exists(self.log_filepath):
                os.remove(self.log_filepath)
        
        if load_model:
            self.load_model()
        else:
            self.params = {
                'e': jnp.zeros(len(self.gateOperatorPool)),
                't': jnp.array([])
            }
            self.selected_gates = []
            self.selected_indices = []
            self.results = {
                'epoch loss': [],
                'iteration loss': [],
                'Sz': [],
                'S^2': [],
                'fidelity': [],
                'n_params': [],
                'selected operators': []
            }

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
        
        dev = qml.device('default.qubit', wires=self.n_qubits)
        
        @qml.qnode(dev, interface='jax', diff_method='backprop')
        def model(params):
            return self.circuit(params, mode='eval')

        grads = jax.grad(model)(self.params)
        grads_e = np.abs(grads['e'])
        
        max_grad = np.max(grads_e)
        self.Ng = np.sum((grads_e >= max_grad * self.ratio) * (grads_e >= self.threshold1))
        selected_indices = np.argsort(grads_e)[::-1][:self.Ng].tolist()
        selected_operator = [self.fermionOperatorPool[index] for index in selected_indices]
        selected_gates = [self.gateOperatorPool[index] for index in selected_indices]
        max_grads = [grads_e[index] for index in selected_indices]

        return selected_operator, selected_gates, max_grads, selected_indices

    def circuit(self, params, mode='train'):
        
        # prepare non-interacting ground state in k-space
        for q in self.spin_up_indices + self.spin_down_indices:
            qml.PauliX(wires=q)
        
        # circuit selected by algorithm
        if mode == 'train' or mode == 'state':
            for i, gate in enumerate(self.selected_gates):
                gate(params['t'][i])
            
        elif mode == 'eval':
            for i, gate in enumerate(self.selected_gates):
                gate(params['t'][i])

            for i, gate in enumerate(self.gateOperatorPool):
                gate(params['e'][i])

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

    def run(self):
        
        self.get_ground_state_properties()

        start_time = time.time()
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        dev = qml.device('default.qubit', wires=self.n_qubits)

        i_epoch = len(self.results['epoch loss'])

        while i_epoch < self.n_epoch:
            
            print(f"\n{'='*20} Epoch {i_epoch + 1} {'='*20}")
            print(">>> Phase 1: Operator Selection")
            selected_operators, selected_gates, max_grads, selected_indices = self.select_operator()
            if len(max_grads) == 0:
                print('>>> Convergence criterion satisfied, ending optimization.')
                break
            
            self.selected_gates += selected_gates
            self.selected_indices += selected_indices
            self.params['t'] = jnp.concatenate([self.params['t'], jnp.zeros(self.Ng)])
            self.results['selected operators'] += selected_operators
            self.results['n_params'].append(len(self.results['selected operators']))
            lr = float(np.linalg.norm(max_grads) / np.sqrt(self.Ng) * 0.05)
            
            # Use optax optimizer
            optimizer = optax.adam(lr)
            opt_state = optimizer.init(self.params)

            print(f">>> Found {len(selected_operators)} new operators. Total parameters: {len(self.params['t'])}")
            print(f">>> Learning rate: {lr:.6f}")
            print(">>> Phase 2: Parameter Optimization")
            
            # Define QNodes once per epoch since the circuit structure is now fixed for this epoch
            @jax.jit
            @qml.qnode(dev, interface='jax', diff_method='backprop')
            def train_circuit(p):
                return self.circuit(p, mode='train')
            
            @jax.jit
            @qml.qnode(dev, interface='jax', diff_method='backprop')
            def state_circuit(p):
                return self.circuit(p, mode='state')
            
            def loss_fn(p):
                res = train_circuit(p)
                return res[0], (res[1], res[2])

            def train_step(params, opt_state):
                (loss, (Sz, S_square)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss, Sz, S_square, grads

            while True:
                
                state = state_circuit(self.params)
                fidelity = self.calculate_fidelity(state)
                
                self.params, opt_state, loss, Sz, S_square, grads = train_step(self.params, opt_state)

                self.results['iteration loss'].append(float(loss))
                self.results['Sz'].append(float(Sz))
                self.results['S^2'].append(float(S_square))
                self.results['fidelity'].append(float(fidelity))

                # Calculate grad norm for 't' parameters
                grad_norm = float(jnp.linalg.norm(grads['t']))

                log_str = (f"Iter: {len(self.results['iteration loss']):4d} | "
                           f"Energy: {float(loss):.8f} | "
                           f"Grad Norm: {grad_norm:.6f} | "
                           f"Fidelity: {float(fidelity):.6f} | "
                           f"Sz: {float(Sz):.6f} | "
                           f"S^2: {float(S_square):.6f}")
                print(log_str)
                
                with open(self.log_filepath, 'a') as f:
                    f.write(log_str + '\n')

                if grad_norm < self.threshold2:
                    print(f">>> Optimization for Epoch {i_epoch + 1} finished (Grad Norm < {self.threshold2})")
                    break
            
            self.results['epoch loss'].append(self.results['iteration loss'][-1])
            i_epoch += 1
            print('')

            #############################
            
            self.save_model()

            ax1.clear()
            length = len(self.results['iteration loss'])
            ax1.plot(np.arange(length)+1, self.results['iteration loss'], color='coral', marker='X', ls='--', label='ADAPT')
            ax1.plot(np.arange(length)+1, np.full(length, self.ground_state_energy), color='violet', label='ED')
            ax1.set_xlabel('iteration')
            ax1.set_ylabel('energy')
            ax1.legend()
            ax1.grid()
            
            ax2.clear()
            length = len(self.results['epoch loss'])
            ax2.plot(np.arange(length)+1, self.results['epoch loss'], color='yellowgreen', marker='X', ls='--', label='ADAPT')
            ax2.plot(np.arange(length)+1, np.full(length, self.ground_state_energy), color='violet', label='ED')
            ax2.set_xlabel('epoch')
            ax2.set_ylabel('energy')
            ax2.legend()
            ax2.grid()

            plt.savefig(self.img_filepath)
        
        end_time = time.time()

        print('total run time: ', end_time - start_time)
        

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
        load_model=False
    )
    # vqe.get_ground_state_properties()
    vqe.run()