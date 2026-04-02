import pennylane as qml
import jax
import jax.numpy as jnp
import optax
import numpy as np
from openfermion import (
    FermionOperator,
    QubitOperator,
    fermi_hubbard,
    jordan_wigner,
    number_operator,
    get_sparse_operator,
    givens_decomposition_square,
)
from openfermion.utils.indexing import down_index, up_index
import matplotlib.pyplot as plt
from .utils import (
    QubitOperator_to_qmlHamiltonian,
    PauliStringRotation,
    get_hva_commuting_hopping_terms
)
from linalg.exact_diagonalization import jw_get_ground_state
from operators.fourier import fourier_transform_matrix, fourier_transform
from operators.tools import get_quadratic_term, get_interacting_term
from functools import reduce
import os
import json
import numpy as np
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

def get_total_spin(x_dimension, y_dimension, spin_type):
    
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    total_spin = FermionOperator()

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

class HVA:
    def __init__(self,
                 n_epoch: int,
                 reps: int,
                 lr: float,
                 threshold: float,
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
        
        self.n_epoch = n_epoch
        self.reps = reps
        self.lr = lr
        self.threshold = threshold
        self.n_sites = x_dimension * y_dimension
        self.n_qubits = x_dimension * y_dimension * 2
        self.n_electrons = n_electrons
        self.n_spin_up = n_spin_up
        self.n_spin_down = n_spin_down

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
            'hopping': get_quadratic_term(self.fermionHamiltonian),
            'coulomb': get_interacting_term(self.fermionHamiltonian),
            'particle number': get_particle_number_operator(x_dimension, y_dimension, spinless),
            'spin up': get_total_spin(x_dimension, y_dimension, 'spin-up'),
            'spin down': get_total_spin(x_dimension, y_dimension, 'spin-down'),
            'Sx': get_spin_operators(self.n_sites, spin_type='Sx'),
            'Sy': get_spin_operators(self.n_sites, spin_type='Sy'),
            'Sz': get_spin_operators(self.n_sites, spin_type='Sz'),
            'S^2': get_spin_operators(self.n_sites, spin_type='S^2')
        }
        _h, _v = get_hva_commuting_hopping_terms(x_dimension, y_dimension, periodic)
        self.Nh = len(_h)
        self.Nv = len(_v)
        self.hvaGenerators = {
            'horizontal': [jordan_wigner(generator) for generator in _h],
            'vertical': [jordan_wigner(generator) for generator in _v],
            'coulomb': jordan_wigner(self.fermionOperators['coulomb'])
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
        self.spin_up_indices, self.spin_down_indices = get_non_interacting_ground_state_index(
                                                            self.k_quadratic_term,
                                                            self.n_qubits,
                                                            n_spin_up, n_spin_down
                                                       )
        print('spin up indices: ', self.spin_up_indices, '  ', 'spin down indices: ', self.spin_down_indices, '\n')
        

        self.img_filepath = f'./images/HVA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}, reps={reps}).pdf'
        self.wf_filepath = f'./results/ground_state_results/Hubbard-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}).npz'
        self.result_filepath = f'./results/vqe_results/HVA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}, reps={reps}).json'
        self.model_filepath = f'./results/saved_model/HVA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}, reps={reps}).npz'
        self.log_filepath = f'./results/vqe_logs/HVA-{x_dimension}x{y_dimension} (t={tunneling}, U={coulomb}, n_electrons={n_electrons}, up={n_spin_up}, down={n_spin_down}, reps={reps}).log'
        self.ground_state_energy, self.ground_state_wfs = self.get_ground_state()
        
        if not load_model:
            if os.path.exists(self.log_filepath):
                os.remove(self.log_filepath)

        if load_model:
            self.load_model()
        else:
            self.params = {
                'theta_U': jnp.zeros(reps+1),
                'theta_v': jnp.zeros(reps * self.Nv),
                'theta_h': jnp.zeros(reps * self.Nh),
            }

            self.results = {
                'loss': [],
                'Sz': [],
                'S^2': [],
                'fidelity': [],
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
    
    def save_model(self):
        
        np.savez(self.model_filepath, **self.params)

        with open(self.result_filepath , 'w') as file:
            json.dump(self.results, file)

    def load_model(self):
        
        if not os.path.exists(self.model_filepath):
            raise ValueError('Please check if the file ' + self.model_filepath + 'exists!')
        if not os.path.exists(self.result_filepath):
            raise ValueError('Please check if the file ' + self.result_filepath + 'exists!')
        
        loaded_params = np.load(self.model_filepath)
        self.params = {key: jnp.array(loaded_params[key]) for key in loaded_params.files}
        
        with open(self.result_filepath, 'r') as file:
            self.results = json.load(file)
    
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

    def circuit(self, params, mode='train'):
        
        theta_U = params['theta_U']
        theta_h = params['theta_h']
        theta_v = params['theta_v']

        # prepare non-interacting ground state
        for q in self.spin_up_indices + self.spin_down_indices:
            qml.PauliX(wires=q)

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
        
        # ansatz circuit
        for rep in range(self.reps):
            Trotterize_generator(theta_U[rep], self.hvaGenerators['coulomb'])
            for i in range(self.Nv):
                Trotterize_generator(theta_v[rep * self.Nv + i], self.hvaGenerators['vertical'][i])
            for i in range(self.Nh):
                Trotterize_generator(theta_h[rep * self.Nh + i], self.hvaGenerators['horizontal'][i])
        Trotterize_generator(theta_U[self.reps], self.hvaGenerators['coulomb'])

        if mode == 'train':
            return qml.expval(self.qmlHamiltonian), qml.expval(self.qmlOperators['Sz']), qml.expval(self.qmlOperators['S^2'])
        elif mode == 'state':
            return qml.state()

    def run(self):
        
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        dev = qml.device('default.qubit', wires=self.n_qubits)
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(self.params)
        
        # Define QNodes once since the HVA circuit structure is fixed from the start
        @qml.qnode(dev, interface='jax')
        def train_circuit(p):
            return self.circuit(p, mode='train')
        
        @qml.qnode(dev, interface='jax')
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

        i_epoch = len(self.results['loss'])

        print(f"\n{'='*20} HVA Optimization Starting {'='*20}")
        print(f">>> Repetitions: {self.reps} | Learning rate: {self.lr}")

        while i_epoch < self.n_epoch:

            state = state_circuit(self.params)
            fidelity = self.calculate_fidelity(state)
            
            self.params, opt_state, loss, Sz, S_square, grads = train_step(self.params, opt_state)

            self.results['loss'].append(float(loss))
            self.results['Sz'].append(float(Sz))
            self.results['S^2'].append(float(S_square))
            self.results['fidelity'].append(float(fidelity))

            grad_norm = float(jnp.linalg.norm(jnp.concatenate([jax.tree_util.tree_flatten(grads)[0][i].flatten() for i in range(len(jax.tree_util.tree_flatten(grads)[0]))])))
            
            log_str = (f"Epoch: {i_epoch+1:4d} | "
                       f"Energy: {float(loss):.8f} | "
                       f"Grad Norm: {grad_norm:.6f} | "
                       f"Fidelity: {float(fidelity):.6f} | "
                       f"Sz: {float(Sz):.6f} | "
                       f"S^2: {float(S_square):.6f}")
            print(log_str)
            
            with open(self.log_filepath, 'a') as f:
                f.write(log_str + '\n')
            
            ax1.clear()
            ax1.plot(np.arange(i_epoch+1)+1, self.results['loss'], marker='X', color='r', label='HVA')
            ax1.plot(np.arange(i_epoch+1)+1, np.full(i_epoch+1, self.ground_state_energy), ls='-', color='g', label='ED')
            ax1.set_xlabel('epochs')
            ax1.set_ylabel('energy')
            ax1.legend()
            ax1.grid()

            ax2.clear()
            ax2.plot(np.arange(i_epoch+1)+1, self.results['fidelity'], marker='X', ls=':', color='coral')
            ax2.set_xlabel('epochs')
            ax2.set_ylabel('fidelity')
            ax2.grid()

            plt.savefig(self.img_filepath)

            if (i_epoch + 1) % 10 == 0:
                self.save_model()

            i_epoch += 1

        self.save_model()
    
if __name__ == '__main__':
    vqe = HVA(
        n_epoch=1000,
        reps=10,
        lr=1e-2,
        threshold=1e-2,
        x_dimension=2,
        y_dimension=2,
        n_electrons=4,
        n_spin_up=2,
        n_spin_down=2,
        tunneling=1,
        coulomb=6,
        periodic=True,
        spinless=False,
        particle_hole_symmetry=False,
        load_model=False
    )
    
    vqe.run()

    
