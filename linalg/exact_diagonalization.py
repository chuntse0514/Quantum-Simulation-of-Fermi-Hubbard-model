import openfermion
from openfermion import (
    FermionOperator,
    QubitOperator,
    get_sparse_operator,
)
import itertools
import numpy as np
import scipy

def jw_number_spin_indices(n_electrons, spin_up, spin_down, n_qubits):

    if spin_up + spin_down != n_electrons:
        raise ValueError('spin up plus spin down must equal to n_electrons!')

    occupations = itertools.combinations(range(n_qubits), n_electrons)
    new_occupations = []
    for occ in occupations:
        num_spin_up = sum([occ_index % 2 == 0 for occ_index in occ])
        if num_spin_up == spin_up:
            new_occupations.append(occ)
            
    indices = [sum([2**(n_qubits-n-1) for n in occupation]) for occupation in reversed(new_occupations)]
    return indices
            
def jw_number_spin_restrict_operator(operator, n_electrons, spin_up, spin_down, n_qubits=None):
    
    if n_qubits is None:
        n_qubits = int(np.log2(operator.shape[0]))

    select_indices = jw_number_spin_indices(n_electrons, spin_up, spin_down, n_qubits)
    return operator[np.ix_(select_indices, select_indices)]

def proj(v1, v2):
    coeff = (v2.conj() @ v1) / (v2.conj() @ v2)
    return coeff * v2   

def gram_schmidt(V):
    U = []
    for v in V:
        temp_vec = v
        for u in U:
            temp_vec = temp_vec - proj(v, u)
        norm = np.linalg.norm(temp_vec, ord=2)
        if norm > 1e-10:
            temp_vec = temp_vec / norm
            U.append(temp_vec)
    return U

def jw_get_ground_state(sparse_operator, particle_number, spin_up, spin_down, tol=1e-6):
    
    n_qubits = int(np.log2(sparse_operator.shape[0]))
    restricted_operator = jw_number_spin_restrict_operator(sparse_operator, particle_number, spin_up, spin_down, n_qubits)

    # Solve for more eigenvalues to detect degeneracy
    k_to_solve = min(restricted_operator.shape[0] - 1, 10)
    
    if k_to_solve <= 1:
        dense_restricted_operator = restricted_operator.toarray()
        eigvals, eigvecs = np.linalg.eigh(dense_restricted_operator)
    else:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(restricted_operator,
                                                     k=k_to_solve,
                                                     which='SA')
        
    # Detect degeneracy
    min_eigval = eigvals[0]
    degenerate_indices = np.where(np.abs(eigvals - min_eigval) < tol)[0]
    
    expanded_states = []
    for i in degenerate_indices:
        state = eigvecs[:, i]
        expanded_state = np.zeros(2**n_qubits, dtype=complex)
        expanded_state[jw_number_spin_indices(particle_number, spin_up, spin_down, n_qubits)] = state
        expanded_states.append(expanded_state)
    
    if len(expanded_states) > 1:
        expanded_states = gram_schmidt(expanded_states)

    return min_eigval, expanded_states


if __name__ == '__main__':
    from models.adapt_vqe import ADAPT

    vqe = ADAPT(
        n_epoch=100,
        threshold1=1e-2,
        threshold2=1e-2,
        x_dimension=3,
        y_dimension=3,
        n_electrons=9,
        n_spin_up=5,
        n_spin_down=4,
        tunneling=1,
        coulomb=6,
        # load_model=True
    )

    new_hamiltonian = get_sparse_operator(vqe.fermionHamiltonian)

    gs_e, gs_wfs = jw_get_ground_state(
        sparse_operator=get_sparse_operator(vqe.fermionHamiltonian),
        particle_number=vqe.n_electrons,
        spin_up=vqe.n_spin_up,
        spin_down=vqe.n_spin_down
    )

    print(f"Ground state energy: {gs_e}")
    print(f"Degeneracy: {len(gs_wfs)}")
    for i, wf in enumerate(gs_wfs):
        print(f"State {i} energy expectation: {wf.conj() @ new_hamiltonian @ wf}")

