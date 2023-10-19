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

def jw_get_ground_state(sparse_operator, particle_number, spin_up, spin_down):
    
    n_qubits = int(np.log2(sparse_operator.shape[0]))
    restricted_operator = jw_number_spin_restrict_operator(sparse_operator, particle_number, spin_up, spin_down, n_qubits)

    if restricted_operator.shape[0] - 1 <= 1:
        dense_restricted_operator = restricted_operator.toarray()
        eigvals, eigvecs = np.linalg.eigh(dense_restricted_operator)
    else:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(restricted_operator,
                                                     k=1,
                                                     which='SA')
    state = eigvecs[:, 0]
    expanded_state = np.zeros(2**n_qubits, dtype=complex)
    expanded_state[jw_number_spin_indices(particle_number, spin_up, spin_down, n_qubits)] = state

    return eigvals[0], expanded_state

def jw_get_ground_state_for_3x3(sparse_operator, particle_number, spin_up, spin_down):

    n_qubits = int(np.log2(sparse_operator.shape[0]))
    restricted_operator = jw_number_spin_restrict_operator(sparse_operator, particle_number, spin_up, spin_down, n_qubits)

    if restricted_operator.shape[0] - 1 <= 1:
        dense_restricted_operator = restricted_operator.toarray()
        eigvals, eigvecs = np.linalg.eigh(dense_restricted_operator)
    else:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(restricted_operator,
                                                     k=1,
                                                     which='SA')
    def rotate_index(index, degree):
        if degree == 90:
            map_table = {0: 0,   1: 1,   2: 12,
                         3: 13,  4: 6,   5: 7,
                         6: 2,   7: 3,   8: 14,
                         9: 15,  10: 8,  11: 9,
                         12: 4,  13: 5,  14: 16,
                         15: 17, 16: 10, 17: 11}
        elif degree == 180:
            map_table = {0: 0,   1: 1,   2: 4,
                         3: 5,   4: 2,   5: 3,
                         6: 12,  7: 13,  8: 16,
                         9: 17,  10: 14, 11: 15,
                         12: 6,  13: 7,  14: 10,
                         15: 11, 16: 8,  17: 9}
        return map_table[index]
    
    def reflect_index(index, axis):
        if axis == 'x':
            map_table = {0: 0, 1: 1}
    
    def rotate_basis(basis_index, degree):
        
        bin_index = bin(basis_index)[2:].zfill(18)
        new_index = ['0'] * 18
        for i, bit in enumerate(bin_index):
            if bit == '1':
                new_index[rotate_index(i, degree)] = '1'
        
        new_index = ''.join(new_index)
        return int(new_index, 2)
        
    def rotate(state, degree):
        non_zero_indices = state.nonzero()
        rotated_state = np.zeros_like(state)

        for index in non_zero_indices:
            value = state[index]
            rotated_state[rotate_basis(index, degree)] = value
        
        return rotated_state
        
    state_0 = eigvals[:, 0]
    state_1 = rotate(state_0, degree=90)

    state_2 = (state_0 + state_1) / 2
    state_3 = rotate(state_2, degree=180)

    state_4 = (state_2 + state_3) / 2

    psi_s = state_4 / np.linalg.norm(state_4, ord=2)



