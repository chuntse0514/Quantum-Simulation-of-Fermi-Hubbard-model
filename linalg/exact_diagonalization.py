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


# def jw_get_ground_state_for_3x3(sparse_operator, particle_number, spin_up, spin_down):

#     def rotate_index(index, degree):
#         if degree == 90:
#             map_table = {0: 0,   1: 1,   2: 12,
#                          3: 13,  4: 6,   5: 7,
#                          6: 2,   7: 3,   8: 14,
#                          9: 15,  10: 8,  11: 9,
#                          12: 4,  13: 5,  14: 16,
#                          15: 17, 16: 10, 17: 11}
#         elif degree == 180:
#             map_table = {0: 0,   1: 1,   2: 4,
#                          3: 5,   4: 2,   5: 3,
#                          6: 12,  7: 13,  8: 16,
#                          9: 17,  10: 14, 11: 15,
#                          12: 6,  13: 7,  14: 10,
#                          15: 11, 16: 8,  17: 9}
#         return map_table[index]

#     def reflect_index(index, axis):
#         if axis == 'x':
#             map_table = {0: 0,   1: 1,   2: 2,
#                          3: 3,   4: 4,   5: 5,
#                          6: 12,  7: 13,  8: 14,
#                          9: 15,  10: 16, 11: 17,
#                          12: 6,  13: 7,  14: 8,
#                          15: 9,  16: 10, 17: 11}
#         elif axis == 'y':
#             map_table = {0: 0,   1: 1,   2: 4,
#                          3: 5,   4: 2,   5: 3,
#                          6: 6,   7: 7,   8: 10,
#                          9: 11,  10: 8,  11: 9,
#                          12: 12, 13: 13, 14: 16,
#                          15: 17, 16: 14, 17: 15}

#         return map_table[index]
    
#     def rotate_basis(basis_index, degree):
        
#         bin_index = bin(basis_index)[2:].zfill(18)
#         new_index = ['0'] * 18
#         for i, bit in enumerate(bin_index):
#             if bit == '1':
#                 new_index[rotate_index(i, degree)] = '1'
        
#         new_index = ''.join(new_index)
#         return int(new_index, 2)
    
#     def reflect_basis(basis_index, axis):

#         bin_index = bin(basis_index)[2:].zfill(18)
#         new_index = ['0'] * 18
#         for i, bit in enumerate(bin_index):
#             if bit == '1':
#                 new_index[reflect_index(i, axis)] = '1'
        
#         new_index = ''.join(new_index)
#         return int(new_index, 2)
        
#     def rotate(state, degree):
#         non_zero_indices = state.nonzero()[0]
#         rotated_state = np.zeros_like(state)

#         for index in non_zero_indices:
#             value = state[index]
#             rotated_state[rotate_basis(index, degree)] = value
        
#         return rotated_state

#     def reflect(state, axis):
#         non_zero_indices = state.nonzero()[0]
#         reflected_state = np.zeros_like(state)

#         for index in non_zero_indices:
#             value = state[index]
#             reflected_state[reflect_basis(index, axis)] = value

#         return reflected_state

#     n_qubits = int(np.log2(sparse_operator.shape[0]))
#     restricted_operator = jw_number_spin_restrict_operator(sparse_operator, particle_number, spin_up, spin_down, n_qubits)

#     if restricted_operator.shape[0] - 1 <= 1:
#         dense_restricted_operator = restricted_operator.toarray()
#         eigvals, eigvecs = np.linalg.eigh(dense_restricted_operator)
#     else:
#         eigvals, eigvecs = scipy.sparse.linalg.eigsh(restricted_operator,
#                                                      k=1,
#                                                      which='SA')
#     state = eigvecs[:, 0]
#     state_0 = np.zeros(2**n_qubits, dtype=complex)
#     state_0[jw_number_spin_indices(particle_number, spin_up, spin_down, n_qubits)] = state

#     state_1 = rotate(state_0, degree=90)

#     state_2 = (state_0 + state_1) / 2
#     state_3 = rotate(state_2, degree=180)

#     state_4 = (state_2 + state_3) / 2
#     print('s wave: ', np.linalg.norm(state_4, ord=2))
#     s_wave_state = state_4 / np.linalg.norm(state_4, ord=2)

#     state_4 = (state_2 - state_3) / 2
#     state_5 = reflect(state_4, axis='x')
#     state_6 = (state_4 + state_5) / 2
#     print('px wave: ', np.linalg.norm(state_6, ord=2))
#     px_wave_state = state_6 / np.linalg.norm(state_6, ord=2)
#     state_6 = (state_4 - state_5) / 2
#     print('py wave: ', np.linalg.norm(state_6, ord=2))
#     py_wave_state = state_6 / np.linalg.norm(state_6, ord=2)

#     state_1 = rotate(state_0, degree=180)
#     state_2 = (state_0 + state_1) / 2
#     state_3 = rotate(state_2, degree=90)
#     state_4 = (state_3 - state_2) / 2
#     d_wave_state = state_4 / np.linalg.norm(state_4, ord=2)
#     print('d wave: ', np.linalg.norm(state_4, ord=2))

#     print('s px: ', np.round(s_wave_state.conj() @ px_wave_state, decimals=6))
#     print('s py: ', np.round(s_wave_state.conj() @ py_wave_state))
#     print('s d: ', np.round(s_wave_state.conj() @ d_wave_state))
#     print('px py: ', np.round(px_wave_state.conj() @ py_wave_state))
#     print('px d: ', np.round(px_wave_state.conj() @ d_wave_state))
#     print('py d :', np.round(py_wave_state.conj() @ d_wave_state))

#     return eigvals[0], [s_wave_state, px_wave_state, py_wave_state, d_wave_state]

def jw_get_ground_state_for_3x3(sparse_operator, particle_number, spin_up, spin_down):

    def proj(v1, v2):
        coeff = (v2.conj() @ v1) / (v2.conj() @ v2)
        return coeff * v2   

    def gram_schmidt(V):
        U = []
        for v in V:
            temp_vec = v
            for u in U:
                temp_vec = temp_vec - proj(v, u)
            temp_vec = temp_vec / np.linalg.norm(temp_vec, ord=2)
            U.append(temp_vec)
        return U

    n_qubits = int(np.log2(sparse_operator.shape[0]))
    restricted_operator = jw_number_spin_restrict_operator(sparse_operator, particle_number, spin_up, spin_down, n_qubits)

    if restricted_operator.shape[0] - 1 <= 1:
        dense_restricted_operator = restricted_operator.toarray()
        eigvals, eigvecs = np.linalg.eigh(dense_restricted_operator)
    else:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(restricted_operator,
                                                     k=10,
                                                     which='SA')
    for eigval in eigvals:
        print(eigval)
    expanded_states = []
    for k in range(4):
        state = eigvecs[:, k]
        expanded_state = np.zeros(2**n_qubits, dtype=complex)
        expanded_state[jw_number_spin_indices(particle_number, spin_up, spin_down, n_qubits)] = state
        expanded_states.append(expanded_state)

    # print('01', expanded_states[0].conj() @ expanded_states[1])
    # print('02', expanded_states[0].conj() @ expanded_states[2])
    # print('03', expanded_states[0].conj() @ expanded_states[3])
    # print('12', expanded_states[1].conj() @ expanded_states[2])
    # print('13', expanded_states[1].conj() @ expanded_states[3])
    # print('23', expanded_states[2].conj() @ expanded_states[3])
    expanded_states = gram_schmidt(expanded_states)
    # print('01', expanded_states[0].conj() @ expanded_states[1])
    # print('02', expanded_states[0].conj() @ expanded_states[2])
    # print('03', expanded_states[0].conj() @ expanded_states[3])
    # print('12', expanded_states[1].conj() @ expanded_states[2])
    # print('13', expanded_states[1].conj() @ expanded_states[3])
    # print('23', expanded_states[2].conj() @ expanded_states[3])
    return eigvals[0], expanded_states

if __name__ == '__main__':
    from models.dha_for_3x3 import DHA

    vqe = DHA(
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

    gs_e, gs_wf = jw_get_ground_state_for_3x3(
        sparse_operator=get_sparse_operator(vqe.fermionHamiltonian),
        particle_number=vqe.n_electrons,
        spin_up=vqe.n_spin_up,
        spin_down=vqe.n_spin_down
    )

    print(gs_wf[0].conj() @ new_hamiltonian @ gs_wf[0])
    print(gs_wf[1].conj() @ new_hamiltonian @ gs_wf[1])
    print(gs_wf[2].conj() @ new_hamiltonian @ gs_wf[2])
    print(gs_wf[3].conj() @ new_hamiltonian @ gs_wf[3])

