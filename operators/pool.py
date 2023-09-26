import openfermion as of
import numpy as np
from openfermion import (
    FermionOperator,
    hermitian_conjugated,
    normal_ordered,
    up_index, down_index
)

# SD = {
#   (p, q), 1 <= p <= q <= M 
#   (pq, rs), 1 <= p <= q <= M, 1 <= r <= s <= M
# }
def excitations(n_electrons,
                n_orbitals,
                delta_sz=0,
                generalized=True) -> list[list[int]]:
    
    n_spin_orbitals = n_orbitals * 2
    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(n_spin_orbitals)])

    singles = []

    endIndex = n_spin_orbitals if generalized else n_electrons
    for q in range(endIndex):

        startIndex = q + 1 if generalized else n_spin_orbitals
        for p in range(startIndex, n_spin_orbitals):
            
            if sz[p] - sz[q] == delta_sz:
                singles.append([q, p])

    doubles = []

    for s in range(endIndex-1):
        for r in range(s + 1, endIndex):

            startIndex = r + 1 if generalized else n_electrons
            for q in range(startIndex, n_spin_orbitals - 1):
                for p in range(q + 1, n_spin_orbitals):

                    if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz:
                        doubles.append([s, r, q, p])

    return singles, doubles

def spin_complemented_pool(n_electrons,
                           n_orbitals,
                           generalized=True) -> list[FermionOperator]:

    # Number of occupied orbitals should not be 1/2 the number of electrons (n_electrons). Hunds Rule will force high-spin states. (Need to discuss with Junze)
                             
    n_occ = n_electrons // 2
    n_vir = n_orbitals - n_occ
    pool = []
    
    # single excitation operator
    """
        τ_{p↑,q↑} = a†_{p↑} a_{q↑} - a†_{q↑} a_{p↑}
        τ_{p↓,q↓} = a†_{p↓} a_{q↓} - a†_{q↓} a_{p↓}
        A_{pq} = τ_{p↑,q↑} + τ_{p↓,q↓}
    
    """
    # occupied orbitals
    endIndex = n_orbitals if generalized else n_occ
    for q in range(endIndex):
        q_up = q * 2
        q_down = q * 2 + 1
        
        # virtual orbitals
        startIndex = q+1 if generalized else n_occ
        for p in range(startIndex, n_orbitals):
            p_up = p * 2
            p_down = p * 2 + 1
            
            tau_up = FermionOperator(f'{p_up}^ {q_up}') - FermionOperator(f'{q_up}^ {p_up}')
            tau_down = FermionOperator(f'{p_down}^ {q_down}') - FermionOperator(f'{q_down}^ {p_down}')
            op = tau_up + tau_down
            op = normal_ordered(op)
            
            if op.many_body_order() > 0:
                pool.append(op)

    # double excitation operator
    """
        τ_{p↑,q↑,r↑,s↑} = a†_{p↑} a†_{q↑} a_{r↑} a_{s↑} -  a†_{r↑} a†_{s↑} a_{p↑} a_{q↑}
        τ_{p↓,q↓,r↓,s↓} = a†_{p↓} a†_{q↓} a_{r↓} a_{s↓} -  a†_{r↓} a†_{s↓} a_{p↓} a_{q↓}
        A1_{pqrs} = τ_{p↑,q↑,r↑,s↑} + τ_{p↓,q↓,r↓,s↓}

        τ_{p↑,q↓,r↑,s↓} = a†_{p↑} a†_{q↓} a_{r↑} a_{s↓} -  a†_{r↑} a†_{s↓} a_{p↑} a_{q↓}        
        τ_{p↓,q↑,r↓,s↑} = a†_{p↓} a†_{q↑} a_{r↓} a_{s↑} -  a†_{r↓} a†_{s↑} a_{p↓} a_{q↑}
        A2_{pqrs} = τ_{p↑,q↓,r↑,s↓} + τ_{p↓,q↑,r↓,s↑}
    """
    # occupied orbitals
    endIndex = n_orbitals if generalized else n_occ
    for s in range(endIndex):
        s_up = s * 2
        s_down = s * 2 + 1

        for r in range(s, endIndex):
            r_up = r * 2
            r_down = r * 2 + 1

            # virtual orbitals
            startIndex = r+1 if generalized else n_occ
            for q in range(startIndex, n_orbitals):
                q_up = q * 2
                q_down = q * 2 + 1

                for p in range(q, n_orbitals):
                    s_up = s * 2
                    s_down = s * 2 + 1

                    op1 =  FermionOperator(f'{p_up}^ {q_up}^ {r_up} {s_up}')
                    op1 += FermionOperator(f'{p_down}^ {q_down}^ {r_down} {s_down}')
                    op1 -= hermitian_conjugated(op1)
                    op1 =  normal_ordered(op1)

                    op2 =  FermionOperator(f'{p_up}^ {q_down}^ {r_up} {s_down}')
                    op2 += FermionOperator(f'{p_down}^ {q_up}^ {r_down} {s_up}')
                    op2 -= hermitian_conjugated(op2)
                    op2 =  normal_ordered(op2)

                    if op1.many_body_order() > 0:
                        pool.append(op1)
                    
                    if op2.many_body_order() > 0:
                        pool.append(op2)

    return pool

def hubbard_interaction_pool(Nx, Ny):

    hubbard_interaction_channel = {
        'ZS channel': [],
        'ZS2 channel': [],
        # 'W channel': [],
        'BCS channel': [],
        # 'BCS2 channel': []
    }

    def tuple2index(ix, iy, spin):
        return 2 * (ix + iy * Nx) + spin 
    def index2tuple(index, spin=False):
        if spin:
            return ((index // 2) % Nx, (index // 2) // Nx, index % 2)
        else:
            return (index % Nx, index // Nx)

    n_sites = Nx * Ny

    for spin in (0, 1):
        for k1 in range(n_sites):
            for k2 in range(n_sites):
                for q in range(n_sites):
                    
                    kx1, ky1 = index2tuple(k1, spin=False)
                    kx2, ky2 = index2tuple(k2, spin=False)
                    qx, qy = index2tuple(q, spin=False)

                    # ZS channel
                    # c^\dagger_{k1+q, \sigma} c^\dagger_{k2-q, -\sigma} c_{k2, -\sigma} c_{k1, \sigma}
                    i1 = tuple2index((kx1+qx) % Nx, (ky1+qy) % Ny, spin)
                    i2 = tuple2index((kx2-qx) % Nx, (ky2-qy) % Ny, spin^1)
                    i3 = tuple2index(kx2, ky2, spin^1)
                    i4 = tuple2index(kx1, ky1, spin)
                    hubbard_interaction_channel['ZS channel'] += [
                        FermionOperator(f'{i1}^ {i2}^ {i3} {i4}', 1j) -
                        FermionOperator(f'{i3}^ {i4}^ {i1} {i2}', 1j) 
                    ]


                    # ZS2 channel
                    # c^\dagger_{k1+q, \sigma} c^\dagger_{k2-q, -\sigma} c_{k2, \sigma} c_{k1, -\sigma}
                    i1 = tuple2index((kx1+qx) % Nx, (ky1+qy) % Ny, spin)
                    i2 = tuple2index((kx2-qx) % Nx, (ky2-qy) % Ny, spin^1)
                    i3 = tuple2index(kx2, ky2, spin)
                    i4 = tuple2index(kx1, ky1, spin^1)
                    hubbard_interaction_channel['ZS2 channel'] += [
                        FermionOperator(f'{i1}^ {i2}^ {i3} {i4}', 1j) -
                        FermionOperator(f'{i3}^ {i4}^ {i1} {i2}', 1j)
                    ]


                    # BCS channel
                    # c^\dagger_{k1, \sigma} c^\dagger_{-k1+q, -\sigma} c_{-k2+q, -sigma} c_{k2, \sigma}
                    i1 = tuple2index(kx1, ky1, spin)
                    i2 = tuple2index((-kx1+qx) % Nx, (-ky1+qy) % Ny, spin^1)
                    i3 = tuple2index((-kx2+qx) % Nx, (-ky2+qy) % Ny, spin^1)
                    i4 = tuple2index(kx2, ky2, spin)
                    hubbard_interaction_channel['BCS channel'] += [
                        FermionOperator(f'{i1}^ {i2}^ {i3} {i4}', 1j) -
                        FermionOperator(f'{i3}^ {i4}^ {i1} {i2}', 1j)
                    ]


    return hubbard_interaction_channel

    