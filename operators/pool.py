import openfermion as of
import numpy as np
from openfermion import FermionOperator, hermitian_conjugated, normal_ordered

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
        
        # virtual oribitals
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