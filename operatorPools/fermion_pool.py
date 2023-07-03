import openfermion as of
from openfermion import FermionOperator, hermitian_conjugated, normal_ordered

class Fermionic_Pool:
    
    def __init__(self, n_electrons, n_orbitals):
        self.n_electrons = n_electrons
        self.n_orbitals = n_orbitals
        self.n_occ = n_electrons // 2
        self.n_vir = n_orbitals - self.n_occ

    def spin_complemented_gsd_excitation(self, generalized=True):
        
        ops = []
        start_id = 0 if generalized else self.n_occ
        end_id = self.n_orbitals if generalized else self.n_occ
        
        # occupied orbitals
        for p in range(end_id):
            p_up = p * 2
            p_down = p * 2 + 1
            
            # virtual oribitals
            for q in range(p, self.n_orbitals):
                q_up = q * 2
                q_down = q * 2 + 1
                
                op =  FermionOperator(f'{q_up}^ {p_up}')
                op += FermionOperator(f'{q_down}^ {p_down}')
                op -= hermitian_conjugated(op) 
                op = normal_ordered(op)
                
                if op.many_body_order() > 0:
                    ops.append(op)

        # occupied orbitals
        for p in range(end_id):
            p_up = p * 2
            p_down = p * 2 + 1

            for q in range(p, end_id):
                q_up = q * 2
                q_down = q * 2 + 1

                # virtual orbitals
                for r in range(start_id, self.n_orbitals):
                    r_up = r * 2
                    r_down = r * 2 + 1

                    for s in range(r, self.n_orbitals):
                        s_up = s * 2
                        s_down = s * 2 + 1

                        if not (r >= p and s >= q):
                            continue

                        op1 =  FermionOperator(f'{r_up}^ {s_up}^ {p_up} {q_up}')
                        op1 += FermionOperator(f'{r_down}^ {s_down}^ {p_down} {q_down}')
                        op1 -= hermitian_conjugated(op1)
                        op1 =  normal_ordered(op1)

                        op2 =  FermionOperator(f'{r_up}^ {s_down}^ {p_up} {q_down}')
                        op2 += FermionOperator(f'{r_down}^ {s_up}^ {p_down} {q_up}')
                        op2 -= hermitian_conjugated(op2)
                        op2 =  normal_ordered(op2)

                        op3 =  FermionOperator(f'{r_up}^ {s_down}^ {q_up} {p_down}')
                        op3 += FermionOperator(f'{r_down}^ {s_up}^ {q_down} {p_up}')
                        op3 -= hermitian_conjugated(op3)
                        op3 =  normal_ordered(op3)

                        if op1.many_body_order() > 0:
                            ops.append(op1)
                        
                        if op2.many_body_order() > 0:
                            ops.append(op2)

                        if op3.many_body_order() > 0:
                            ops.append(op3)

        return ops
    
    def spin_adapted_sd_excitation(self):
        pass


# def test_pool():
#     fp = Fermionic_Pool(4, 6)
#     pool = fp.spin_complemented_gsd_excitation()
#     # print(pool[0].many_body_order())
#     print(len(pool))

# test_pool()



