from openfermion import FermionOperator

def get_quadratic_term(operator: FermionOperator):

    quadratic_term = FermionOperator()

    for term in operator.get_operators():
        if term.many_body_order() == 2:
            quadratic_term += term
    
    quadratic_term.compress()
    return quadratic_term

def get_interacting_term(operator: FermionOperator):

    interacting_term = FermionOperator()

    for term in operator.get_operators():
        if term.many_body_order() > 2:
            interacting_term += term

    interacting_term.compress()
    return interacting_term