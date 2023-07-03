import numpy as np
from molecules import H2
from openfermion import (
    QubitOperator,
    get_sparse_operator, 
    jordan_wigner
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

def print_example_operator():
    molecule = H2(r=1.0)
    fermionicHamiltonian = molecule.get_molecular_hamiltonian()
    qubitHamiltonian = jordan_wigner(fermionicHamiltonian)
    print('******openfermion.QubitOperator******\n', qubitHamiltonian)
    print('******QubitOperator.terms******\n', qubitHamiltonian.terms.items())

# print_example_operator()

def QubitOperator_to_SparsePauliOp(qubitHamiltonian: QubitOperator, num_qubits) -> SparsePauliOp:
    """
    Convert the QubitOperator to the SparsePauliOp

    Args:
        qubitHamiltonian (openfermion.QubitOperator)

    Returns:
        sparseHamiltonian (qiskit.quantum_info.SparsePauliOp)
    """

    # This link demostrates how to construct a SparsePauliOp
    # https://qiskit.org/documentation/stubs/qiskit.quantum_info.SparsePauliOp.from_sparse_list.html#qiskit.quantum_info.SparsePauliOp.from_sparse_list
    
    sparseList = []

    for pauliString, coeff in qubitHamiltonian.terms.items():
        
        convertedString = ""
        indicies = []

        for index, pauli in pauliString:
            convertedString += pauli
            indicies.append(index)

        sparseList.append((convertedString, indicies, coeff))

    sparseHamiltonian = SparsePauliOp.from_sparse_list(sparseList, num_qubits=num_qubits)

    return sparseHamiltonian

def prepare_HF_state(n_electrons, n_qubits) -> QuantumCircuit:
    
    hf_circ = QuantumCircuit(n_qubits, name='hf circ')
    for i in range(n_electrons):
        hf_circ.x(i)
    return hf_circ