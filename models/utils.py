import numpy as np
from openfermion import QubitOperator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter

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

def processPauliString(operator: SparsePauliOp):

    newPauliStrings = []
    coeffs = []

    for pauliString, coeff in operator.to_list():

        processedString = []
        qIndex = []
        coeffs.append((coeff * 2j).real)

        for index, pauli in enumerate(pauliString[::-1]):
            
            if pauli != 'I':
                processedString.append(pauli)
                qIndex.append(index)
                
        newPauliStrings.append((processedString, qIndex))

    return newPauliStrings, coeffs

def exponentialPauliString(theta: Parameter, pauliString: tuple[list[str], list[int]], coeff: float):

    # generate the name of the pauli rotation gate
    name = '$e^{i ' + theta.name[1:-1] + ' '
    for pauli, index in zip(*pauliString):
        name += pauli + '_' + str(index)
    name += ' / 2}$'

    qc = QuantumCircuit(len(pauliString[0]), name=name)

    # Apply the basis rotation unitary
    for index, pauli in enumerate(pauliString[0]):
        if pauli == 'X':
            qc.ry(-np.pi / 2, index)
        elif pauli == 'Y':
            qc.rx(np.pi / 2, index)

    # Apply the CNOT gates to compute the parity
    for index in range(len(pauliString[0])-1):
        qc.cx(index, index+1)

    # Apply the parameterized Rz gate
    qc.rz(coeff * theta, len(pauliString[0])-1)        

    # Apply the CNOT gates to uncompute the parity
    for index in reversed(range(len(pauliString[0])-1)):
        qc.cx(index, index+1)

        # Apply the basis rotation unitary
    for index, pauli in enumerate(pauliString[0]):
        if pauli == 'X':
            qc.ry(np.pi / 2, index)
        elif pauli == 'Y':
            qc.rx(-np.pi / 2, index)

    qc_inst = qc.to_instruction()

    return qc_inst
