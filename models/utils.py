import numpy as np
from openfermion import QubitOperator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter
import pennylane as qml
from functools import reduce

def QubitOperator_to_SparsePauliOp(qubitHamiltonian: QubitOperator, num_qubits) -> SparsePauliOp:

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

def QubitOperator_to_qmlHamiltonian(op: QubitOperator):

    pauliDict = {
        'X': qml.PauliX,
        'Y': qml.PauliY,
        'Z': qml.PauliZ
    }

    coeffs = []
    obs = []

    for pauliString, coeff in op.terms.items():

        if not pauliString:
            coeffs.append(coeff)
            obs.append(qml.Identity(wires=[0]))
            continue
        
        coeffs.append(coeff)
        pauliList = [pauliDict[pauli](index) for index, pauli in pauliString]
        obs.append(reduce(lambda a, b: a @ b, pauliList))

    return qml.Hamiltonian(coeffs, obs)

def PauliStringRotation(theta, pauliString: tuple[str, list[int]]):
    
    # Basis rotation
    for pauli, qindex in zip(*pauliString):
        if pauli == 'X':
            qml.RY(-np.pi / 2, wires=qindex)
        elif pauli == 'Y':
            qml.RX(np.pi / 2, wires=qindex)
    
    # CNOT layer
    for q, q_next in zip(pauliString[1][:-1], pauliString[1][1:]):
        qml.CNOT(wires=[q, q_next])
    
    # Z rotation
    qml.RZ(theta, pauliString[1][-1])

    # CNOT layer
    for q, q_next in zip(reversed(pauliString[1][:-1]), reversed(pauliString[1][1:])):
        qml.CNOT(wires=[q, q_next])

    # Basis rotation
    for pauli, qindex in zip(*pauliString):
        if pauli == 'X':
            qml.RY(np.pi / 2, wires=qindex)
        elif pauli == 'Y':
            qml.RX(-np.pi / 2, wires=qindex)

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
