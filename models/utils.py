import numpy as np
from openfermion import QubitOperator, up_index, down_index
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

def compile_hva_gate(x_dimension, y_dimension, periodic):

    def tuple2index(x, y, spin):
        return 2 * (x + y * x_dimension) + spin

    horizontal_gate_set = []
    vertical_gate_set = []

    # compile horizontal gate
    # for Nx = 2, only one parameter
    if x_dimension == 2:
        h_gates = []
        for y in range(y_dimension):
            h_gates += [
                (tuple2index(0, y, spin), tuple2index(1, y, spin))
                for spin in (0, 1)
            ]
        horizontal_gate_set.append(h_gates)
    
    # for Nx > 2 and Nx odd, there are 3 parameters
    elif periodic and x_dimension % 2 == 1:
        h_gates1 = []
        h_gates2 = []
        h_gates3 = []
        for y in range(y_dimension):
            h_gates1 += [
                (tuple2index(x, y, spin), tuple2index(x+1, y, spin))
                for x in range(x_dimension) if x % 2 == 0 and x+1 != x_dimension
                for spin in (0, 1)
            ]
            h_gates2 += [
                (tuple2index(x, y, spin), tuple2index(x+1, y, spin))
                for x in range(x_dimension) if x % 2 == 1 
                for spin in (0, 1)
            ]
            h_gates3 += [
                (tuple2index(0, y, spin), tuple2index(x_dimension-1, y, spin))
                for spin in (0, 1)
            ]
        horizontal_gate_set.append(h_gates1)
        horizontal_gate_set.append(h_gates2)
        horizontal_gate_set.append(h_gates3)

    # for other cases of Nx > 2, there are 2 parameters 
    else:
        h_gates1 = []
        h_gates2 = []

        # even and periodic
        if periodic:
            for y in range(y_dimension):
                h_gates1 += [
                    (tuple2index(x, y, spin), tuple2index(x+1, y, spin))
                    for x in range(x_dimension) if x % 2 == 0
                    for spin in (0, 1)
                ]
                h_gates2 += [
                    (tuple2index(x, y, spin), tuple2index(x+1, y, spin))
                    for x in range(x_dimension) if x % 2 == 1 and x+1 != x_dimension
                    for spin in (0, 1)
                ] + [
                    (tuple2index(0, y, spin), tuple2index(x_dimension-1, y, spin))
                    for spin in (0, 1)
                ]

        # not periodic
        else:
            for y in range(y_dimension):
                h_gates1 += [
                    (tuple2index(x, y, spin), tuple2index(x+1, y, spin))
                    for x in range(x_dimension) if x % 2 == 0 and x+1 != x_dimension
                    for spin in (0, 1)
                ]
                h_gates2 += [
                    (tuple2index(x, y, spin), tuple2index(x+1, y, spin))
                    for x in range(x_dimension) if x % 2 == 1 and x+1 != x_dimension
                    for spin in (0, 1)
                ]

        horizontal_gate_set.append(h_gates1)
        horizontal_gate_set.append(h_gates2)


    # compile vertical gate
    # for Ny = 2, only one parameter
    if y_dimension == 2:
        v_gates = []
        for x in range(x_dimension):
            v_gates += [
                (tuple2index(x, 0, spin), tuple2index(x, 1, spin))
                for spin in (0, 1)
            ]
        vertical_gate_set.append(v_gates)
    
    # for Ny > 2 and Ny odd, there are 3 parameters
    elif periodic and y_dimension % 2 == 1:
        v_gates1 = []
        v_gates2 = []
        v_gates3 = []
        for x in range(x_dimension):
            v_gates1 += [
                (tuple2index(x, y, spin), tuple2index(x, y+1, spin))
                for y in range(y_dimension) if y % 2 == 0 and y+1 != y_dimension
                for spin in (0, 1)
            ]
            v_gates2 += [
                (tuple2index(x, y, spin), tuple2index(x, y+1, spin))
                for y in range(y_dimension) if y % 2 == 1
                for spin in (0, 1)
            ]
            v_gates3 += [
                (tuple2index(x, 0, spin), tuple2index(x, y_dimension-1, spin))
                for spin in (0, 1)
            ]
        vertical_gate_set.append(v_gates1)
        vertical_gate_set.append(v_gates2)
        vertical_gate_set.append(v_gates3)

    # for other cases of Ny > 2, there are 2 parameters 
    else:
        v_gates1 = []
        v_gates2 = []

        # even and periodic
        if periodic:
            for x in range(x_dimension):
                v_gates1 += [
                    (tuple2index(x, y, spin), tuple2index(x, y+1, spin))
                    for y in range(y_dimension) if y % 2 == 0
                    for spin in (0, 1)
                ]
                v_gates2 += [
                    (tuple2index(x, y, spin), tuple2index(x, y+1, spin))
                    for y in range(y_dimension) if y % 2 == 1 and y+1 != y_dimension
                    for spin in (0, 1)
                ] + [
                    (tuple2index(x, 0, spin), tuple2index(x, y_dimension-1, spin))
                    for spin in (0, 1)
                ]

        # not periodic
        else:
            for x in range(x_dimension):
                v_gates1 += [
                    (tuple2index(x, y, spin), tuple2index(x, y+1, spin))
                    for y in range(y_dimension) if y % 2 == 0 and y+1 != y_dimension
                    for spin in (0, 1)
                ]
                v_gates2 += [
                    (tuple2index(x, y, spin), tuple2index(x, y+1, spin))
                    for y in range(y_dimension) if y % 2 == 1 and y+1 != y_dimension
                    for spin in (0, 1)
                ]

        vertical_gate_set.append(v_gates1)
        vertical_gate_set.append(v_gates2)

    return horizontal_gate_set, vertical_gate_set