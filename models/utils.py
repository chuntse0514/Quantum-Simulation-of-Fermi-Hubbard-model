import numpy as np
from openfermion import (
    QubitOperator, FermionOperator, up_index, down_index, jordan_wigner
)
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

def QubitOperator_to_qmlHamiltonian(op: QubitOperator, mapper=jordan_wigner):

    if isinstance(op, FermionOperator):
        op = mapper(op)

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

def compile_hva_hopping_indices(x_dimension, y_dimension, periodic):

    def tuple2index(x, y, spin):
        return 2 * (x + y * x_dimension) + spin

    horizontal_set = []
    vertical_set = []

    # compile horizontal gate
    # for Nx = 2, only one parameter
    if x_dimension == 2:
        h_terms = []
        for y in range(y_dimension):
            h_terms += [
                (tuple2index(0, y, spin), tuple2index(1, y, spin))
                for spin in (0, 1)
            ]
        horizontal_set.append(h_terms)
    
    # for Nx > 2 and Nx odd, there are 3 parameters
    elif periodic and x_dimension % 2 == 1:
        h_terms1 = []
        h_terms2 = []
        h_terms3 = []
        for y in range(y_dimension):
            h_terms1 += [
                (tuple2index(x, y, spin), tuple2index(x+1, y, spin))
                for x in range(x_dimension) if x % 2 == 0 and x+1 != x_dimension
                for spin in (0, 1)
            ]
            h_terms2 += [
                (tuple2index(x, y, spin), tuple2index(x+1, y, spin))
                for x in range(x_dimension) if x % 2 == 1 
                for spin in (0, 1)
            ]
            h_terms3 += [
                (tuple2index(0, y, spin), tuple2index(x_dimension-1, y, spin))
                for spin in (0, 1)
            ]
        horizontal_set.append(h_terms1)
        horizontal_set.append(h_terms2)
        horizontal_set.append(h_terms3)

    # for other cases of Nx > 2, there are 2 parameters 
    else:
        h_terms1 = []
        h_terms2 = []

        # even and periodic
        if periodic:
            for y in range(y_dimension):
                h_terms1 += [
                    (tuple2index(x, y, spin), tuple2index(x+1, y, spin))
                    for x in range(x_dimension) if x % 2 == 0
                    for spin in (0, 1)
                ]
                h_terms2 += [
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
                h_terms1 += [
                    (tuple2index(x, y, spin), tuple2index(x+1, y, spin))
                    for x in range(x_dimension) if x % 2 == 0 and x+1 != x_dimension
                    for spin in (0, 1)
                ]
                h_terms2 += [
                    (tuple2index(x, y, spin), tuple2index(x+1, y, spin))
                    for x in range(x_dimension) if x % 2 == 1 and x+1 != x_dimension
                    for spin in (0, 1)
                ]

        horizontal_set.append(h_terms1)
        horizontal_set.append(h_terms2)


    # compile vertical gate
    # for Ny = 2, only one parameter
    if y_dimension == 2:
        v_terms = []
        for x in range(x_dimension):
            v_terms += [
                (tuple2index(x, 0, spin), tuple2index(x, 1, spin))
                for spin in (0, 1)
            ]
        vertical_set.append(v_terms)
    
    # for Ny > 2 and Ny odd, there are 3 parameters
    elif periodic and y_dimension % 2 == 1:
        v_terms1 = []
        v_terms2 = []
        v_terms3 = []
        for x in range(x_dimension):
            v_terms1 += [
                (tuple2index(x, y, spin), tuple2index(x, y+1, spin))
                for y in range(y_dimension) if y % 2 == 0 and y+1 != y_dimension
                for spin in (0, 1)
            ]
            v_terms2 += [
                (tuple2index(x, y, spin), tuple2index(x, y+1, spin))
                for y in range(y_dimension) if y % 2 == 1
                for spin in (0, 1)
            ]
            v_terms3 += [
                (tuple2index(x, 0, spin), tuple2index(x, y_dimension-1, spin))
                for spin in (0, 1)
            ]
        vertical_set.append(v_terms1)
        vertical_set.append(v_terms2)
        vertical_set.append(v_terms3)

    # for other cases of Ny > 2, there are 2 parameters 
    else:
        v_terms1 = []
        v_terms2 = []

        # even and periodic
        if periodic:
            for x in range(x_dimension):
                v_terms1 += [
                    (tuple2index(x, y, spin), tuple2index(x, y+1, spin))
                    for y in range(y_dimension) if y % 2 == 0
                    for spin in (0, 1)
                ]
                v_terms2 += [
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
                v_terms1 += [
                    (tuple2index(x, y, spin), tuple2index(x, y+1, spin))
                    for y in range(y_dimension) if y % 2 == 0 and y+1 != y_dimension
                    for spin in (0, 1)
                ]
                v_terms2 += [
                    (tuple2index(x, y, spin), tuple2index(x, y+1, spin))
                    for y in range(y_dimension) if y % 2 == 1 and y+1 != y_dimension
                    for spin in (0, 1)
                ]

        vertical_set.append(v_terms1)
        vertical_set.append(v_terms2)

    return horizontal_set, vertical_set

def get_hva_commuting_hopping_terms(x_dimesion, y_dimension, periodic):
    
    horizontal_set, vertical_set = compile_hva_hopping_indices(x_dimesion,
                                                               y_dimension,
                                                               periodic)
    
    horizontal_op_set = []
    vertical_op_set = []

    for commuting_indices in horizontal_set:
        
        generator = FermionOperator()

        for i, j in commuting_indices:
            generator += FermionOperator(f'{i}^ {j}')
            generator += FermionOperator(f'{j}^ {i}')

        horizontal_op_set.append(generator)

    for commuting_indices in vertical_set:
        
        generator = FermionOperator()

        for i, j in commuting_indices:
            generator += FermionOperator(f'{i}^ {j}')
            generator += FermionOperator(f'{j}^ {i}')

        vertical_op_set.append(generator)

    return horizontal_op_set, vertical_op_set