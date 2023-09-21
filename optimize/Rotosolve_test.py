from optimize.Rotosolve import (
    analytic_solver,
    numeric_solver,
    full_reconstruction_eq,
    RotosolveTorch
)
import pennylane as qml
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import unitary_group
import numpy.random as rnd

def random_state(N, seed):
    """Create a random state on N qubits."""
    states = unitary_group.rvs(2 ** N, random_state=rnd.default_rng(seed))
    return states[0]


def random_observable(N, seed):
    """Create a random observable on N qubits."""
    rnd.seed(seed)
    # Generate real and imaginary part separately and (anti-)symmetrize them for Hermiticity
    real_part, imag_part = rnd.random((2, 2 ** N, 2 ** N))
    real_part += real_part.T
    imag_part -= imag_part.T
    return real_part + 1j * imag_part

def make_multi_frequency_cost(N, seed):
    """Create a cost function on N qubits with N frequencies."""
    dev = qml.device("default.qubit.torch", wires=N)

    @qml.qnode(dev, interface="torch")
    def cost(x):
        """Cost function on N qubits with N frequencies."""
        qml.StatePrep(random_state(N, seed), wires=dev.wires)
        for w in dev.wires:
            qml.RZ(x, wires=w, id="x")
        return qml.expval(qml.Hermitian(random_observable(N, seed), wires=dev.wires))

    return cost

def make_single_frequency_cost(N, seed):
    """Create a cost function on N qubits with single frequency"""
    dev = qml.device("default.qubit.torch", wires=N)

    @qml.qnode(dev, interface='torch')
    def cost(x):
        qml.StatePrep(random_state(N, seed), wires=dev.wires)
        qml.RZ(x, wires=0)
        return qml.expval(qml.Hermitian(random_observable(N, seed), wires=dev.wires))
    
    return cost

################################################################################
def test_full_reconstruction_eq(n_qubits):

    device = torch.device('cuda:0')

    cost = make_multi_frequency_cost(n_qubits, 438)

    X = torch.linspace(-np.pi, np.pi, 50).to(device)
    reconst_func = full_reconstruction_eq(func=cost, R=n_qubits, device=device)

    cost_eval = np.array([cost(x).item() for x in X])
    reconst_eval = np.array([reconst_func(x).item() for x in X])

    plt.plot(X.cpu(), cost_eval, label='real cost', color='darkorange')
    plt.plot(X.cpu(), reconst_eval, label='reconst cost', ls=':', color='darkolivegreen')
    plt.legend()
    plt.show()

def test_analytic_solver(n_qubits):

    device = torch.device('cuda:0')

    cost = make_single_frequency_cost(n_qubits, 1822)

    X = torch.linspace(-np.pi, np.pi, 50).to(device)
    cost_eval = np.array([cost(x).item() for x in X])
    x_min, y_min = analytic_solver(cost)

    plt.plot(X.cpu(), cost_eval, label='loss function', color='lightseagreen')
    plt.plot([x_min], [y_min], label='analytic minimum', ls="", marker="X", color='deeppink')
    plt.title('test for analytic solver')
    plt.legend()
    plt.show()

def test_numeric_solver(n_qubits):

    device = torch.device('cuda:0')

    cost = make_multi_frequency_cost(n_qubits, 1124)

    X = torch.linspace(-np.pi, np.pi, 50).to(device)
    cost_eval = np.array([cost(x).item() for x in X])
    reconstructed_func = full_reconstruction_eq(func=cost, R=n_qubits, device=device)
    x_min, y_min = numeric_solver(reconstructed_func, n_steps=2, n_points=50, device=device)

    plt.plot(X.cpu(), cost_eval, label='loss function', color='lightseagreen')
    plt.plot([x_min], [y_min], label='analytic minimum', ls="", marker="X", color='deeppink')
    plt.title('test for numeric solver')
    plt.legend()
    plt.show()