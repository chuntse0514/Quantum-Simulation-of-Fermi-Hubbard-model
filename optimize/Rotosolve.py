import torch
from torch.optim import Optimizer
import numpy as np
import pennylane as qml
from inspect import signature

def _univariate_func(func, params, index, ix):
    
    variable_vec = torch.zeros_like(params[index])
    variable_vec[ix] = 1

    def shift_func(x):
        return func(*params[:index], params[index] + x * variable_vec, *params[index+1:])
    
    return shift_func

def analytic_solver(func, f0=None):
        
    if f0 is None:
        f0 = func(0)

    f_plus = func(np.pi / 2) 
    f_minus = func(-np.pi / 2)

    a3 = (f_plus + f_minus) / 2
    a2 = np.arctan2(f_minus - a3, f0 - a3)
    a1 = (f_minus - f_plus) / (2 * np.sin(a2))
    
    x_min = np.pi - a2
    y_min = a3 - a1
    
    if x_min < -np.pi:
        x_min += 2 * np.pi
    if x_min > np.pi:
        x_min -= 2 * np.pi

    return x_min, y_min

def numeric_solver(func, n_steps, n_points, device):

    def torch_brute(func, interval: tuple):
        X = torch.linspace(interval[0], interval[1], steps=n_points).to(device)
        Y = torch.Tensor([func(x) for x in X])
        index = torch.argmin(Y)
        return X[index], Y[index]
    
    width = 2 * np.pi
    center = 0
    for _ in range(n_steps):
        interval = (center - width / 2, center + width / 2)
        x_min, y_min = torch_brute(func, interval)
        center = x_min
        width /= n_points

    return x_min.item(), y_min.item()

def full_reconstruction_eq(func, R, device, f0=None):

    if not f0:
        f0 = func(0)

    X_mu = torch.linspace(-R, R, steps=2*R+1).to(device) * (2*np.pi) / (2*R + 1)
    E_mu = torch.Tensor([func(x_mu) for x_mu in X_mu[:R]] + [f0] + [func(x_mu) for x_mu in X_mu[R+1:]]).to(device)
    
    def reconstruction_func(x):

        # Definition of torch.sinc(x) is sin(\pi * x) / (\pi * x)
        # However, our definition is sinc(x) = sin(x) / x
        
        kernel = torch.sinc((2*R + 1) / (2*np.pi) * (x - X_mu)) / torch.sinc(1 / (2 * np.pi) * (x - X_mu))
        return torch.inner(E_mu, kernel)
    
    return reconstruction_func
        


class RotosolveTorch:

    def __init__(self,
                 params,
                 objective_func,
                 num_freqs,
                 full_output=False
                 ):
        
        self.params = params
        self.device = params[0].device
        self.objective_func = objective_func
        self.num_freqs = num_freqs
        self.full_output = full_output

    def step_and_cost(self):
        
        # Get the argument name of the qfunc of the input qnode
        qfunc = self.objective_func.func
        params_name = list(signature(qfunc).parameters.keys())
        requires_grad = {
            param_name: param.requires_grad for param_name, param in zip(params_name, self.params)
        }

        if self.full_output:
            loss_history = []

        init_func_val = self.objective_func(*self.params)

        for index, (param, param_name) in enumerate(zip(self.params, params_name)):

            if not requires_grad[param_name]:
                continue

            for ix, x in enumerate(param):
                
                total_freq = self.num_freqs[param_name][ix]
                univariate_func = _univariate_func(self.objective_func, self.params, index, ix)

                if total_freq == 1:
                    x_min, y_min = analytic_solver(univariate_func, f0=init_func_val)
                    self.params[index].data[ix] = x + x_min
                    init_func_val = y_min
                    loss_history.append(y_min)

                else:
                    reconstructed_func = full_reconstruction_eq(univariate_func, 
                                                                total_freq, 
                                                                device=self.device,
                                                                f0=init_func_val)
                    
                    x_min, y_min = numeric_solver(reconstructed_func,
                                                  n_steps=2, 
                                                  n_points=50,
                                                  device=self.device)
                    self.params[index].data[ix] = x + x_min
                    init_func_val = y_min
                    loss_history.append(y_min)
        
        if self.full_output:
            return loss_history
        
if __name__ == '__main__':

    from models.vqe_hva_pennylane import HVA

    vqe = HVA(
        n_epoch=100,
        lr=1e-2,
        threshold=1e-3,
        reps=3,
        x_dimension=2,
        y_dimension=2,
        tunneling=1,
        coulomb=4
    )

    params = (vqe.params['theta_U'], vqe.params['theta_v'], vqe.params['theta_h'])
    dev = qml.device('default.qubit.torch', wires=vqe.n_qubits)
    qnode = qml.QNode(vqe.circuit, dev, interface='torch', diff_method='backprop')
    num_freqs = {
        'theta_U': torch.full((vqe.reps+1,), 2),
        'theta_v': torch.full((vqe.reps,), 4),
        'theta_h': torch.full((vqe.reps,), 4)
    }
    
    opt = RotosolveTorch(
        params=params,
        objective_func=qnode,
        num_freqs=num_freqs,
        full_output=True
    )

    loss = opt.step_and_cost()

    import matplotlib.pyplot as plt

    plt.plot(np.arange(len(loss)) + 1, loss)
    plt.show()