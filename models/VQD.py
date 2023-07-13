#https://qiskit.org/documentation/_modules/qiskit/algorithms/eigensolvers/vqd.html#VQD
#test 
from collections.abc import Callable, Sequence
from typing import Any
from time import time

import numpy as np

from openfermion import (
    MolecularData, get_sparse_operator, jordan_wigner
)
from .utils import (
    QubitOperator_to_SparsePauliOp,
    prepare_HF_state,
)
from qiskit.primitives import Estimator
from ..utils import validate_bounds, validate_initial_point # I cannot find source code
from ..utils.set_batching import _set_default_batchsize
class VQD:
    def __init__(
        self,
        estimator: BaseEstimator,
        fidelity: BaseStateFidelity,
        ansatz: QuantumCircuit,
        optimizer: Optimizer | Minimizer | Sequence[Optimizer | Minimizer],
        *,
        k: int = 2,
        betas: Sequence[float] | None = None,
        initial_point: Sequence[float] | Sequence[Sequence[float]] | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
    ) -> None:
        
        self.estimator = estimator
        self.fidelity = fidelity
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.k = k
        self.betas = betas

        self.initial_point = initial_point
        self.callback = callback

        self._eval_count = 0
    
    def compute_eigenvalues(self, operator: BaseOperator | PauliSumOp,):
        
        # self._check_operator_ansatz(operator)

        #I cannot find good documentation on validate_bounds()
        bounds = validate_bounds(self.ansatz)

        #ignoring "aux_operators"
        betas = self.betas
        result = self._build_vqd_result() #!!!

        prev_states = []
        num_initial_points = 0
        if self.initial_point is not None:
            initial_points = np.reshape(self.initial_point, (-1, self.ansatz.num_parameters))
            num_initial_points = len(initial_points)

        #if the initial point was not specified:
            #I cannot find source code, so it is best if you just chose your own initial points
        if num_initial_points <= 1:
            initial_point = validate_initial_point(self.initial_point, self.ansatz)

        for step in range(1, self.k + 1):
            if num_initial_points > 1:
                initial_point = validate_initial_point(initial_points[step - 1], self.ansatz)

            if step > 1:
                prev_states.append(self.ansatz.bind_parameters(result.optimal_points[-1]))

            self._eval_count = 0
            energy_evaluation = self._get_evaluate_energy(
                step, operator, betas, prev_states=prev_states
            )

            start_time = time()

            #This is for if you have a sequence of optimizers, I don't think this is important
            if isinstance(self.optimizer, Sequence):
                optimizer = self.optimizer[step - 1]
            else:
                optimizer = self.optimizer  # fall back to single optimizer if not list

            #I think the optimizer we want will always be callable
            if callable(optimizer):
                opt_result = optimizer(  # pylint: disable=not-callable
                    fun=energy_evaluation, x0=initial_point, bounds=bounds
                )
            else: #again, I don't we will use this part, but I kept it here
                # we always want to submit as many estimations per job as possible for minimal
                # overhead on the hardware
                was_updated = _set_default_batchsize(optimizer)

                opt_result = optimizer.minimize(
                    fun=energy_evaluation, x0=initial_point, bounds=bounds
                )

                # reset to original value
                if was_updated:
                    optimizer.set_max_evals_grouped(None)

            eval_time = time() - start_time

            # self._update_vqd_result(result, opt_result, eval_time, self.ansatz.copy())

            result.eigenvalues = np.array(result.eigenvalues)
            return result
        

#Untouched from Qiskit documentation:
    def _get_evaluate_energy(
        self,
        step: int,
        operator: BaseOperator | PauliSumOp,
        betas: Sequence[float],
        prev_states: list[QuantumCircuit] | None = None,
    ) -> Callable[[np.ndarray], float | np.ndarray]:
        """Returns a function handle to evaluate the ansatz's energy for any given parameters.
            This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            step: level of energy being calculated. 0 for ground, 1 for first excited state...
            operator: The operator whose energy to evaluate.
            betas: Beta parameters in the VQD paper.
            prev_states: List of optimal circuits from previous rounds of optimization.

        Returns:
            A callable that computes and returns the energy of the hamiltonian
            of each parameter.

        Raises:
            AlgorithmError: If the circuit is not parameterized (i.e. has 0 free parameters).
            AlgorithmError: If operator was not provided.
            RuntimeError: If the previous states array is of the wrong size.
        """

        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")

        if step > 1 and (len(prev_states) + 1) != step:
            raise RuntimeError(
                f"Passed previous states of the wrong size."
                f"Passed array has length {str(len(prev_states))}"
            )

        self._check_operator_ansatz(operator)

        def evaluate_energy(parameters: np.ndarray) -> float | np.ndarray:
            # handle broadcasting: ensure parameters is of shape [array, array, ...]
            if len(parameters.shape) == 1:
                parameters = np.reshape(parameters, (-1, num_parameters))
            batch_size = len(parameters)

            estimator_job = self.estimator.run(
                batch_size * [self.ansatz], batch_size * [operator], parameters
            )

            total_cost = np.zeros(batch_size)

            if step > 1:
                # compute overlap cost
                batched_prev_states = [state for state in prev_states for _ in range(batch_size)]
                fidelity_job = self.fidelity.run(
                    batch_size * [self.ansatz] * (step - 1),
                    batched_prev_states,
                    np.tile(parameters, (step - 1, 1)),
                )
                costs = fidelity_job.result().fidelities

                costs = np.reshape(costs, (step - 1, -1))
                for state, cost in enumerate(costs):
                    total_cost += np.real(betas[state] * cost)

            try:
                estimator_result = estimator_job.result()

            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

            values = estimator_result.values + total_cost

            if self.callback is not None:
                metadata = estimator_result.metadata
                for params, value, meta in zip(parameters, values, metadata):
                    self._eval_count += 1
                    self.callback(self._eval_count, params, value, meta, step)
            else:
                self._eval_count += len(values)

            return values if len(values) > 1 else values[0]

        return evaluate_energy
    




    
    def _update_vqd_result(
        result: VQDResult, opt_result: OptimizerResult, eval_time, ansatz
    ) -> VQDResult:
        result.optimal_points = (
            np.concatenate([result.optimal_points, [opt_result.x]])
            if len(result.optimal_points) > 0
            else np.array([opt_result.x])
        )
        result.optimal_parameters.append(dict(zip(ansatz.parameters, opt_result.x)))
        result.optimal_values = np.concatenate([result.optimal_points, [opt_result.x]])
        result.cost_function_evals = np.concatenate([result.cost_function_evals, [opt_result.nfev]])
        result.optimizer_times = np.concatenate([result.optimizer_times, [eval_time]])
        result.eigenvalues.append(opt_result.fun + 0j)
        result.optimizer_results.append(opt_result)
        result.optimal_circuits.append(ansatz)
        return result