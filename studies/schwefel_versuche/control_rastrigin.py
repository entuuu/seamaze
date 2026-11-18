"""Control experiment: on Rastrigin (multimodal, but with a strong global
structure), larger populations are known to help CMA-ES. If the
implementation is fine, f should drop clearly with growing population."""

from time import perf_counter

from numpy import array, random

from seamaze.benchmarks import Rastrigin
from seamaze.optimizers import CMAES, DLRCMAES

ndim = 100
problem = Rastrigin(ndim)
lower, upper = array(problem.bounds[0]), array(problem.bounds[1])
seed = 0

print(f'{"solver":9s} {"pop":>5s} {"f":>9s} {"iter":>6s} {"time":>6s}  stop')
for name, cls in {'CMAES': CMAES, 'DLRCMAES': DLRCMAES}.items():
    for pop in [17, 136, 1088]:
        x0 = random.default_rng(seed).uniform(-5.12, 5.12, ndim)
        solver = cls(
            number_of_variables=ndim, objective=problem.__call__,
            lower_variable_bounds=lower, upper_variable_bounds=upper,
            number_of_individuals=pop, initial_sigma=2.0,
            maximum_iterations=100000, maximum_wall_time=120,
            fitness_window_size=50, tolerance=1e-6, sigma_threshold=1e-8,
            min_log_level='error', callback=None, random_state=seed)
        t0 = perf_counter()
        res = solver.optimize(x0)
        print(f'{name:9s} {pop:5d} {res["optimal_value"]:9.2f} '
              f'{res["iterations"]:6d} {perf_counter()-t0:6.1f}  '
              f'{res["solver_info"]}', flush=True)
