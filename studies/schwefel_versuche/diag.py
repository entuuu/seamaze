"""Diagnose: why do larger populations perform worse on Schwefel-100D?

Runs CMAES, LMMAES and DLRCMAES with identical settings (bounds, random
start) for several population sizes and initial step sizes, and reports
final value, iterations, evaluations, stop reason and final step size.
"""

import sys
from time import perf_counter

from numpy import array, random

from seamaze.benchmarks import Schwefel
from seamaze.optimizers import CMAES, DLRCMAES, LMMAES

ndim = 100
problem = Schwefel(ndim)
lower, upper = array(problem.bounds[0]), array(problem.bounds[1])

solvers = {'CMAES': CMAES, 'LMMAES': LMMAES, 'DLRCMAES': DLRCMAES}
pops = [17, 136, 1088]
sigmas = [100.0, 300.0]
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

common = dict(
    number_of_variables=ndim,
    objective=problem.__call__,
    lower_variable_bounds=lower,
    upper_variable_bounds=upper,
    maximum_iterations=100000,
    maximum_wall_time=120,
    fitness_window_size=50,
    tolerance=1e-6,
    sigma_threshold=1e-8,
    min_log_level='error',
    callback=None,
    random_state=seed,
    )

print(f'{"solver":9s} {"sigma0":>6s} {"pop":>5s} {"f":>9s} {"iter":>6s} '
      f'{"evals":>8s} {"sigma_end":>10s} {"mean_x":>7s} {"time":>6s}  stop')
for name, cls in solvers.items():
    for s0 in sigmas:
        for pop in pops:
            x0 = random.default_rng(seed).uniform(-500.0, 500.0, ndim)
            solver = cls(number_of_individuals=pop, initial_sigma=s0,
                         **common)
            t0 = perf_counter()
            res = solver.optimize(x0)
            dt = perf_counter() - t0
            print(f'{name:9s} {s0:6.0f} {pop:5d} '
                  f'{res["optimal_value"]:9.1f} {res["iterations"]:6d} '
                  f'{res["iterations"] * pop:8d} '
                  f'{getattr(solver, "_sigma", float("nan")):10.2e} '
                  f'{res["optimal_point"].mean():7.1f} {dt:6.1f}  '
                  f'{res["solver_info"]}', flush=True)
