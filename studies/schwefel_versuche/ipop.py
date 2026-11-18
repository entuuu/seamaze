"""Schwefel-100D: IPOP-style restarts (doubling population) for DLR-CMA-ES."""

from math import floor, log
from time import perf_counter

from numpy import array, mean, median, random

from seamaze.benchmarks import Schwefel
from seamaze.optimizers import DLRCMAES

ndim = 100
problem = Schwefel(ndim)
lower, upper = array(problem.bounds[0]), array(problem.bounds[1])
seeds = range(5)
base_pop = 4 + floor(3 * log(ndim))   # default CMA-ES population size
n_restarts = 8                         # populations: base_pop * 2**k
sigma0 = 100.0


def single_run(pop, seed):
    solver = DLRCMAES(
        number_of_variables=ndim,
        objective=problem.__call__,
        lower_variable_bounds=lower,
        upper_variable_bounds=upper,
        number_of_individuals=pop,
        initial_sigma=sigma0,
        low_rank_is_adaptive=True,
        low_rank_energy_tolerance=1e-2,
        maximum_iterations=100000,
        maximum_wall_time=120,
        fitness_window_size=50,
        tolerance=1e-6,
        sigma_threshold=1e-8,
        min_log_level='error',
        callback=None,
        random_state=seed
        )
    x0 = random.default_rng(seed).uniform(-500.0, 500.0, ndim)
    return solver.optimize(x0)


if __name__ == '__main__':
    print(f'base population = {base_pop}, restarts = {n_restarts}')
    finals, per_pop = [], {}
    t0 = perf_counter()
    for s in seeds:
        best, trace = float('inf'), []
        for k in range(n_restarts):
            pop = base_pop * 2**k
            # different start/seed per restart
            res = single_run(pop, 1000 * s + k)
            trace.append((pop, round(float(res['optimal_value']), 1)))
            per_pop.setdefault(pop, []).append(float(res['optimal_value']))
            best = min(best, float(res['optimal_value']))
        finals.append(best)
        print(f'seed {s}: best over restarts = {best:.1f}\n   '
              f'(pop, f) = {trace}', flush=True)
    print(f'\nIPOP best-of-restarts: best={min(finals):.1f} '
          f'median={median(finals):.1f} mean={mean(finals):.1f} '
          f'time={perf_counter()-t0:.1f}s')
    print('\nmean f per population size:')
    for pop, v in per_pop.items():
        print(f'   pop={pop:5d}: mean={mean(v):.1f}  min={min(v):.1f}')
