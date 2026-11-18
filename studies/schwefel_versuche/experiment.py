"""Schwefel-100D: compare DLR-CMA-ES settings over several seeds."""

from time import perf_counter

from numpy import array, mean, median, random

from seamaze.benchmarks import Schwefel
from seamaze.optimizers import DLRCMAES

ndim = 100
problem = Schwefel(ndim)
lower, upper = array(problem.bounds[0]), array(problem.bounds[1])
seeds = range(5)

# name: (use_bounds, random_start, fitness_window_size, tolerance)
configs = {
    'A baseline (sigma=100, start=100)': (False, False, 50, 1e-6),
    'B + bounds': (True, False, 50, 1e-6),
    'C + bounds + random start': (True, True, 50, 1e-6),
    'D + bounds + random start + loose stop': (True, True, 300, 1e-12),
}


def run(use_bounds, random_start, window, tol, seed):
    kwargs = {}
    if use_bounds:
        kwargs = {'lower_variable_bounds': lower,
                  'upper_variable_bounds': upper}
    solver = DLRCMAES(
        number_of_variables=ndim,
        objective=problem.__call__,
        number_of_individuals=None,
        initial_sigma=100.0,
        low_rank_init_dimension=None,
        low_rank_max_dimension=None,
        low_rank_is_adaptive=True,
        low_rank_energy_tolerance=1e-2,
        maximum_iterations=100000,
        maximum_wall_time=120,
        fitness_threshold=None,
        fitness_window_size=window,
        tolerance=tol,
        sigma_threshold=1e-8,
        update_interval=None,
        min_log_level='error',
        callback=None,
        random_state=seed,
        **kwargs
        )
    if random_start:
        x0 = random.default_rng(seed).uniform(-500.0, 500.0, ndim)
    else:
        x0 = array([100.0] * ndim)
    return solver.optimize(x0)


if __name__ == '__main__':
    for name, (b, r, w, t) in configs.items():
        values, iters, t0 = [], [], perf_counter()
        for s in seeds:
            res = run(b, r, w, t, s)
            values.append(res['optimal_value'])
            iters.append(res['iterations'])
        print(f'{name}\n   f per seed: '
              f'{[round(float(v), 1) for v in values]}\n'
              f'   best={min(values):.1f}  median={median(values):.1f}  '
              f'mean={mean(values):.1f}  mean iter={mean(iters):.0f}  '
              f'time={perf_counter()-t0:.1f}s', flush=True)
