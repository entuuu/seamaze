"""Diagnosis: effect of the negative recombination weights (active update)
on the covariance scale and on the evaluations to target.

The negative weights are set to zero on the solver instance after
construction; the library itself is not modified.

Usage:
    python diag_weights.py [function] [seed]
"""

import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(ROOT), str(ROOT / 'variants')]

import sys

import numpy as np

from study import (FUNCTIONS, SIGMA_FRACTION, TARGET, _seed_numba,
                   default_pop)
from seamaze.optimizers import CMAES, DLRCMAES

N = 100


def run(fname, seed, cls, kwargs, drop_negative):
    problem = FUNCTIONS[fname](N)
    lower, upper = map(np.asarray, problem.bounds)
    sigma0 = SIGMA_FRACTION * float(np.mean(upper - lower))
    np.random.seed(seed)
    _seed_numba(seed)
    x0 = np.random.default_rng(10_000 + seed).uniform(lower, upper)
    solver = cls(
        number_of_variables=N, objective=problem.__call__,
        initial_sigma=sigma0, maximum_iterations=int(1e6 / default_pop(N)),
        maximum_wall_time=600, fitness_threshold=TARGET, tolerance=0.0,
        sigma_threshold=1e-14, min_log_level='critical',
        random_state=seed, **kwargs)
    if drop_negative:
        solver._weights = np.maximum(solver._weights, 0.0)
    res = solver.optimize(x0)
    if hasattr(solver, '_cov'):
        scale = np.trace(solver._cov) / N
    else:
        scale = (np.sum(solver._core) + np.sum(solver._psi)) / N
    return res, scale, solver


if __name__ == '__main__':
    fname = sys.argv[1] if len(sys.argv) > 1 else 'Sphere'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    s = CMAES(number_of_variables=N, objective=lambda x: 0.0,
              min_log_level='critical')
    w = s._weights.ravel()
    print(f'Gewichte: Summe positiv {w[w > 0].sum():.3f}, '
          f'Summe negativ {w[w < 0].sum():.3f}')
    configs = [
        ('CMA-ES, Intervall 1', CMAES, {'update_interval': 1}),
        ('DLR voller Rang, Intervall 1', DLRCMAES,
         {'low_rank_is_adaptive': False, 'low_rank_init_dimension': N,
          'update_interval': 1}),
        ('DLR Standard', DLRCMAES, {}),
        ]
    print(f'{fname}, n={N}, seed={seed}')
    for name, cls, kwargs in configs:
        for drop in (False, True):
            res, scale, _ = run(fname, seed, cls, kwargs, drop)
            tag = 'ohne negative Gewichte' if drop else 'mit negativen Gewichten'
            ok = int(res['optimal_value'] < TARGET)
            print(f'{name:30s} {tag:24s} Evals={res["iterations"] * 17:8d} '
                  f'Erfolg={ok} tr(C)/n={scale:.3e}', flush=True)
