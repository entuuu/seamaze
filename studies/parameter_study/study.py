"""DLR-CMA-ES parameter study: configurations and single-run worker."""

import os

# Avoid BLAS/numba oversubscription in parallel workers (must precede numpy)
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import sys
from pathlib import Path

# Diagnostic solver copies live in ./variants
_VARIANTS = str(Path(__file__).resolve().parent / 'variants')
if _VARIANTS not in sys.path:
    sys.path.insert(0, _VARIANTS)

from math import ceil, log
from time import perf_counter

import numpy as np
from numba import njit

from seamaze.benchmarks import (
    BentCigar, Discus, Ellipsoid, Rastrigin, Rosenbrock, RotatedEllipsoid,
    Sphere)
from seamaze.optimizers import CMAES, DLRCMAES

# %% Global settings

NDIM = 100
TARGET = 1e-8
BUDGET_FACTOR = 1e4          # evaluation budget = BUDGET_FACTOR * n
WALL_CAP = 900               # safety cap per run at n = 100 (seconds)


def wall_cap(n):
    """Safety cap per run, grows with n^2 (unchanged 900 s at n <= 100)."""
    return max(WALL_CAP, WALL_CAP * (n / 100) ** 2)
SIGMA_FRACTION = 0.3         # sigma0 = fraction of the search range width

class Rotated:
    """f(Q x) with a fixed random orthogonal Q (seeded per function name)."""

    def __init__(self, base_cls, n):
        self.base = base_cls(n)
        self.name = f'Rot-{self.base.name}'
        self.ndim = n
        self.bounds = self.base.bounds
        seed = sum(map(ord, self.base.name)) + 7919 * n
        gauss = np.random.default_rng(seed).standard_normal((n, n))
        q, r = np.linalg.qr(gauss)
        self.q = q * np.sign(np.diag(r))      # Haar-distributed rotation

    def __call__(self, x):
        return self.base(self.q @ np.asarray(x))


def _rotated(base_cls):
    return lambda n: Rotated(base_cls, n)


FUNCTIONS = {
    'Sphere': Sphere,
    'Bent-Cigar': BentCigar,
    'Discus': Discus,
    'Ellipsoid': Ellipsoid,
    'Rotated Ellipsoid': RotatedEllipsoid,
    'Rosenbrock': Rosenbrock,
    'Rastrigin': Rastrigin,
    'Rot-Bent-Cigar': _rotated(BentCigar),
    'Rot-Discus': _rotated(Discus),
    'Rot-Ellipsoid': _rotated(Ellipsoid),
    }

ROTATED_FUNCTIONS = ['Rot-Bent-Cigar', 'Rot-Discus', 'Rot-Ellipsoid']
MAIN_FUNCTIONS = ['Sphere', 'Bent-Cigar', 'Discus', 'Ellipsoid',
                  'Rotated Ellipsoid', 'Rosenbrock']


def default_pop(n):
    return 4 + int(3 * log(n))


# %% Configurations (stage 1: one factor at a time around the default)
# Each config: (group, label, kwargs, sigma_scale, functions)
# kwargs may contain the placeholder 'n' via callables (resolved per n).

def stage1_configs(n=NDIM):
    lam = default_pop(n)
    cfgs = [('default', 'default', {}, 1.0, MAIN_FUNCTIONS + ['Rastrigin'])]
    for tol in [1e-4, 1e-3, 5e-2, 1e-1, 3e-1]:
        cfgs.append(('energy_tol', f'tol={tol:g}',
                     {'low_rank_energy_tolerance': tol}, 1.0,
                     MAIN_FUNCTIONS))
    # Bootstrap noise filter of the rank selection (since upstream 42989fb;
    # defaults: quantile 0.95, size 32)
    for q in [0.5, 0.8, 0.99]:
        cfgs.append(('bootstrap_quantile', f'bq={q:g}',
                     {'low_rank_bootstrap_quantile': q}, 1.0,
                     MAIN_FUNCTIONS))
    for b in [8, 128]:
        cfgs.append(('bootstrap_size', f'bsize={b}',
                     {'low_rank_bootstrap_size': b}, 1.0, MAIN_FUNCTIONS))
    for frac in [20, 10, 4, 2]:
        r = max(1, n // frac)
        cfgs.append(('max_dim', f'rmax={r}',
                     {'low_rank_max_dimension': r}, 1.0, MAIN_FUNCTIONS))
    for r in [1, 2, 5, 10, 25, 50]:
        cfgs.append(('fixed_rank', f'fixed r={r}',
                     {'low_rank_is_adaptive': False,
                      'low_rank_init_dimension': r}, 1.0, MAIN_FUNCTIONS))
    for r in [5, 20, n // 2]:
        cfgs.append(('init_dim', f'r0={r}',
                     {'low_rank_init_dimension': r}, 1.0, MAIN_FUNCTIONS))
    for k in [1, 2, 5, 10, 20]:
        cfgs.append(('update_interval', f'interval={k}',
                     {'update_interval': k}, 1.0, MAIN_FUNCTIONS))
    for mult in [2, 4, 8]:
        cfgs.append(('pop_size', f'pop={mult * lam}',
                     {'number_of_individuals': mult * lam}, 1.0,
                     MAIN_FUNCTIONS + ['Rastrigin']))
    # Larger populations only for the multimodal Rastrigin function
    for mult in [16, 32]:
        cfgs.append(('pop_size', f'pop={mult * lam}',
                     {'number_of_individuals': mult * lam}, 1.0,
                     ['Rastrigin']))
    for scale in [0.1, 10.0]:
        cfgs.append(('sigma0', f'sigma0 x{scale:g}', {}, scale,
                     MAIN_FUNCTIONS))
    return cfgs


def ladder_configs(n=NDIM):
    """Attribution ladder: CMA-ES -> full-rank DLR -> fixed rank -> adaptive.

    Complements stage 1, which already holds 'default' (adaptive rank, auto
    interval), 'interval=1' (adaptive rank) and 'fixed r=k' (auto interval).
    """
    funcs = MAIN_FUNCTIONS
    cfgs = [
        ('ladder', 'CMA-ES', {'_solver': 'cmaes'}, 1.0,
         funcs + ['Rastrigin']),
        ('ladder', 'CMA-ES, interval=1',
         {'_solver': 'cmaes', 'update_interval': 1}, 1.0, funcs),
        ('ladder', 'full rank',
         {'low_rank_is_adaptive': False, 'low_rank_init_dimension': n},
         1.0, funcs),
        ('ladder', 'full rank, interval=1',
         {'low_rank_is_adaptive': False, 'low_rank_init_dimension': n,
          'update_interval': 1}, 1.0, funcs),
        ]
    for r in [1, 2, 5, 10, 25, 50]:
        cfgs.append(('ladder', f'fixed r={r}, interval=1',
                     {'low_rank_is_adaptive': False,
                      'low_rank_init_dimension': r, 'update_interval': 1},
                     1.0, funcs))
    return cfgs


def ref_configs():
    """Corrected CMA-ES reference (step-size path in original coordinates,
    see cmaes_path_variant.py)."""
    return [
        ('ladder_ref', 'CMA-ES korr., interval=1',
         {'_solver': 'cmaes_fixed', 'update_interval': 1}, 1.0,
         MAIN_FUNCTIONS),
        ('ladder_ref', 'CMA-ES korr.', {'_solver': 'cmaes_fixed'}, 1.0,
         MAIN_FUNCTIONS),
        ]


def psifold_configs(n=NDIM):
    """Diagnosis: psi projected into the core update (dlr_variant_psifold),
    full rank and update interval 1. Library counterpart: 'full rank,
    interval=1' in ladder.csv."""
    return [('psifold', 'psifold full rank, interval=1',
             {'_solver': 'dlr_psifold', 'low_rank_is_adaptive': False,
              'low_rank_init_dimension': n, 'update_interval': 1}, 1.0,
             MAIN_FUNCTIONS)]


STAGE2_TOLS = [1e-3, 1e-2, 5e-2, 1e-1, 3e-1]
STAGE2_INTERVALS = [None, 1, 2, 3]          # None = automatic (default)
STAGE2_FUNCTIONS = ['Ellipsoid', 'Rotated Ellipsoid', 'Rosenbrock',
                    'Bent-Cigar', 'Discus']


def stage2_label(tol, interval):
    return f'tol={tol:g}, interval={"auto" if interval is None else interval}'


def stage2_configs():
    """Stage 2: full grid low_rank_energy_tolerance x update_interval.
    'tol=0.01, interval=auto' is the library default."""
    cfgs = []
    for tol in STAGE2_TOLS:
        for interval in STAGE2_INTERVALS:
            kw = {'low_rank_energy_tolerance': tol}
            if interval is not None:
                kw['update_interval'] = interval
            cfgs.append(('stage2', stage2_label(tol, interval), kw, 1.0,
                         STAGE2_FUNCTIONS))
    return cfgs


STAGE3_DIMS = [30, 100, 300]
STAGE3_FUNCTIONS = ['Ellipsoid', 'Rotated Ellipsoid', 'Rosenbrock',
                    'Bent-Cigar']
# 1e4*n is not enough at n = 300 (pilot 18.11.2026: all runs failed);
# stage 3 is self-contained with a 3e4*n budget at every n.
STAGE3_BUDGET_FACTOR = 3e4
# Sequential timing runs (stage3_timing): subset, see PLAN.md
STAGE3_TIMING_FUNCTIONS = ['Ellipsoid', 'Bent-Cigar']


def stage3_configs():
    """Stage 3: scaling over n (own CSV, budget STAGE3_BUDGET_FACTOR * n).
    Labels follow stage 2 / ladder_ref; results are not mixed with them
    because the budget differs."""
    funcs = STAGE3_FUNCTIONS
    extra = {'_budget_factor': STAGE3_BUDGET_FACTOR}
    cfgs = []
    for tol, interval in [(1e-2, None), (5e-2, None), (3e-1, None),
                          (1e-2, 1)]:
        kw = {'low_rank_energy_tolerance': tol, **extra}
        if interval is not None:
            kw['update_interval'] = interval
        cfgs.append(('stage3', stage2_label(tol, interval), kw, 1.0, funcs))
    cfgs.append(('stage3', 'CMA-ES korr.',
                 {'_solver': 'cmaes_fixed', **extra}, 1.0, funcs))
    cfgs.append(('stage3', 'CMA-ES korr., interval=1',
                 {'_solver': 'cmaes_fixed', 'update_interval': 1, **extra},
                 1.0, funcs))
    return cfgs


def stage2_rosenbrock_configs():
    """Stage 2 grid on Rosenbrock only (to extend it to more seeds; results
    go into the stage2 CSV, same labels)."""
    return [(g, lab, kw, sc, ['Rosenbrock'])
            for g, lab, kw, sc, _ in stage2_configs()]


def _solver_class(name):
    """Map a solver key to its class (variants are imported lazily)."""
    if name == 'cmaes':
        return CMAES
    if name == 'cmaes_fixed':
        from cmaes_path_variant import CMAES as CMAESFixed
        return CMAESFixed
    if name.startswith('dlr_'):
        import importlib
        return importlib.import_module(f'dlr_variant_{name[4:]}').DLRCMAES
    return DLRCMAES


# %% Reproducibility helpers

@njit
def _seed_numba(seed):
    np.random.seed(seed)


class _RankTracker:
    """Minimal callback: track rank statistics without extra linear algebra."""

    def __init__(self):
        self.sum = 0
        self.max = 0
        self.count = 0

    def __call__(self, solver):
        r = getattr(solver, 'rank', solver._number_of_variables)
        self.sum += r
        self.count += 1
        if r > self.max:
            self.max = r


# %% Single run

def run_single(task):
    """Run one (config, function, n, seed) task and return a result row."""

    group, label, kwargs, sigma_scale, fname, n, seed = task
    kwargs = dict(kwargs)
    solver_cls = _solver_class(kwargs.pop('_solver', 'dlr'))

    problem = FUNCTIONS[fname](n)
    lower, upper = np.asarray(problem.bounds[0]), np.asarray(problem.bounds[1])
    width = float(np.mean(upper - lower))
    sigma0 = SIGMA_FRACTION * width * sigma_scale

    # Seed all RNGs used by the solver (own RNG, global numpy, numba)
    np.random.seed(seed)
    _seed_numba(seed)
    x0 = np.random.default_rng(10_000 + seed).uniform(lower, upper)

    pop = kwargs.get('number_of_individuals', default_pop(n))
    budget_factor = kwargs.pop('_budget_factor', BUDGET_FACTOR)
    budget = int(budget_factor * n)
    max_iter = ceil(budget / pop)

    tracker = _RankTracker()
    solver = solver_cls(
        number_of_variables=n,
        objective=problem.__call__,
        initial_sigma=sigma0,
        maximum_iterations=max_iter,
        maximum_wall_time=wall_cap(n) * budget_factor / BUDGET_FACTOR,
        fitness_threshold=TARGET,
        fitness_window_size=50,
        tolerance=0.0,               # disables both plateau criteria
        sigma_threshold=1e-14,
        min_log_level='critical',
        callback=tracker,
        random_state=seed,
        **kwargs)

    t0 = perf_counter()
    res = solver.optimize(x0)
    wall = perf_counter() - t0

    iters = res['iterations']
    return {
        'group': group, 'label': label, 'function': fname, 'n': n,
        'seed': seed, 'pop': pop, 'sigma0': sigma0,
        'success': int(res['optimal_value'] < TARGET),
        'f_best': float(res['optimal_value']),
        'iterations': iters,
        'evals': iters * pop,     # evaluations = iterations x population
        'budget': budget,
        'final_rank': getattr(solver, 'rank', n),
        'mean_rank': tracker.sum / max(tracker.count, 1),
        'max_rank': tracker.max,
        'final_sigma': float(solver._sigma),
        'wall_time': wall,
        'time_per_iter_ms': 1e3 * wall / max(iters, 1),
        'stop': res['solver_info'],
        }


def task_key(task):
    group, label, _, _, fname, n, seed = task
    return f'{label}|{fname}|{n}|{seed}'
