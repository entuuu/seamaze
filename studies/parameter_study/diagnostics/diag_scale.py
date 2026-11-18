"""Diagnosis: does the covariance update act as a second step-size control?

Tracks, per iteration, the step size sigma, the overall covariance scale
tr(C)/n and the effective step size sigma * sqrt(tr(C)/n) for CMA-ES and
DLR-CMA-ES on one function with identical start point and seed.

Usage:
    python diag_scale.py [function] [seed]
"""

import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(ROOT), str(ROOT / 'variants')]

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from study import (FUNCTIONS, SIGMA_FRACTION, TARGET, _seed_numba,
                   default_pop)
from seamaze.optimizers import CMAES, DLRCMAES

N = 100
FROZEN = 10**9          # update interval that never triggers
OUT = ROOT / 'results' / 'diagnostics'

CONFIGS = [
    ('CMA-ES, Intervall 1', CMAES, {'update_interval': 1}, '#eb6834', '-'),
    ('CMA-ES, Kovarianz eingefroren', CMAES, {'update_interval': FROZEN},
     '#eb6834', ':'),
    ('DLR Standard', DLRCMAES, {}, '#2a78d6', '-'),
    ('DLR voller Rang, Intervall 1', DLRCMAES,
     {'low_rank_is_adaptive': False, 'low_rank_init_dimension': N,
      'update_interval': 1}, '#2a78d6', '--'),
    ('DLR, Kovarianz eingefroren', DLRCMAES, {'update_interval': FROZEN},
     '#2a78d6', ':'),
    ]


class Tracker:
    def __init__(self):
        self.rows = []

    def __call__(self, s):
        n = s._number_of_variables
        if hasattr(s, '_cov'):
            scale = np.trace(s._cov) / n
        else:
            scale = (np.sum(s._core) + np.sum(s._psi)) / n
        dist = np.linalg.norm(s._mean - self.optimum)
        self.rows.append((s._opt_iter * s._pop_size, s._sigma, scale, dist,
                          np.linalg.norm(s._path_sigma)
                          / s._expected_path_length))


def run(fname, seed, cls, kwargs):
    problem = FUNCTIONS[fname](N)
    lower, upper = map(np.asarray, problem.bounds)
    sigma0 = SIGMA_FRACTION * float(np.mean(upper - lower))
    np.random.seed(seed)
    _seed_numba(seed)
    x0 = np.random.default_rng(10_000 + seed).uniform(lower, upper)
    tracker = Tracker()
    tracker.optimum = np.ones(N) if fname == 'Rosenbrock' else np.zeros(N)
    solver = cls(
        number_of_variables=N, objective=problem.__call__,
        initial_sigma=sigma0, maximum_iterations=int(1e6 / default_pop(N)),
        maximum_wall_time=600, fitness_threshold=TARGET, tolerance=0.0,
        sigma_threshold=1e-14, min_log_level='critical', callback=tracker,
        random_state=seed, **kwargs)
    res = solver.optimize(x0)
    return np.array(tracker.rows), res


def main(fname, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    titles = ['Abstand des Mittelwerts zum Optimum',
              'Schrittweite σ', 'Kovarianz-Skala tr(C)/n']
    print(f'{fname}, n={N}, seed={seed}')
    print(f'{"Konfiguration":32s} {"Evals":>8s} {"Erfolg":>6s} '
          f'{"tr(C)/n Ende":>13s} {"sigma_eff*n/dist (Median)":>26s} '
          f'{"|p_sigma|/E (Median)":>21s}')
    for name, cls, kwargs, color, ls in CONFIGS:
        d, res = run(fname, seed, cls, kwargs)
        ev, sigma, scale, dist, ps = d.T
        eff = sigma * np.sqrt(scale)
        norm_step = eff * N / dist
        half = slice(len(ev) // 4, None)       # skip the initial transient
        print(f'{name:32s} {int(ev[-1]):8d} '
              f'{int(res["optimal_value"] < TARGET):6d} {scale[-1]:13.3e} '
              f'{np.median(norm_step[half]):26.2f} '
              f'{np.median(ps[half]):21.3f}')
        for ax, y in zip(axes, (dist, sigma, scale)):
            ax.plot(ev, y, color=color, linestyle=ls, linewidth=2,
                    label=name)
    for ax, title in zip(axes, titles):
        ax.set_yscale('log')
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_xlabel('Evaluierungen', fontsize=9)
        ax.grid(color='#e4e3dd', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'{fname}, n = {N}, Seed {seed}', x=0.01, ha='left',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / f'scale_{fname.replace(" ", "_")}_s{seed}.png'
    fig.savefig(path, dpi=150)
    print('Plot:', path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'Sphere',
         int(sys.argv[2]) if len(sys.argv) > 2 else 0)
