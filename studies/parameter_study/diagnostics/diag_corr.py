"""Control for diag_rank.py: how non-diagonal does C become on the
axis-parallel Ellipsoid in corrected CMA-ES and in DLR with fixed rank?
(The optimal covariance is diagonal there, i.e. all correlations 0.)

Usage:
    python diag_corr.py [function] [seed]
"""

import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(ROOT), str(ROOT / 'variants')]

import sys

import numpy as np

from study import (FUNCTIONS, SIGMA_FRACTION, TARGET, _seed_numba,
                   default_pop)
from seamaze.optimizers import DLRCMAES
from cmaes_path_variant import CMAES as CMAESFixed

N = 100
EVERY = 200


class Corr:
    def __init__(self):
        self.rows = []

    def __call__(self, s):
        if s._opt_iter % EVERY:
            return
        if hasattr(s, '_cov'):
            c, psi_share = s._cov, np.nan
        else:
            c = (s._basis * s._core) @ s._basis.T + np.diag(s._psi)
            psi_share = s._psi.sum() / np.trace(c)
        d = np.sqrt(np.diag(c))
        corr = c / np.outer(d, d) - np.eye(N)
        self.rows.append((s._opt_iter * s._pop_size, np.abs(corr).max(),
                          np.linalg.norm(corr), psi_share))


CONFIGS = [
    ('CMA-ES korr., Intervall 1', CMAESFixed, {'update_interval': 1}),
    ('DLR fester Rang 1, Intervall 1', DLRCMAES,
     {'low_rank_is_adaptive': False, 'low_rank_init_dimension': 1,
      'update_interval': 1}),
    ('DLR fester Rang 25, Intervall 1', DLRCMAES,
     {'low_rank_is_adaptive': False, 'low_rank_init_dimension': 25,
      'update_interval': 1}),
    ('DLR voller Rang, Intervall 1', DLRCMAES,
     {'low_rank_is_adaptive': False, 'low_rank_init_dimension': N,
      'update_interval': 1}),
    ]


def main(fname, seed):
    problem = FUNCTIONS[fname](N)
    lower, upper = map(np.asarray, problem.bounds)
    sigma0 = SIGMA_FRACTION * float(np.mean(upper - lower))
    print(f'{fname}, n={N}, seed={seed}')
    for name, cls, kw in CONFIGS:
        np.random.seed(seed)
        _seed_numba(seed)
        x0 = np.random.default_rng(10_000 + seed).uniform(lower, upper)
        log = Corr()
        s = cls(number_of_variables=N, objective=problem.__call__,
                initial_sigma=sigma0,
                maximum_iterations=int(1e6 / default_pop(N)),
                maximum_wall_time=1800, fitness_threshold=TARGET,
                tolerance=0.0, sigma_threshold=1e-14,
                min_log_level='critical', callback=log, random_state=seed,
                **kw)
        res = s.optimize(x0)
        r = np.array(log.rows)
        print(f'\n{name}: Evals={res["iterations"] * 17}, '
              f'Erfolg={int(res["optimal_value"] < TARGET)}')
        print(f'{"Evals":>8s} {"|corr|max":>9s} {"corr_F":>7s} {"psi%":>6s}')
        for i in sorted(set(np.linspace(0, len(r) - 1, 6).astype(int))):
            print(f'{int(r[i, 0]):8d} {r[i, 1]:9.3f} {r[i, 2]:7.3f} '
                  f'{100 * r[i, 3]:6.1f}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'Ellipsoid',
         int(sys.argv[2]) if len(sys.argv) > 2 else 0)
