"""Diagnosis: why are intermediate fixed ranks worse than rank 1 and full
rank on the axis-parallel Ellipsoid?

Logs for fixed rank 1, 25 and 100 (interval 1):
- cond(A^1/2 C A^1/2): overall distance of C to the inverse Hessian,
- diagonal mismatch: std of log(diag(C) * coef) (0 = diagonal perfectly
  adapted up to scale),
- psi share of tr(C).

Usage:
    python diag_midrank.py [seed]
"""

import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(ROOT), str(ROOT / 'variants')]

import sys

import numpy as np

from diag_rotell import cov_of, hessian_root
from study import (FUNCTIONS, SIGMA_FRACTION, TARGET, _seed_numba,
                   default_pop)
from seamaze.optimizers import DLRCMAES

N = 100
EVERY = 200
COEF = 10.0 ** (6.0 * np.arange(N) / (N - 1.0))


class Log:
    def __init__(self):
        self.broot, self.rows = hessian_root('Ellipsoid'), []

    def __call__(self, s):
        if s._opt_iter % EVERY:
            return
        c, psi = cov_of(s)
        m = self.broot @ c @ self.broot.T
        mev = np.linalg.eigvalsh(0.5 * (m + m.T))
        mismatch = np.std(np.log10(np.diag(c) * COEF))
        self.rows.append((s._opt_iter * s._pop_size,
                          mev[-1] / max(mev[0], 1e-300), mismatch,
                          psi.sum() / np.trace(c)))


def main(seed):
    problem = FUNCTIONS['Ellipsoid'](N)
    lower, upper = map(np.asarray, problem.bounds)
    print(f'Ellipsoid, n={N}, seed={seed}, fester Rang, Intervall 1')
    print('cond = cond(A^1/2 C A^1/2); diag-Fehler = Std von '
          'log10(diag(C)*coef) in Dekaden (0 = perfekt)')
    for r in (1, 25, N):
        np.random.seed(seed)
        _seed_numba(seed)
        x0 = np.random.default_rng(10_000 + seed).uniform(lower, upper)
        log = Log()
        s = DLRCMAES(
            number_of_variables=N, objective=problem.__call__,
            initial_sigma=SIGMA_FRACTION * float(np.mean(upper - lower)),
            maximum_iterations=int(1e6 / default_pop(N)),
            maximum_wall_time=1800, fitness_threshold=TARGET, tolerance=0.0,
            sigma_threshold=1e-14, min_log_level='critical', callback=log,
            random_state=seed, low_rank_is_adaptive=False,
            low_rank_init_dimension=r, update_interval=1)
        res = s.optimize(x0)
        d = np.array(log.rows)
        print(f'\nRang {r}: Evals={res["iterations"] * 17}, '
              f'Erfolg={int(res["optimal_value"] < TARGET)}')
        print(f'{"Evals":>8s} {"cond":>9s} {"diag-Fehler":>11s} {"psi%":>6s}')
        for i in sorted(set(np.linspace(0, len(d) - 1, 7).astype(int))):
            print(f'{int(d[i, 0]):8d} {d[i, 1]:9.2e} {d[i, 2]:11.3f} '
                  f'{100 * d[i, 3]:6.1f}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
