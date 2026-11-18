"""Diagnosis: why is DLR at full rank worse than CMA-ES on Rotated Ellipsoid?

Logs every few iterations, for corrected CMA-ES and DLR (full rank and
adaptive rank, both with update interval 1):

- Lanczos accuracy: relative error of the library's C^(-1/2) y with
  y = C^(1/2) z (exact answer z),
- how well C matches the inverse Hessian: cond(A^(1/2) C A^(1/2)),
- psi share tr(diag psi)/tr(C) and min(psi)/lambda_min(C),
- |p_sigma| / E||N(0,I)||.

Usage:
    python diag_rotell.py [function] [seed]
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
from seamaze.optimizers import DLRCMAES
from seamaze.optimizers._dlrcmaes import _lanczos_inverse_sqrt_product
from cmaes_path_variant import CMAES as CMAESFixed

N = 100
EVERY = 10
OUT = ROOT / 'results' / 'diagnostics'


def hessian_root(fname):
    """Matrix B with f(x) = ||B x||^2 (up to a constant)."""
    if fname == 'Rotated Ellipsoid':
        return np.tril(np.ones((N, N)))
    if fname == 'Ellipsoid':
        return np.diag(np.sqrt(10.0 ** (6.0 * np.arange(N) / (N - 1.0))))
    raise ValueError(fname)


def cov_of(s):
    if hasattr(s, '_cov'):
        return s._cov.copy(), None
    c = (s._basis * s._core) @ s._basis.T + np.diag(s._psi)
    return c, s._psi


class Logger:
    def __init__(self, broot, rng):
        self.broot, self.rng, self.rows = broot, rng, []

    def __call__(self, s):
        if s._opt_iter % EVERY:
            return
        c, psi = cov_of(s)
        c = 0.5 * (c + c.T)
        ev, vec = np.linalg.eigh(c)
        ev = np.maximum(ev, 1e-300)
        z = self.rng.standard_normal(N)
        y = vec @ (np.sqrt(ev) * (vec.T @ z))
        if psi is None:
            lanczos_err = np.nan
            psi_share = psi_floor = np.nan
        else:
            approx = _lanczos_inverse_sqrt_product(
                np.ascontiguousarray(y), np.asfortranarray(s._basis),
                s._core.copy(), s._psi.copy())
            lanczos_err = np.linalg.norm(approx - z) / np.linalg.norm(z)
            psi_share = psi.sum() / np.trace(c)
            psi_floor = psi.min() / ev[0]
        m = self.broot @ c @ self.broot.T
        mev = np.linalg.eigvalsh(0.5 * (m + m.T))
        cond_fit = mev[-1] / max(mev[0], 1e-300)
        ps = np.linalg.norm(s._path_sigma) / s._expected_path_length
        self.rows.append((s._opt_iter * s._pop_size, lanczos_err, cond_fit,
                          psi_share, psi_floor, ps, s._sigma,
                          getattr(s, 'rank', N)))


CONFIGS = [
    ('CMA-ES korr., Intervall 1', CMAESFixed, {'update_interval': 1},
     '#eb6834', '-'),
    ('DLR voller Rang, Intervall 1', DLRCMAES,
     {'low_rank_is_adaptive': False, 'low_rank_init_dimension': N,
      'update_interval': 1}, '#2a78d6', '-'),
    ('DLR adaptiv, Intervall 1', DLRCMAES, {'update_interval': 1},
     '#2a78d6', '--'),
    ]


def run(fname, seed, cls, kwargs):
    problem = FUNCTIONS[fname](N)
    lower, upper = map(np.asarray, problem.bounds)
    sigma0 = SIGMA_FRACTION * float(np.mean(upper - lower))
    np.random.seed(seed)
    _seed_numba(seed)
    x0 = np.random.default_rng(10_000 + seed).uniform(lower, upper)
    log = Logger(hessian_root(fname), np.random.default_rng(999))
    solver = cls(
        number_of_variables=N, objective=problem.__call__,
        initial_sigma=sigma0, maximum_iterations=int(1e6 / default_pop(N)),
        maximum_wall_time=900, fitness_threshold=TARGET, tolerance=0.0,
        sigma_threshold=1e-14, min_log_level='critical', callback=log,
        random_state=seed, **kwargs)
    res = solver.optimize(x0)
    return np.array(log.rows), res


def main(fname, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    panels = [(2, 'cond(A^½ C A^½): Abstand von C zur inversen Hesse-Matrix',
               True),
              (1, 'Lanczos: relativer Fehler von C^(-1/2) y', True),
              (5, '|p_σ| / E|N(0,I)|', False),
              (3, 'ψ-Anteil an tr(C)', False)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    print(f'{fname}, n={N}, seed={seed}')
    for name, cls, kwargs, color, ls in CONFIGS:
        d, res = run(fname, seed, cls, kwargs)
        ev = d[:, 0]
        ok = res['optimal_value'] < TARGET
        q = slice(len(d) // 4, None)
        print(f'{name:30s} Evals={res["iterations"] * 17:8d} Erfolg={int(ok)}'
              f' | Lanczos-Fehler Median={np.nanmedian(d[q, 1]):.2e}'
              f' max={np.nanmax(d[:, 1]) if np.isfinite(d[:, 1]).any() else np.nan:.2e}'
              f' | cond-fit Ende={d[-1, 2]:.2e}'
              f' | |p_s|/E Median={np.median(d[q, 5]):.3f}'
              f' | psi-Anteil Median={np.nanmedian(d[q, 3]):.3f}'
              f' | min psi/lambda_min Median={np.nanmedian(d[q, 4]):.3f}'
              f' | Rang Ende={int(d[-1, 7])}', flush=True)
        for ax, (col, _, _) in zip(axes.ravel(), panels):
            ax.plot(ev, d[:, col], color=color, linestyle=ls, linewidth=1.8,
                    label=name)
    for ax, (_, title, logy) in zip(axes.ravel(), panels):
        if logy:
            ax.set_yscale('log')
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_xlabel('Evaluierungen', fontsize=9)
        ax.grid(color='#e4e3dd', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'{fname}, n = {N}, Seed {seed}', x=0.01, ha='left',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / f'rotell_{fname.replace(" ", "_")}_s{seed}.png'
    fig.savefig(path, dpi=150)
    print('Plot:', path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'Rotated Ellipsoid',
         int(sys.argv[2]) if len(sys.argv) > 2 else 0)
