"""Diagnosis: why does the adaptive rank grow to ~60 on the axis-parallel
Ellipsoid, although rank 1 is enough?

A subclass of the library DLRCMAES re-implements `tell` unchanged, but
before each covariance update it recomputes the rank-selection quantities
of `_adaptive_bug_step` in Python (same inputs) and logs them:

- rank proposal of the energy criterion (library setting) and
  counterfactual proposals for other filter thresholds / tolerances,
- spectrum of the whitened off-diagonal core (largest values, count of
  values surviving the soft threshold),
- how non-diagonal the full covariance C really is: largest |correlation|
  and Frobenius norm of the correlation matrix minus I,
- psi share of tr(C).

The optimization trajectory is identical to the library (the Python
recomputation only uses numpy's global RNG, which the solver does not use
after initialization).

Usage:
    python diag_rank.py [function] [seed]
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
from numpy import argsort

from study import (FUNCTIONS, SIGMA_FRACTION, TARGET, _seed_numba,
                   default_pop)
from seamaze.optimizers import DLRCMAES
from seamaze.optimizers._dlrcmaes import _adaptive_bug_step, _tell

N = 100
OUT = ROOT / 'results' / 'diagnostics'
THRESHOLDS = [0.0, 0.02, 0.05, 0.1, 0.2]
TOLERANCES = [1e-2, 1e-1, 3e-1]


def energy_rank(values, fraction):
    values = np.maximum(values, 0.0)
    total = values.sum()
    if total <= 1e-15:
        return 1
    cum = np.cumsum(values)
    return max(1, int(np.argmax(cum >= fraction * total)) + 1)


def rank_quantities(basis, core, psi, steps, sorting, weights, path_cov,
                    lr_cov, lr_rank_one, lr_rank_mu, update_switch):
    """Python copy of the rank-selection part of _adaptive_bug_step."""
    steps_sorted = steps[sorting]
    dim, rank = basis.shape
    aug_size = min(2 * rank, dim) - rank
    w = weights.ravel().copy()
    neg = np.where(w < 0.0)[0]
    if neg.size:
        sn = steps_sorted[neg]
        inv_psi = 1.0 / np.maximum(psi, 1e-15)
        ipb = inv_psi[:, None] * basis
        wood = np.diag(1.0 / np.maximum(core, 1e-15)) + basis.T @ ipb
        ips = inv_psi[:, None] * sn.T
        rhs = basis.T @ ips
        corr = np.linalg.solve(wood, rhs)
        qf = np.maximum(np.sum(sn.T * ips, axis=0) - np.sum(rhs * corr, 0), 0)
        w[neg] *= np.minimum(1.0, dim / (qf + 1e-15))
    w2 = w.reshape(-1, 1)
    adj = (1.0 - update_switch) * lr_rank_one * lr_cov * (2.0 - lr_cov)
    decay = lr_rank_one + lr_rank_mu - adj
    diag_upd = (lr_rank_one * path_cov**2
                + lr_rank_mu * np.sum(w2 * steps_sorted**2, axis=0)
                + (1.0 - decay) * (np.sum(basis**2 * core, axis=1) + psi))
    k = (lr_rank_one * path_cov[:, None] * (path_cov @ basis)
         + lr_rank_mu * steps_sorted.T @ (w2 * (steps_sorted @ basis))
         + (1.0 - decay) * (basis * core))
    q, _ = np.linalg.qr(k)
    if aug_size > 0:
        aug = np.zeros((dim, aug_size))
        aug[:, 0] = path_cov - q @ (q.T @ path_cov)
        for i in range(1, aug_size):
            cand = (steps_sorted[i - 1] if i - 1 < steps_sorted.shape[0]
                    else np.random.randn(dim))
            aug[:, i] = cand - q @ (q.T @ cand)
        for _ in range(2):
            aug -= q @ (q.T @ aug)
        qa, _ = np.linalg.qr(aug)
        u = np.hstack((q, qa))
    else:
        u = q
    pa = u.T @ path_cov
    sa = steps_sorted @ u
    bp = u.T @ basis
    shat = (lr_rank_one * np.outer(pa, pa) + lr_rank_mu * sa.T @ (w2 * sa)
            + (1.0 - decay) * (bp * core) @ bp.T)
    shat = 0.5 * (shat + shat.T)
    daug = (u.T * diag_upd) @ u
    ev, vec = np.linalg.eigh(0.5 * (daug + daug.T))
    isq = (vec * (1.0 / np.sqrt(np.maximum(ev, 1e-15)))) @ vec.T
    sw = isq @ shat @ isq
    sw = 0.5 * (sw + sw.T)
    off = sw - np.diag(np.diag(sw))
    s_off = np.linalg.eigvalsh(0.5 * (off + off.T))
    out = {'rank': rank, 'top': np.sort(np.abs(s_off))[::-1][:5],
           'diag_mean': float(np.mean(np.diag(sw)))}
    for thr in THRESHOLDS:
        filt = np.sort(np.maximum(np.abs(s_off) - thr, 0.0))[::-1]
        out[f'n_survive_{thr}'] = int(np.sum(filt > 0))
        for tol in TOLERANCES:
            out[f'prop_{thr}_{tol}'] = energy_rank(filt, 1.0 - tol)
    return out


class InstrumentedDLR(DLRCMAES):
    """Library DLRCMAES with an unchanged tell plus logging."""

    log = None

    def tell(self, fitness):
        sorting = argsort(fitness)
        (ps, mean, sigma, pc, switch) = _tell(
            self._steps, sorting, self._weights, self._basis, self._core,
            self._psi, self._path_sigma, self._path_cov, self._mean,
            self._sigma, self._lr_sigma, self._lr_cov, self._lr_mean,
            self._mu_eff, self._damp_sigma, self._expected_path_length,
            self._opt_iter, self._elite_size)
        self._path_sigma[:] = ps
        self._mean[:] = mean
        self._sigma = sigma
        self._path_cov[:] = pc
        if self._opt_iter % self._update_interval == 0:
            state = np.random.get_state()
            q = rank_quantities(
                self._basis.copy(), self._core.copy(), self._psi.copy(),
                self._steps.copy(), sorting, self._weights, pc.copy(),
                self._lr_cov, self._lr_rank_one, self._lr_rank_mu, switch)
            np.random.set_state(state)
            c = (self._basis * self._core) @ self._basis.T + np.diag(
                self._psi)
            d = np.sqrt(np.diag(c))
            corr = c / np.outer(d, d) - np.eye(len(d))
            q['corr_max'] = float(np.abs(corr).max())
            q['corr_fro'] = float(np.linalg.norm(corr))
            q['psi_share'] = float(self._psi.sum() / np.trace(c))
            q['evals'] = self._opt_iter * self._pop_size
            self.log.append(q)
            rank_now = self.rank
            b, co, p, r = _adaptive_bug_step(
                self._basis, self._core, self._psi, self._steps, sorting,
                self._weights, self._path_cov, self._lr_cov,
                self._lr_rank_one, self._lr_rank_mu,
                self._low_rank_max_dimension, self._low_rank_is_adaptive,
                self._low_rank_energy_tolerance, update_switch=switch)
            self._basis, self._core = b, co
            self._psi[:] = p
            self.rank = r
            if r != rank_now:
                (self._lr_cov, self._lr_rank_one,
                 self._lr_rank_mu) = self._get_learning_rates()
                if self._update_interval_user is None:
                    self._update_interval = self._get_update_interval()


def run(fname, seed):
    problem = FUNCTIONS[fname](N)
    lower, upper = map(np.asarray, problem.bounds)
    sigma0 = SIGMA_FRACTION * float(np.mean(upper - lower))
    np.random.seed(seed)
    _seed_numba(seed)
    x0 = np.random.default_rng(10_000 + seed).uniform(lower, upper)
    solver = InstrumentedDLR(
        number_of_variables=N, objective=problem.__call__,
        initial_sigma=sigma0, maximum_iterations=int(1e6 / default_pop(N)),
        maximum_wall_time=1800, fitness_threshold=TARGET, tolerance=0.0,
        sigma_threshold=1e-14, min_log_level='critical', random_state=seed,
        update_interval=1)
    solver.log = []
    res = solver.optimize(x0)
    return solver.log, res


def main(fname, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    log, res = run(fname, seed)
    ev = np.array([q['evals'] for q in log])
    rank = np.array([q['rank'] for q in log])
    print(f'{fname}, n={N}, seed={seed}, adaptiv, Intervall 1: '
          f'Evals={res["iterations"] * 17}, Erfolg='
          f'{int(res["optimal_value"] < TARGET)}, Rang Ende={rank[-1]}')
    marks = sorted(set(np.linspace(0, len(log) - 1, 7).astype(int)))
    print(f'{"Evals":>8s} {"Rang":>4s} {"|corr|max":>9s} {"corr_F":>7s} '
          f'{"psi%":>5s} {"top|s_off|":>24s} '
          + ' '.join(f'n>{t:g}' for t in THRESHOLDS))
    for i in marks:
        q = log[i]
        top = ' '.join(f'{v:.3f}' for v in q['top'][:4])
        surv = ' '.join(f'{q[f"n_survive_{t}"]:5d}' for t in THRESHOLDS)
        print(f'{q["evals"]:8d} {q["rank"]:4d} {q["corr_max"]:9.3f} '
              f'{q["corr_fro"]:7.3f} {100 * q["psi_share"]:5.1f} '
              f'{top:>24s} {surv}')
    print('\nMedian der Rang-Vorschläge (2. Hälfte des Laufs), '
          'Zeilen = Filter-Schwelle, Spalten = Toleranz')
    half = log[len(log) // 2:]
    print('Schwelle ' + ' '.join(f'tol={t:g}'.rjust(9) for t in TOLERANCES))
    for thr in THRESHOLDS:
        vals = [int(np.median([q[f'prop_{thr}_{t}'] for q in half]))
                for t in TOLERANCES]
        print(f'{thr:8g} ' + ' '.join(f'{v:9d}' for v in vals))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(ev, rank, color='#2a78d6', linewidth=2)
    axes[0].set_title('Rang', loc='left', fontsize=10)
    axes[1].plot(ev, [q['corr_max'] for q in log], color='#eb6834',
                 linewidth=1.5, label='max |Korrelation|')
    axes[1].plot(ev, [q['corr_fro'] for q in log], color='#1baf7a',
                 linewidth=1.5, label='‖Korrelation − I‖_F')
    axes[1].set_title('Wie stark ist C wirklich nicht-diagonal?',
                      loc='left', fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    for t, col in zip(THRESHOLDS, ['#6b6a63', '#2a78d6', '#eda100',
                                   '#e87ba4', '#008300']):
        axes[2].plot(ev, [q[f'prop_{t}_0.01'] for q in log], color=col,
                     linewidth=1.5, label=f'Schwelle {t:g}')
    axes[2].set_title('Rang-Vorschlag (tol = 0,01) je Filter-Schwelle',
                      loc='left', fontsize=10)
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel('Evaluierungen', fontsize=9)
        ax.grid(color='#e4e3dd', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle(f'Rangwahl auf {fname}, n = {N}, Seed {seed} '
                 '(adaptiv, Intervall 1)', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT / f'rank_{fname.replace(" ", "_")}_s{seed}.png'
    fig.savefig(path, dpi=150)
    print('Plot:', path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'Ellipsoid',
         int(sys.argv[2]) if len(sys.argv) > 2 else 0)
