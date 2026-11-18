"""Measured noise of the learned covariance vs. the supervisor's formulas.

Idea: run the solvers where the ideal covariance is known to be isotropic,
so every off-diagonal correlation is pure estimation noise:
- 'Zufall': random fitness values (= random selection, no information)
- 'Sphere': real selection, optimal covariance proportional to I
Every EVERY iterations the correlation matrix R = C / sqrt(diag C diag C^T)
is formed and the RMS of its off-diagonal entries is recorded (this is the
per-entry noise std eps, since the true off-diagonal values are 0).
All runs use update_interval=1, so k (covariance updates) = iterations.

Predictions (formulas from the supervisor's mail, parameters read from the
solver: c1=_lr_rank_one, cmu=_lr_rank_mu, cc=_lr_cov, w=_weights):
- noise_age(k) with beta = 1 - c1 - cmu   (decay factor used in seamaze)
- noise_age(k) with beta = 1 - c1 - cmu*sum(w)   (beta from the mail; = 1
  with seamaze's negative weights, i.e. noise grows like sqrt(k))
- noise_stationary with beta = 1 - c1 - cmu   (limit of the first line)

Additionally recorded: largest |eigenvalue| of the off-diagonal correlation
matrix (spectral noise size, rough reference 2*eps*sqrt(n)), DLR rank and
the psi share tr(psi)/tr(C).

Usage (from studies/parameter_study):
    python diagnostics/diag_noise.py [n_seeds] [workers] [iterations] [settings]
    settings: comma list of Zufall, Sphere, Ellipsoid (default Zufall,Sphere;
    Ellipsoid stops at f < 1e-8)
Output: results/diagnostics/noise/ (<tag>.csv, <tag>_table.txt, <tag>.png,
<tag>_dlr.png; tag = 'noise' for the default, else 'noise_<settings>')
"""

import os

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(ROOT), str(ROOT / 'variants')]

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

N = 100
EVERY = 25
OUT = ROOT / 'results' / 'diagnostics' / 'noise'

CONFIGS = {
    'CMA-ES korr.': ('cmaes_fixed', {}),
    'DLR voller Rang': ('dlr', {'low_rank_is_adaptive': False,
                                'low_rank_init_dimension': N}),
    'DLR fester Rang 10': ('dlr', {'low_rank_is_adaptive': False,
                                   'low_rank_init_dimension': 10}),
    'DLR adaptiv (Standard)': ('dlr', {}),
    }
SETTINGS = ['Zufall', 'Sphere']          # default; override on the CLI
TITLES = {'Zufall': 'Zufällige Selektion (keine Information)',
          'Sphere': 'Sphere (echte Selektion, Ziel ∝ I)',
          'Ellipsoid': 'Ellipsoid (achsenparallel, Zielkorrelation 0)'}
TAG = 'noise'                             # output file prefix


# --- Formulas from the mail (unchanged) -----------------------------------

def noise_stationary(c1, cmu, cc, w, beta=None):
    """Stationary per-entry noise std eps of the learned covariance
    (AR(1) path)."""
    if beta is None:
        beta = 1.0 - c1 - cmu
    rho = (1.0 - cc)**2
    a = rho * beta
    with np.errstate(divide='ignore'):
        return np.sqrt((cmu**2 * np.sum(w**2)
                        + c1**2 * (1.0 + a) / (1.0 - a)) / (1.0 - beta**2))


def noise_age(c1, cmu, cc, w, beta, k):
    """Per-entry noise std eps_k after k updates (V_0 = R_0 = Q_0 = 0)."""
    rho = (1.0 - cc)**2
    V = R = Q = 0.0
    for _ in range(k):
        V, R, Q = (beta**2 * V + 1.0,
                   beta**2 * R + 1.0 + 2.0 * beta * rho * Q,
                   1.0 + rho * beta * Q)
    return np.sqrt(cmu**2 * np.sum(w**2) * V + c1**2 * R)


def noise_age_curve(c1, cmu, cc, w, beta, k_max):
    """noise_age for k = 0..k_max in one pass (same recursion)."""
    rho = (1.0 - cc)**2
    V = R = Q = 0.0
    out = [0.0]
    for _ in range(k_max):
        V, R, Q = (beta**2 * V + 1.0,
                   beta**2 * R + 1.0 + 2.0 * beta * rho * Q,
                   1.0 + rho * beta * Q)
        out.append(np.sqrt(cmu**2 * np.sum(w**2) * V + c1**2 * R))
    return np.array(out)


def solver_params(s):
    w = np.ravel(s._weights)
    return s._lr_rank_one, s._lr_rank_mu, s._lr_cov, w


# --- Measurement ----------------------------------------------------------

class NoiseLog:
    """Logs the noise measures every EVERY iterations. For DLR it also
    counts, every iteration, the updates in which the low-rank core was
    scaled down globally (alpha < 1 in _adaptive_bug_step): then psi is
    exactly 0 at the coordinate that set alpha, since
    psi_new = max(diag_update - alpha * diag_lowrank, 0)."""

    def __init__(self):
        self.rows = []
        self.alpha_events = 0

    def __call__(self, s):
        if hasattr(s, '_psi') and s._psi.min() <= 1e-12 * s._psi.mean():
            self.alpha_events += 1
        if s._opt_iter % EVERY:
            return
        if hasattr(s, '_cov'):
            c, psi_share, rank = s._cov, np.nan, N
        else:
            c = (s._basis * s._core) @ s._basis.T + np.diag(s._psi)
            psi_share, rank = s._psi.sum() / np.trace(c), s.rank
        d = np.sqrt(np.diag(c))
        off = c / np.outer(d, d) - np.eye(N)      # zero diagonal
        iu = np.triu_indices(N, 1)
        rms = np.sqrt(np.mean(off[iu]**2))
        lam = np.abs(np.linalg.eigvalsh(off)).max()
        self.rows.append({'k': s._opt_iter, 'rms_offdiag': rms,
                          'lam_max_offdiag': lam, 'rank': rank,
                          'psi_share': psi_share, 'sigma': s._sigma,
                          'alpha_events': self.alpha_events,
                          'f_best': s._result['optimal_value']})


def run_one(task):
    label, setting, seed, iterations = task
    from study import (FUNCTIONS, SIGMA_FRACTION, TARGET, _seed_numba,
                       _solver_class)
    solver_key, kw = CONFIGS[label]
    problem = FUNCTIONS['Sphere' if setting == 'Zufall' else setting](N)
    lower, upper = map(np.asarray, problem.bounds)
    sigma0 = SIGMA_FRACTION * float(np.mean(upper - lower))
    # Ellipsoid: stop at the study target (the interesting phase is the
    # adaptation before it); Zufall/Sphere: fixed number of iterations
    threshold = TARGET if setting == 'Ellipsoid' else -np.inf
    if setting != 'Zufall':
        objective = problem.__call__
    else:
        frng = np.random.default_rng(20_000 + seed)

        def objective(x):
            return float(frng.random())
    np.random.seed(seed)
    _seed_numba(seed)
    x0 = np.random.default_rng(10_000 + seed).uniform(lower, upper)
    log = NoiseLog()
    s = _solver_class(solver_key)(
        number_of_variables=N, objective=objective, initial_sigma=sigma0,
        maximum_iterations=iterations, maximum_wall_time=3600,
        fitness_threshold=threshold, tolerance=0.0, sigma_threshold=0.0,
        update_interval=1, min_log_level='critical', callback=log,
        random_state=seed, **kw)
    s.optimize(x0)
    c1, cmu, cc, w = solver_params(s)
    rows = pd.DataFrame(log.rows)
    rows.insert(0, 'seed', seed)
    rows.insert(0, 'setting', setting)
    rows.insert(0, 'label', label)
    params = {'label': label, 'c1': c1, 'cmu': cmu, 'cc': cc,
              'sum_w': w.sum(), 'sum_w2': np.sum(w**2),
              'sum_wpos2': np.sum(w[w > 0]**2), 'w': w}
    return rows, params


# --- Output ---------------------------------------------------------------

def predictions(p, k_max):
    w = p['w']
    beta_code = 1.0 - p['c1'] - p['cmu']
    beta_mail = 1.0 - p['c1'] - p['cmu'] * w.sum()
    return {
        'beta_code': beta_code, 'beta_mail': beta_mail,
        'age_code': noise_age_curve(p['c1'], p['cmu'], p['cc'], w,
                                    beta_code, k_max),
        'age_mail': noise_age_curve(p['c1'], p['cmu'], p['cc'], w,
                                    beta_mail, k_max),
        'stat_code': noise_stationary(p['c1'], p['cmu'], p['cc'], w),
        }


def write_table(df, pred, k_max):
    ks = sorted({k for k in (100, 250, 500, 1000, 2000, 3000, 5000, 7500,
                             10000, 15000, 20000, 30000, k_max)
                 if k <= k_max and k % EVERY == 0})
    lines = [
        f'Rauschen der gelernten Kovarianz, n={N}, Intervall 1, '
        f'{df.seed.nunique()} Seeds',
        f'beta_code = 1 - c1 - cmu        = {pred["beta_code"]:.6f}',
        f'beta_mail = 1 - c1 - cmu*sum(w) = {pred["beta_mail"]:.6f}',
        f'noise_stationary(beta_code)     = {pred["stat_code"]:.4f}   '
        f'(2*eps*sqrt(n) = {2 * pred["stat_code"] * np.sqrt(N):.3f})',
        '',
        'Gemessen: RMS der Nebendiagonal-Korrelationen (Median über Seeds,'
        ' in Klammern min-max); lam = größter |Eigenwert| der',
        'Nebendiagonal-Korrelationsmatrix (Median).', '']
    for setting in SETTINGS:
        lines.append(setting)
        head = f'  {"k":>6s} {"Formel beta_code":>16s} {"Formel beta_mail":>16s}'
        for lab in CONFIGS:
            head += f' | {lab:>26s}'
        lines.append(head)
        for k in ks:
            line = (f'  {k:6d} {pred["age_code"][k]:16.4f} '
                    f'{pred["age_mail"][k]:16.4f}')
            for lab in CONFIGS:
                d = df[(df.setting == setting) & (df.label == lab)
                       & (df.k == k)]
                if d.empty:
                    line += f' | {"-":>26s}'
                    continue
                cell = (f'{d.rms_offdiag.median():.4f} '
                        f'({d.rms_offdiag.min():.4f}-'
                        f'{d.rms_offdiag.max():.4f}) '
                        f'lam {d.lam_max_offdiag.median():.2f}')
                line += f' | {cell:>26s}'
            lines.append(line)
        sub = df[df.setting == setting]
        last = sub.loc[sub.groupby(['label', 'seed']).k.idxmax()]
        lines.append('  Ende je Konfiguration (Median über Seeds):')
        for lab, g in last.groupby('label', sort=False):
            run = sub[sub.label == lab]
            txt = (f'    {lab:24s} k={g.k.median():6.0f}  '
                   f'Rang Ende {g["rank"].median():3.0f}  '
                   f'Rang max {run.groupby("seed")["rank"].max().median():3.0f}')
            if np.isfinite(g.psi_share.median()):
                share = 100 * g.alpha_events.median() / g.k.median()
                txt += (f'  psi {100 * g.psi_share.median():3.0f} %  '
                        f'alpha<1 in {share:4.1f} % der Updates')
            lines.append(txt)
        lines.append('')
    text = '\n'.join(lines)
    (OUT / f'{TAG}_table.txt').write_text(text, encoding='utf-8')
    print(text)


def plot(df, pred, k_max):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ink, muted, grid = '#1f1f1e', '#6b6a63', '#e4e3dd'
    colors = {'CMA-ES korr.': '#eb6834', 'DLR voller Rang': '#2a78d6',
              'DLR fester Rang 10': '#eda100',
              'DLR adaptiv (Standard)': '#1baf7a'}
    titles = TITLES
    fig, axes = plt.subplots(1, len(SETTINGS),
                             figsize=(5.5 * max(2, len(SETTINGS)), 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    kk = np.arange(k_max + 1)
    for ax, setting in zip(axes, SETTINGS):
        for lab, col in colors.items():
            d = df[(df.setting == setting) & (df.label == lab)]
            if d.empty:
                continue
            g = d.groupby('k').rms_offdiag
            med, lo, hi = g.median(), g.min(), g.max()
            ax.fill_between(med.index, lo, hi, color=col, alpha=0.15,
                            linewidth=0)
            ax.plot(med.index, med, color=col, linewidth=2, label=lab)
        ax.plot(kk[1:], pred['age_code'][1:], color=ink, linestyle='--',
                linewidth=1.5, label='Formel noise_age, β = 1 − c1 − cμ')
        ax.plot(kk[1:], pred['age_mail'][1:], color=muted, linestyle=':',
                linewidth=1.8,
                label=f'Formel noise_age, β = 1 − c1 − cμ·Σw '
                      f'(= {pred["beta_mail"]:.4f})')
        ax.axhline(pred['stat_code'], color=ink, linewidth=0.8,
                   alpha=0.5)
        ax.annotate(f'noise_stationary = {pred["stat_code"]:.3f}',
                    (k_max, pred['stat_code']), xytext=(-4, 4),
                    textcoords='offset points', ha='right', fontsize=7,
                    color=muted)
        ax.set_yscale('log')
        ax.set_xlabel('Kovarianz-Updates k', fontsize=8, color=muted)
        ax.set_title(titles[setting], loc='left', fontsize=10, color=ink)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].set_ylabel('RMS Nebendiagonal-Korrelation', fontsize=8,
                       color=muted)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(f'Gemessenes Rauschen der Kovarianz gegen Formel '
                 f'(n = {N}, Intervall 1, {df.seed.nunique()} Seeds)',
                 x=0.01, ha='left', fontsize=11, color=ink)
    fig.text(0.01, 0.01, 'Linie: Median über Seeds, Fläche: min–max. '
             'Gemessen wird die Korrelationsmatrix, die Diagonale ist '
             'herausgerechnet.', fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 0.8))
    fig.savefig(OUT / f'{TAG}.png', dpi=150)
    plt.close(fig)

    # DLR internals: rank and psi share over k
    fig, axes = plt.subplots(len(SETTINGS), 2,
                             figsize=(11, 3.4 * len(SETTINGS)),
                             squeeze=False)
    for row, setting in zip(axes, SETTINGS):
        for ax, col, ylab in ((row[0], 'rank', 'Rang'),
                              (row[1], 'psi_share', 'Anteil ψ an tr(C)')):
            for lab, c in colors.items():
                if lab == 'CMA-ES korr.':
                    continue
                d = df[(df.setting == setting) & (df.label == lab)]
                if d.empty:
                    continue
                g = d.groupby('k')[col]
                ax.fill_between(g.median().index, g.min(), g.max(), color=c,
                                alpha=0.15, linewidth=0)
                ax.plot(g.median().index, g.median(), color=c, linewidth=2,
                        label=lab)
            ax.set_title(f'{titles[setting]}: {ylab}', loc='left',
                         fontsize=10, color=ink)
            ax.set_xlabel('Kovarianz-Updates k', fontsize=8, color=muted)
            ax.grid(color=grid, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=muted, labelsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f'{TAG}_dlr.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    if len(sys.argv) > 4:
        SETTINGS = sys.argv[4].split(',')
        TAG = 'noise_' + '_'.join(s.lower() for s in SETTINGS)
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(lab, setting, seed, iterations) for lab in CONFIGS
             for setting in SETTINGS for seed in range(n_seeds)]
    print(f'{len(tasks)} Läufe, n={N}, {iterations} Iterationen', flush=True)
    frames, params = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (rows, p) in enumerate(pool.map(run_one, tasks), 1):
            frames.append(rows)
            params.append(p)
            print(f'[{i}/{len(tasks)}] {rows.label.iloc[0]:24s} '
                  f'{rows.setting.iloc[0]:7s} s={rows.seed.iloc[0]} '
                  f'RMS am Ende={rows.rms_offdiag.iloc[-1]:.4f}', flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / f'{TAG}.csv', index=False)
    pdf = (pd.DataFrame([{k: v for k, v in p.items() if k != 'w'}
                         for p in params]).drop_duplicates('label'))
    pdf.to_csv(OUT / f'{TAG}_params.csv', index=False)
    print(pdf.to_string(index=False))
    # CMA-ES and DLR use the same learning-rate and weight formulas (see
    # noise_params.csv); the formulas use the parameters of the first run.
    pred = predictions(params[0], int(df.k.max()))
    write_table(df, pred, int(df.k.max()))
    plot(df, pred, int(df.k.max()))
