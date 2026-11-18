"""Stage 3: scaling over the dimension n.

Reads results/stage3/stage3.csv (all seeds, parallel) and, if present,
results/stage3/stage3_timing.csv (seed 0, sequential, clean wall times).
Writes to results/stage3/:
- stage3_summary.csv, stage3_table.txt
- plots/evals.png      evaluations to target over n
- plots/time.png       wall time to target over n
- plots/iter_cost.png  milliseconds per iteration over n

Usage (from studies/parameter_study):
    python analysis/analyze_stage3.py [--base <folder with stage3.csv>]
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
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

CONFIGS = ['tol=0.01, interval=auto', 'tol=0.05, interval=auto',
           'tol=0.3, interval=auto', 'tol=0.01, interval=1',
           'CMA-ES korr.', 'CMA-ES korr., interval=1']
NAMES = {'tol=0.01, interval=auto': 'DLR Standard (tol 0,01, auto)',
         'tol=0.05, interval=auto': 'DLR tol 0,05',
         'tol=0.3, interval=auto': 'DLR tol 0,3',
         'tol=0.01, interval=1': 'DLR Intervall 1',
         'CMA-ES korr.': 'CMA-ES korr.',
         'CMA-ES korr., interval=1': 'CMA-ES korr., Intervall 1'}
COLORS = ['#2a78d6', '#1baf7a', '#eda100', '#e87ba4', '#eb6834', '#6250d6']
STYLES = ['-', '-', '-', '-', '--', '--']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
FUNCS = ['Ellipsoid', 'Rotated Ellipsoid', 'Rosenbrock', 'Bent-Cigar']
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dd'


def summarize(df):
    rows = []
    for (label, func, n), d in df.groupby(['label', 'function', 'n']):
        ok = d[d.success == 1]
        rows.append({
            'label': label, 'function': func, 'n': n, 'runs': len(d),
            'successes': len(ok),
            'evals': ok.evals.median() if len(ok) else np.nan,
            'time': ok.wall_time.median() if len(ok) else np.nan,
            'ms_per_iter': d.time_per_iter_ms.median(),
            'mean_rank': d.mean_rank.mean(),
            'f_median': d.f_best.median(),
            'budget': d.budget.iloc[0]})
    return pd.DataFrame(rows)


def style(ax, title):
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def human(v, _=None):
    if v >= 1e6:
        return f'{v / 1e6:g} Mio.'
    if v >= 1e3:
        return f'{v / 1e3:g}k'
    return f'{v:g}'


def line_plot(s, col, title, fname, ylabel, note, annotate=True):
    funcs = [f for f in FUNCS if f in set(s.function)]
    dims = sorted(s.n.unique())
    fig, axes = plt.subplots(1, len(funcs), figsize=(4.2 * len(funcs), 4.4),
                             squeeze=False)
    for ax, func in zip(axes[0], funcs):
        for k, lab in enumerate(CONFIGS):
            d = s[(s.function == func) & (s.label == lab)].set_index('n')
            d = d.reindex(dims)
            y = d[col].to_numpy(dtype=float)
            ax.plot(dims, y, color=COLORS[k], linestyle=STYLES[k],
                    marker=MARKERS[k], markersize=6, linewidth=2,
                    markeredgecolor='white', markeredgewidth=1,
                    label=NAMES[lab])
            if not annotate:
                continue
            for n, yv, ok, runs in zip(dims, y, d.successes, d.runs):
                if np.isfinite(yv) and ok < runs:
                    ax.plot(n, yv, marker=MARKERS[k], markersize=9,
                            markerfacecolor='white', color=COLORS[k],
                            markeredgewidth=1.5, linestyle='none')
                    ax.annotate(f'{int(ok)}/{int(runs)}', (n, yv),
                                xytext=(5, 3), textcoords='offset points',
                                fontsize=7, color=MUTED)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(dims, [str(n) for n in dims])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(human))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('Dimension n', fontsize=8, color=MUTED)
        style(ax, func)
    axes[0][0].set_ylabel(ylabel, fontsize=8, color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.92))
    fig.suptitle(title, x=0.01, ha='left', fontsize=11, color=INK)
    fig.text(0.01, 0.01, note, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.8))
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def table(s, timing):
    lines = ['Stufe 3: Skalierung über n. Median erfolgreicher Läufe; '
             'Zeit und ms/Iteration ' + ('aus stage3_timing (sequenziell, '
                                         'Seed 0)' if timing is not None
                                         else 'aus parallelen Läufen (ca.)'),
             '']
    for func in FUNCS:
        d = s[s.function == func]
        if d.empty:
            continue
        lines.append(func)
        lines.append(f'  {"Konfiguration":30s} {"n":>4s} {"Erfolg":>7s} '
                     f'{"Eval.":>9s} {"Zeit s":>8s} {"ms/It":>7s} '
                     f'{"Rang":>5s}')
        for lab in CONFIGS:
            for r in d[d.label == lab].sort_values('n').itertuples():
                t = r.time
                ms = r.ms_per_iter
                if timing is not None:
                    tt = timing[(timing.label == lab)
                                & (timing.function == func)
                                & (timing.n == r.n)]
                    if len(tt):
                        t = tt.time.iloc[0]
                        ms = tt.ms_per_iter.iloc[0]
                ev = (f'{r.evals / 1e3:.0f}k' if np.isfinite(r.evals)
                      else '-')
                ts = f'{t:.0f}' if np.isfinite(t) else '-'
                lines.append(f'  {NAMES[lab]:30s} {r.n:4d} '
                             f'{r.successes:3d}/{r.runs:<3d} {ev:>9s} '
                             f'{ts:>8s} {ms:7.1f} {r.mean_rank:5.0f}')
        lines.append('')
    text = '\n'.join(lines)
    (BASE / 'stage3_table.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    BASE = (Path(sys.argv[sys.argv.index('--base') + 1]).resolve()
            if '--base' in sys.argv else ROOT / 'results' / 'stage3')
    OUT = BASE / 'plots'
    OUT.mkdir(parents=True, exist_ok=True)
    summary = summarize(pd.read_csv(BASE / 'stage3.csv'))
    summary.to_csv(BASE / 'stage3_summary.csv', index=False)
    tpath = BASE / 'stage3_timing.csv'
    timing = summarize(pd.read_csv(tpath)) if tpath.exists() else None

    line_plot(summary, 'evals',
              'Evaluierungen bis f < 1e-8 über der Dimension '
              '(Median erfolgreicher Läufe, Budget 3·10⁴·n)',
              'evals.png', 'Evaluierungen',
              'Hohle Punkte mit k/N: nicht alle Läufe erfolgreich; kein '
              'Punkt: kein Erfolg. Durchgezogen: DLR, gestrichelt: CMA-ES.')
    tsrc = timing if timing is not None else summary
    tnote = ('Sequenziell gemessen (stage3_timing, Seed 0, nur '
             + ', '.join(sorted(set(timing.function))) + ').'
             if timing is not None else
             'Aus parallel laufenden Prozessen, nur grob vergleichbar.')
    line_plot(tsrc, 'time',
              'Laufzeit bis f < 1e-8 über der Dimension', 'time.png',
              'Sekunden', tnote, annotate=timing is None)
    line_plot(tsrc, 'ms_per_iter',
              'Rechenaufwand pro Iteration über der Dimension', 'iter_cost.png',
              'ms pro Iteration', tnote + ' Median über alle Läufe, auch '
              'erfolglose.', annotate=False)
    table(summary, timing)
