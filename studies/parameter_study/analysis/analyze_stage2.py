"""Stage 2: low_rank_energy_tolerance x update_interval.

Reads results/stage2/stage2.csv and writes to results/stage2/:
- stage2_summary.csv, stage2_table.txt
- plots/evals.png   evaluations to target relative to the default
- plots/time.png    wall time to target relative to the default (approx.)
- plots/rank.png    mean rank (only functions on which the rank grows)

Usage (from studies/parameter_study):
    python analysis/analyze_stage2.py
"""

import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(ROOT), str(ROOT / 'variants')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import NullFormatter

from study import (STAGE2_FUNCTIONS, STAGE2_INTERVALS, STAGE2_TOLS,
                   stage2_label)

BASE = ROOT / 'results' / 'stage2'
OUT = BASE / 'plots'
DEFAULT = stage2_label(1e-2, None)
INK, MUTED = '#1f1f1e', '#6b6a63'
CMAP = LinearSegmentedColormap.from_list(
    'div', ['#184f95', '#86b6ef', '#f0efec', '#f2a3a2', '#b83232'])
CMAP.set_bad('#d9d8d2')
COLS = ['auto' if i is None else str(i) for i in STAGE2_INTERVALS]


def summarize(df):
    rows = []
    for (label, func), d in df.groupby(['label', 'function']):
        ok = d[d.success == 1]
        rows.append({
            'label': label, 'function': func,
            'tol': d.label.iloc[0].split(',')[0].split('=')[1],
            'interval': d.label.iloc[0].split('interval=')[1],
            'runs': len(d), 'successes': len(ok),
            'evals': ok.evals.median() if len(ok) else np.nan,
            'time': ok.wall_time.median() if len(ok) else np.nan,
            'mean_rank': d.mean_rank.mean(),
            'f_median': d.f_best.median()})
    s = pd.DataFrame(rows)
    base = s[s.label == DEFAULT].set_index('function')
    s['evals_rel'] = s.evals / s.function.map(base.evals)
    s['time_rel'] = s.time / s.function.map(base.time)
    return s


def grid(s, func, col):
    g = np.full((len(STAGE2_TOLS), len(STAGE2_INTERVALS)), np.nan)
    k = np.zeros_like(g)
    n = np.zeros_like(g)
    for i, tol in enumerate(STAGE2_TOLS):
        for j, iv in enumerate(STAGE2_INTERVALS):
            r = s[(s.function == func) & (s.label == stage2_label(tol, iv))]
            if len(r):
                g[i, j] = r[col].iloc[0]
                k[i, j], n[i, j] = r.successes.iloc[0], r.runs.iloc[0]
    return g, k, n


def heatmaps(s, col, title, fname, note):
    funcs = [f for f in STAGE2_FUNCTIONS if f in set(s.function)]
    fig, axes = plt.subplots(1, len(funcs), figsize=(3.3 * len(funcs), 3.9))
    norm = LogNorm(vmin=1 / 4, vmax=4)
    for ax, func in zip(np.atleast_1d(axes), funcs):
        g, k, n = grid(s, func, col)
        im = ax.imshow(np.clip(g, 1 / 4, 4), cmap=CMAP, norm=norm,
                       aspect='auto')
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if n[i, j] == 0:
                    continue
                if k[i, j] == 0:
                    txt = f'0/{int(n[i, j])}'
                else:
                    txt = f'{g[i, j]:.2f}'
                    if k[i, j] < n[i, j]:
                        txt += f'\n({int(k[i, j])}/{int(n[i, j])})'
                dark = np.isfinite(g[i, j]) and (g[i, j] < 0.4
                                                 or g[i, j] > 2.5)
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='white' if dark else INK)
        i0, j0 = STAGE2_TOLS.index(1e-2), STAGE2_INTERVALS.index(None)
        ax.add_patch(plt.Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False,
                                   edgecolor=INK, linewidth=1.5))
        ax.set_xticks(range(len(COLS)), COLS, fontsize=8)
        ax.set_yticks(range(len(STAGE2_TOLS)),
                      [f'{t:g}' for t in STAGE2_TOLS], fontsize=8)
        ax.set_xlabel('update_interval', fontsize=8, color=MUTED)
        ax.set_title(func, loc='left', fontsize=10, color=INK)
        ax.tick_params(length=0, colors=INK)
        for side in ax.spines.values():
            side.set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel('low_rank_energy_tolerance',
                                      fontsize=8, color=MUTED)
    cb = fig.colorbar(im, cax=fig.add_axes([0.945, 0.18, 0.008, 0.64]))
    cb.set_ticks([0.25, 0.5, 1, 2, 4],
                 labels=['<=0.25', '0.5', '1', '2', '>=4'])
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.01, ha='left', fontsize=11, color=INK)
    fig.text(0.01, 0.01, note, fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.06, right=0.925, top=0.82, bottom=0.18,
                        wspace=0.25)
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def rank_plot(s):
    funcs = ['Ellipsoid', 'Rotated Ellipsoid', 'Rosenbrock']
    funcs = [f for f in funcs if f in set(s.function)]
    fig, axes = plt.subplots(1, len(funcs), figsize=(3.6 * len(funcs), 3.9))
    for ax, func in zip(np.atleast_1d(axes), funcs):
        g, _, _ = grid(s, func, 'mean_rank')
        ax.imshow(g, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, f'{g[i, j]:.0f}', ha='center', va='center',
                        fontsize=8, color='white' if g[i, j] > 55 else INK)
        ax.set_xticks(range(len(COLS)), COLS, fontsize=8)
        ax.set_yticks(range(len(STAGE2_TOLS)),
                      [f'{t:g}' for t in STAGE2_TOLS], fontsize=8)
        ax.set_xlabel('update_interval', fontsize=8, color=MUTED)
        ax.set_title(func, loc='left', fontsize=10, color=INK)
        ax.tick_params(length=0, colors=INK)
        for side in ax.spines.values():
            side.set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel('low_rank_energy_tolerance',
                                      fontsize=8, color=MUTED)
    fig.suptitle('Mittlerer Rang über den Lauf (n = 100). Auf Bent-Cigar und '
                 'Discus bleibt er bei 1–2 und ist hier weggelassen.',
                 x=0.01, ha='left', fontsize=11, color=INK)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.16,
                        wspace=0.3)
    fig.savefig(OUT / 'rank.png', dpi=150)
    plt.close(fig)


def table(s):
    lines = [f'Stufe 2, n = 100. Standard = {DEFAULT}.', '']
    for func in STAGE2_FUNCTIONS:
        d = s[s.function == func]
        if d.empty:
            continue
        b = d[d.label == DEFAULT].iloc[0]
        # Candidates: configurations with the most successes (on Rosenbrock
        # no configuration reaches 5/5, success depends on the start point)
        full = d[d.successes == d.successes.max()]
        lines.append(f'{func}: Standard {b.evals / 1e3:.1f}k Evaluierungen, '
                     f'{b.time:.0f} s, Rang {b.mean_rank:.0f}, '
                     f'Erfolge {b.successes}/{b.runs}')
        if len(full):
            e = full.sort_values('evals').iloc[0]
            t = full.sort_values('time').iloc[0]
            lines.append(f'  wenigste Evaluierungen: {e.label} -> '
                         f'{e.evals / 1e3:.1f}k ({e.evals_rel:.2f}x), '
                         f'{e.time:.0f} s ({e.time_rel:.2f}x)')
            lines.append(f'  kürzeste Laufzeit:      {t.label} -> '
                         f'{t.evals / 1e3:.1f}k ({t.evals_rel:.2f}x), '
                         f'{t.time:.0f} s ({t.time_rel:.2f}x)')
        fail = d[d.successes < d.runs]
        if len(fail):
            lines.append('  nicht alle Läufe erfolgreich: ' + '; '.join(
                f'{r.label} ({r.successes}/{r.runs})'
                for r in fail.itertuples()))
        lines.append('')
    text = '\n'.join(lines)
    (BASE / 'stage2_table.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(BASE / 'stage2.csv')
    summary = summarize(data)
    summary.to_csv(BASE / 'stage2_summary.csv', index=False)
    heatmaps(summary, 'evals_rel',
             'Evaluierungen bis f < 1e-8 relativ zum Standard '
             '(Median erfolgreicher Läufe, n = 100, 5 Seeds; blau = besser)',
             'evals.png',
             'Umrahmt: Standard (tol = 0,01, Intervall automatisch). '
             '(k/5): nur k Läufe erfolgreich; grau: kein Erfolg.')
    heatmaps(summary, 'time_rel',
             'Laufzeit bis f < 1e-8 relativ zum Standard '
             '(Median erfolgreicher Läufe; blau = schneller)',
             'time.png',
             'Zeiten aus 6 parallel laufenden Prozessen, nur grob '
             'vergleichbar. Umrahmt: Standard.')
    rank_plot(summary)
    table(summary)
