"""Attribution ladder: CMA-ES -> full-rank DLR -> fixed rank -> adaptive
rank -> default. Reads results/stage1.csv and results/ladder.csv.

Usage:
    python analyze_ladder.py
"""

import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(ROOT), str(ROOT / 'variants')]

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

RESULTS = ROOT / 'results' / 'ladder'
OUT = RESULTS / 'plots'

FUNCS = ['Sphere', 'Bent-Cigar', 'Discus', 'Ellipsoid', 'Rotated Ellipsoid',
         'Rosenbrock']
RANKS = [1, 2, 5, 10, 25, 50]
COLORS = {'Ellipsoid': '#eda100', 'Rotated Ellipsoid': '#e87ba4',
          'Rosenbrock': '#008300'}
MARKERS = {'Ellipsoid': 'D', 'Rotated Ellipsoid': 'v', 'Rosenbrock': 'P'}
BLUE, INK, MUTED, GRID, FAIL = ('#2a78d6', '#1f1f1e', '#6b6a63', '#e4e3dd',
                                '#b9b8b0')
BUDGET = 1e6


def load():
    a = pd.read_csv(ROOT / 'results' / 'stage1' / 'stage1.csv')
    b = pd.read_csv(RESULTS / 'ladder.csv')
    # Corrected CMA-ES reference (library CMA-ES has a step-size path bug)
    c = pd.read_csv(RESULTS / 'ladder_ref.csv')
    keep = ['default', 'interval=1'] + [f'fixed r={r}' for r in RANKS]
    return pd.concat([a[a.label.isin(keep)], b, c], ignore_index=True)


def cell(df, label, func):
    d = df[(df.label == label) & (df.function == func)]
    ok = d[d.success == 1]
    return {'label': label, 'n': len(d), 'ok': len(ok),
            'evals': ok.evals.median() if len(ok) else np.nan,
            'time': ok.wall_time.median() if len(ok) else np.nan,
            'rank': d.mean_rank.mean(), 'runs': d}


def best_fixed(df, func):
    """Fixed rank (interval 1) with the fewest evaluations; prefers ranks
    with the most successes."""
    cells = [cell(df, f'fixed r={r}, interval=1', func) for r in RANKS]
    most = max(c['ok'] for c in cells)
    cells = [c for c in cells if c['ok'] == most]
    # Smallest rank within 2 % of the minimum (differences below are noise)
    low = min(c['evals'] for c in cells)
    best = next(c for c in cells if c['evals'] <= 1.02 * low)
    best['r'] = int(best['label'].split('=')[1].split(',')[0])
    return best


def ladder(df, func):
    return [
        ('CMA-ES\n(korrigiert)', cell(df, 'CMA-ES korr., interval=1', func)),
        ('DLR\nvoller Rang', cell(df, 'full rank, interval=1', func)),
        ('DLR\nbester\nfester Rang', best_fixed(df, func)),
        ('DLR\nadaptiver\nRang', cell(df, 'interval=1', func)),
        ('DLR\nStandard\n(Auto-Intervall)', cell(df, 'default', func)),
        ]


def fmt_evals(v, _=None):
    if v >= 1e6:
        return f'{v / 1e6:g} Mio.'
    return f'{v / 1e3:g}k'


def style(ax):
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def log_axis(ax, lo, hi):
    ax.set_yscale('log')
    ticks = [t for t in (5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6)
             if lo <= t <= hi]
    ax.set_ylim(lo, hi)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_evals))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_ladder(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for ax, func in zip(axes.ravel(), FUNCS):
        steps = ladder(df, func)
        vals = []
        for i, (_, c) in enumerate(steps):
            runs = c['runs']
            ok = runs[runs.success == 1]
            ax.scatter(np.full(len(ok), i), ok.evals, s=22, color=BLUE,
                       alpha=0.35, linewidths=0, zorder=2)
            if c['ok']:
                ax.plot([i - 0.28, i + 0.28], [c['evals']] * 2, color=BLUE,
                        linewidth=3, solid_capstyle='round', zorder=3)
                txt = fmt_evals(round(c['evals'], -2))
                if c['ok'] < c['n']:
                    txt += f"\n({c['ok']}/{c['n']} Erfolge)"
                ax.annotate(txt, (i, c['evals']), xytext=(0, 9),
                            textcoords='offset points', ha='center',
                            fontsize=8, color=INK)
                vals.append(c['evals'])
            else:
                ax.annotate('kein Erfolg', (i, BUDGET), ha='center',
                            fontsize=8, color=MUTED)
            vals.extend(ok.evals.tolist())
        lo, hi = min(vals), max(vals)
        log_axis(ax, lo / 1.5, hi * 2.2)
        names = [name for name, _ in steps]
        names[2] += f" (r={steps[2][1]['r']})"
        ax.set_xticks(range(len(steps)), names, fontsize=8)
        ax.set_xlim(-0.6, len(steps) - 0.4)
        ax.axvline(3.5, color=GRID, linewidth=1.2)
        ax.set_title(func, loc='left', fontsize=11, color=INK)
        style(ax)
    fig.suptitle('Evaluierungen bis f < 1e-8 entlang der Stufenleiter '
                 '(n = 100, Strich: Median, Punkte: einzelne Seeds)',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.text(0.01, 0.005,
             'Die ersten vier Stufen aktualisieren die Kovarianz in jeder '
             'Generation (Intervall 1). Rechts der Trennlinie: '
             'Standardeinstellung mit automatischem Intervall. '
             'Jedes Panel hat eine eigene y-Achse.', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(OUT / 'ladder.png', dpi=150)
    plt.close(fig)


def plot_fixed_rank(df):
    """Evaluations over the fixed rank (interval 1) for the functions on
    which the rank matters; hollow marker: adaptive rank at its mean rank."""
    funcs = ['Ellipsoid', 'Rotated Ellipsoid', 'Rosenbrock']
    ranks = RANKS + [100]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for func in funcs:
        labels = [f'fixed r={r}, interval=1' for r in RANKS]
        labels.append('full rank, interval=1')
        cells = [cell(df, lab, func) for lab in labels]
        y = [c['evals'] for c in cells]
        col, mk = COLORS[func], MARKERS[func]
        ax.plot(ranks, y, color=col, marker=mk, markersize=7, linewidth=2,
                markeredgecolor='white', markeredgewidth=1, label=func)
        for r, c in zip(ranks, cells):
            if c['ok'] < c['n']:
                ax.annotate(f"{c['ok']}/{c['n']}", (r, c['evals']),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=7, color=MUTED)
        a = cell(df, 'interval=1', func)
        ax.plot(a['rank'], a['evals'], marker=mk, markersize=11,
                markerfacecolor='white', markeredgecolor=col,
                markeredgewidth=2, linestyle='none', zorder=4)
        ax.annotate('adaptiv', (a['rank'], a['evals']), xytext=(0, -15),
                    textcoords='offset points', ha='center', fontsize=7,
                    color=MUTED)
    ax.set_xscale('log')
    ax.set_xticks(ranks, [str(r) for r in ranks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('fester Rang (100 = voller Rang)', fontsize=9, color=MUTED)
    log_axis(ax, 1.5e5, 8e5)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc='upper center', ncol=3,
              bbox_to_anchor=(0.5, 1.12))
    ax.set_title('Evaluierungen bis f < 1e-8 über dem festen Rang '
                 '(Intervall 1, n = 100)', loc='left', fontsize=10,
                 color=INK, pad=28)
    fig.text(0.01, 0.01, 'Hohle Marker: adaptiver Rang (Intervall 1), '
             'eingetragen bei seinem mittleren Rang. k/5: nicht alle Läufe '
             'erfolgreich.', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / 'fixed_rank.png', dpi=150)
    plt.close(fig)


def table(df):
    lines = ['Evaluierungen bis f < 1e-8 (Median erfolgreicher Läufe), '
             'in Klammern Erfolge, wenn nicht 5/5', '']
    head = ['Funktion', 'CMA-ES korr.', 'voller Rang', 'bester fester Rang',
            'adaptiv', 'Standard']
    lines.append(' | '.join(head))
    factors = ['', 'Faktor zur Stufe davor', '']
    for func in FUNCS:
        steps = ladder(df, func)
        row, frow, prev = [func], [func], None
        for i, (_, c) in enumerate(steps):
            txt = fmt_evals(round(c['evals'], -2)) if c['ok'] else '-'
            if i == 2:
                txt += f" (r={c['r']})"
            if c['ok'] < c['n']:
                txt += f" [{c['ok']}/{c['n']}]"
            row.append(txt)
            frow.append('' if prev is None else f"{c['evals'] / prev:.2f}")
            prev = c['evals']
        lines.append(' | '.join(row))
        factors.append(' | '.join(frow))
    lines += factors
    extra = ['', 'Weitere Konfigurationen (Median Evaluierungen, Erfolge)',
             '']
    for lab in ['CMA-ES korr.', 'CMA-ES', 'CMA-ES, interval=1',
                'full rank']:
        parts = []
        for func in FUNCS:
            c = cell(df, lab, func)
            v = fmt_evals(round(c['evals'], -2)) if c['ok'] else '-'
            parts.append(f"{func}: {v} [{c['ok']}/{c['n']}]")
        extra.append(f'{lab} (Auto-Intervall) -> ' + '; '.join(parts))
    r = df[df.function == 'Rastrigin'].groupby('label').f_best.median()
    extra.append('Rastrigin, Median f: ' + '; '.join(
        f'{k}: {v:.0f}' for k, v in r.items()))
    text = '\n'.join(lines + extra)
    (RESULTS / 'ladder_table.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    data = load()
    table(data)
    plot_ladder(data)
    plot_fixed_rank(data)
