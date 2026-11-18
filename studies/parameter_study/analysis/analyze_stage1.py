"""Analyse a study CSV: ERT table, summary and one figure per parameter group.

Usage:
    python analyze.py stage1      (reads results/stage1/stage1.csv)

ERT (expected running time) = sum of evaluations over all runs (failed runs
included with the evaluations they used) / number of successful runs.
Evaluations are counted as iterations x population size.
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
from matplotlib.ticker import NullFormatter

from study import stage1_configs

# Fixed label order as defined in the study (not CSV completion order)
CONFIG_ORDER = [c[1] for c in stage1_configs()]


def ordered(labels):
    present = set(labels)
    known = [lab for lab in CONFIG_ORDER if lab in present]
    return known + sorted(present - set(known))

RESULTS = ROOT / 'results'

# Categorical palette (fixed order) plus markers as secondary encoding
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
          '#6250d6']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
FUNC_ORDER = ['Sphere', 'Bent-Cigar', 'Discus', 'Ellipsoid',
              'Rotated Ellipsoid', 'Rosenbrock', 'Rastrigin']
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dd'

# Where the default configuration sits on each group's axis (None: reference
# line only, the default is not a member of that group)
DEFAULT_POSITION = {
    'energy_tol': ('tol=0.01', 2), 'max_dim': ('rmax=100', 4),
    'fixed_rank': None, 'init_dim': ('r0=1', 0),
    'update_interval': ('interval=auto', 0), 'pop_size': ('pop=17', 0),
    'sigma0': ('sigma0 x1', 1),
    }
GROUP_TITLE = {
    'energy_tol': 'low_rank_energy_tolerance',
    'max_dim': 'low_rank_max_dimension',
    'fixed_rank': 'fester Rang (low_rank_is_adaptive=False)',
    'init_dim': 'low_rank_init_dimension',
    'update_interval': 'update_interval',
    'pop_size': 'number_of_individuals',
    'sigma0': 'initial_sigma (Faktor auf 0.3 x Bereichsbreite)',
    }


def summarize(df):
    g = df.groupby(['group', 'label', 'function'], sort=False)
    out = g.agg(runs=('success', 'size'), successes=('success', 'sum'),
                evals_sum=('evals', 'sum'),
                f_median=('f_best', 'median'),
                mean_rank=('mean_rank', 'mean'),
                final_rank=('final_rank', 'median'),
                ms_per_iter=('time_per_iter_ms', 'median'),
                stops=('stop', lambda s: ','.join(sorted(set(s))))
                ).reset_index()
    ok = df[df.success == 1].groupby(['group', 'label', 'function'])
    keys = list(zip(out.group, out.label, out.function))
    med_ev, med_t = ok['evals'].median(), ok['wall_time'].median()
    out['evals_median_success'] = [med_ev.get(k, np.nan) for k in keys]
    out['wall_median_success'] = [med_t.get(k, np.nan) for k in keys]
    out['success_rate'] = out.successes / out.runs
    out['ERT'] = np.where(out.successes > 0,
                          out.evals_sum / out.successes.replace(0, 1), np.inf)
    base = out[out.label == 'default'].set_index('function')
    out['ERT_rel'] = out.ERT / out.function.map(base.ERT)
    out['evals_rel'] = (out.evals_median_success
                        / out.function.map(base.evals_median_success))
    out['wall_rel'] = (out.wall_median_success
                       / out.function.map(base.wall_median_success))
    return out


def group_frame(summary, group):
    part = summary[summary.group == group].copy()
    labels = ordered(part.label)
    pos = DEFAULT_POSITION.get(group)
    d = summary[summary.label == 'default'].copy()
    d = d[d.function.isin(part.function.unique())]
    if pos is not None:
        name, idx = pos
    else:                                  # default shown as first column
        name, idx = 'default', 0
    d['group'], d['label'] = group, name
    part = pd.concat([part, d])
    labels.insert(idx, name)
    return part, labels


def style_axis(ax, title):
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_group(summary, group, outdir):
    """Evaluations and wall time to target (successful runs) and mean rank.

    Points with failed runs are drawn hollow and labelled with k/n
    successes; configurations without any success have no point.
    """
    part, labels = group_frame(summary, group)
    part = part[part.function != 'Rastrigin']
    x = np.arange(len(labels))
    funcs = [f for f in FUNC_ORDER if f in set(part.function)]
    panels = [('evals_rel', 'Evaluierungen bis Ziel, rel. zu default', True),
              ('wall_rel', 'Laufzeit bis Ziel, rel. zu default (ca.)', True),
              ('mean_rank', 'mittlerer Rang über den Lauf', False)]
    if group == 'fixed_rank':            # rank is fixed: panel is trivial
        panels = panels[:2]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.8),
                             sharex=True)
    for ax, (col, title, logy) in zip(axes, panels):
        for f in funcs:
            k = FUNC_ORDER.index(f)
            s = part[part.function == f].set_index('label').reindex(labels)
            y = s[col].to_numpy(dtype=float)
            ax.plot(x, y, color=COLORS[k], linewidth=2, label=f,
                    marker=MARKERS[k], markersize=6, markeredgecolor='white',
                    markeredgewidth=1)
            if col == 'mean_rank':
                continue
            for xi, yi, ok, n in zip(x, y, s.successes, s.runs):
                if np.isfinite(yi) and ok < n:
                    ax.plot(xi, yi, marker=MARKERS[k], markersize=9,
                            markerfacecolor='white', color=COLORS[k],
                            markeredgewidth=1.5)
                    ax.annotate(f'{int(ok)}/{int(n)}', (xi, yi),
                                xytext=(5, 4), textcoords='offset points',
                                fontsize=7, color=MUTED)
        if col != 'mean_rank':
            ax.axhline(1.0, color=MUTED, linewidth=1, linestyle='--')
        if logy:
            ax.set_yscale('log')
        style_axis(ax, title)
        ax.set_xticks(x, labels, rotation=30, ha='right')
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=len(funcs),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(GROUP_TITLE.get(group, group), x=0.01, ha='left', y=0.985,
                 fontsize=12, color=INK)
    fig.text(0.01, 0.01,
             'Median über erfolgreiche Läufe. Hohle Punkte mit k/n: nicht '
             'alle Läufe erfolgreich; kein Punkt: kein Erfolg. '
             'Laufzeiten aus parallelen Läufen, nur grob vergleichbar.',
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    fig.savefig(outdir / f'{group}.png', dpi=150)
    plt.close(fig)


def plot_overview(summary, outdir):
    """Heatmap: evaluations to target relative to default, all configs."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    s = summary[summary.function != 'Rastrigin']
    funcs = [f for f in FUNC_ORDER if f in set(s.function)]
    labels = ordered(s.label)

    def pivot(col):
        return s.pivot_table(index='label', columns='function', values=col,
                             sort=False).reindex(index=labels, columns=funcs)

    val, ok, n = pivot('evals_rel'), pivot('successes'), pivot('runs')
    cmap = LinearSegmentedColormap.from_list(
        'div', ['#184f95', '#86b6ef', '#f0efec', '#f2a3a2', '#b83232'])
    cmap.set_bad('#d9d8d2')
    norm = LogNorm(vmin=1 / 4, vmax=4)
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(labels) + 1.8))
    im = ax.imshow(val.clip(1 / 4, 4).to_numpy(dtype=float), cmap=cmap,
                   norm=norm, aspect='auto')
    for i in range(len(labels)):
        for j in range(len(funcs)):
            v, k, m = val.iat[i, j], ok.iat[i, j], n.iat[i, j]
            if not np.isfinite(k):
                continue
            if k == 0:
                txt = f'0/{int(m)}'
            else:
                txt = f'{v:.2f}' + ('' if k == m else f' ({int(k)}/{int(m)})')
            dark = np.isfinite(v) and (v < 0.4 or v > 2.5)
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    color='white' if dark else INK)
    ax.set_xticks(range(len(funcs)), funcs, rotation=20, ha='right',
                  fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.tick_params(length=0, colors=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    groups = (s.drop_duplicates('label').set_index('label').group
              .reindex(labels).tolist())
    for i in range(1, len(labels)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color='white', linewidth=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_ticks([0.25, 0.5, 1, 2, 4],
                 labels=['<=0.25', '0.5', '1', '2', '>=4'])
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    cb.outline.set_visible(False)
    ax.set_title('Evaluierungen bis Ziel f<1e-8 relativ zu default '
                 '(Median erfolgreicher Läufe; blau = besser)', loc='left',
                 fontsize=10, color=INK)
    fig.text(0.01, 0.005, '(k/n): nur k von n Läufen erfolgreich, '
             'grau: kein Erfolg.', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(outdir / 'overview.png', dpi=150)
    plt.close(fig)


def plot_rastrigin(df, outdir):
    """Rastrigin: final f per population size (success is not expected)."""
    d = df[df.function == 'Rastrigin'].copy()
    if d.empty:
        return
    d = d.sort_values('pop')
    pops = sorted(d['pop'].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(pops))
    med = [d[d['pop'] == p].f_best.median() for p in pops]
    for i, p in enumerate(pops):
        vals = d[d['pop'] == p].f_best
        ax.scatter(np.full(len(vals), i), vals, s=36, color=COLORS[0],
                   alpha=0.45, edgecolors='white', linewidths=1)
    ax.plot(x, med, color=COLORS[0], marker='o', markersize=8, linewidth=2,
            label='Median')
    ax.set_xticks(x, [str(p) for p in pops])
    ax.set_xlabel('number_of_individuals', color=MUTED, fontsize=9)
    style_axis(ax, 'Rastrigin-100D: bester f-Wert nach Budget (Punkte: Seeds)')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / 'rastrigin_pop.png', dpi=150)
    plt.close(fig)


def main(stage):
    base = RESULTS / stage
    df = pd.read_csv(base / f'{stage}.csv')
    summary = summarize(df)
    summary.to_csv(base / f'{stage}_summary.csv', index=False)
    outdir = base / 'plots'
    outdir.mkdir(exist_ok=True)
    for old in outdir.glob('*.png'):
        old.unlink()
    plot_overview(summary, outdir)
    for group in summary.group.unique():
        # init_dim and sigma0 show no effect: covered by the overview only
        if group not in ('default', 'init_dim', 'sigma0'):
            plot_group(summary, group, outdir)
    plot_rastrigin(df, outdir)

    cols = ['label', 'function', 'successes', 'runs', 'ERT', 'ERT_rel',
            'evals_median_success', 'evals_rel', 'wall_median_success',
            'wall_rel', 'mean_rank', 'stops']
    with pd.option_context('display.width', 250, 'display.max_rows', 500,
                           'display.float_format', '{:.3g}'.format):
        print(summary[cols].to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'stage1')
