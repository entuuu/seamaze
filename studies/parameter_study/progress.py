"""Live progress view of a running study stage.

Opens a window that refreshes every 20 s: one cell per (configuration,
function) with the number of finished seeds, failed runs marked, plus
overall progress and a rough remaining-time estimate from the log.
Only reads the result CSV and the log; does not disturb the running stage.

Usage (from studies/parameter_study):
    python progress.py stage2 5          (stage, number of seeds)
    python progress.py stage2 5 --png    (no window: write
                                          results/<stage>/progress.png
                                          every 20 s, e.g. to open in VS Code)
Close the window or press Ctrl+C to stop.
"""

import json
import re
import sys
import time
from pathlib import Path

import matplotlib

PNG = '--png' in sys.argv
matplotlib.use('Agg' if PNG else 'TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
STAGE_DIR = {'ladder_ref': 'ladder', 'psifold': 'ladder',
             'rotpilot': 'archiv_rotpilot'}       # same as run_study.py
REFRESH = 20
INK, MUTED = '#1f1f1e', '#6b6a63'
CMAP = LinearSegmentedColormap.from_list('prog', ['#f0efec', '#86b6ef',
                                                  '#2a78d6'])
LINE = re.compile(r'^\[(\d+)/(\d+)\s+(\d+)s\]', re.M)


def eta(log_path):
    """Remaining time from the last progress line (assumes constant rate;
    expensive functions run first, so the estimate is pessimistic)."""
    try:
        lines = log_path.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return None, None, None
    hits = LINE.findall(lines)
    if not hits:
        return 0, None, None
    i, n, t = map(int, hits[-1])
    finished = 'done in' in lines
    rest = 0 if finished else t * (n - i) / max(i, 1)
    return i, t, rest


def fmt(sec):
    if sec is None:
        return '?'
    h, m = divmod(int(sec) // 60, 60)
    return f'{h} h {m:02d} min' if h else f'{m} min'


def load_tasks(stage, n_seeds):
    """Task list (label, function) per run from the manifest written by
    run_study.py. Without a manifest it is built once via run_study (slow:
    imports the library) and saved."""
    folder = RESULTS / STAGE_DIR.get(stage, stage)
    path = folder / f'{stage}_tasks_{n_seeds}seeds.json'
    if not path.exists():
        print('Kein Manifest gefunden, erzeuge es einmalig '
              '(Import der Bibliothek, kann 1-2 min dauern) ...', flush=True)
        import run_study as rs
        tasks = [[t[1], t[4]] for t in rs.build_tasks(stage, n_seeds)]
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tasks), encoding='utf-8')
    return [tuple(t) for t in json.loads(path.read_text(encoding='utf-8'))]


def draw(fig, stage, n_seeds, tasks):
    labels = list(dict.fromkeys(t[0] for t in tasks))
    funcs = list(dict.fromkeys(t[1] for t in tasks))
    folder = RESULTS / STAGE_DIR.get(stage, stage)
    csv = folder / f'{stage}.csv'
    df = (pd.read_csv(csv) if csv.exists()
          else pd.DataFrame(columns=['label', 'function', 'success']))
    seeds_per = set(tasks)
    done = np.zeros((len(labels), len(funcs)))
    fail = np.zeros_like(done)
    total = np.zeros_like(done)
    for t in tasks:
        total[labels.index(t[0]), funcs.index(t[1])] += 1
    for (lab, fn), g in df.groupby(['label', 'function']):
        if (lab, fn) in seeds_per:
            i, j = labels.index(lab), funcs.index(fn)
            done[i, j] = len(g)
            fail[i, j] = int((g.success == 0).sum())
    frac = np.divide(done, total, out=np.full_like(done, np.nan),
                     where=total > 0)

    fig.clf()
    ax = fig.add_axes([0.24, 0.08, 0.72, 0.78])
    ax.imshow(frac, cmap=CMAP, vmin=0, vmax=1, aspect='auto')
    for i in range(len(labels)):
        for j in range(len(funcs)):
            if total[i, j] == 0:
                continue
            txt = f'{int(done[i, j])}/{int(total[i, j])}'
            if fail[i, j]:
                txt += f'  ✗{int(fail[i, j])}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='white' if frac[i, j] > 0.7 else INK,
                    fontweight='bold' if fail[i, j] else 'normal')
    ax.set_xticks(range(len(funcs)), funcs, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.tick_params(length=0, colors=INK)
    for side in ax.spines.values():
        side.set_visible(False)

    n_done, n_all = int(done.sum()), int(total.sum())
    i, elapsed, rest = eta(RESULTS / 'logs' / f'{stage}.log')
    fig.text(0.02, 0.95,
             f'{stage}: {n_done}/{n_all} Läufe fertig '
             f'({100 * n_done / max(n_all, 1):.0f} %), '
             f'{int(fail.sum())} ohne Erfolg (✗)', fontsize=12, color=INK)
    fig.text(0.02, 0.915,
             f'Laufzeit bisher: {fmt(elapsed)}   ·   geschätzt verbleibend: '
             f'{fmt(rest)} (eher zu hoch, teure Läufe kommen zuerst)   ·   '
             f'aktualisiert {time.strftime("%H:%M:%S")}',
             fontsize=9, color=MUTED)
    return n_done >= n_all


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    stage = args[0] if args else 'stage2'
    n_seeds = int(args[1]) if len(args) > 1 else 5
    tasks = load_tasks(stage, n_seeds)
    fig = plt.figure(figsize=(10, 7.5))
    folder = RESULTS / STAGE_DIR.get(stage, stage)
    while True:
        finished = draw(fig, stage, n_seeds, tasks)
        if PNG:
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / 'progress.png', dpi=110)
        else:
            if not plt.fignum_exists(fig.number) and plt.get_fignums() == []:
                plt.show(block=False)
            fig.canvas.draw_idle()
            plt.pause(0.1)
        if finished:
            print('Stufe vollständig.')
            if not PNG:
                plt.show()
            break
        end = time.time() + REFRESH
        while time.time() < end:
            if not PNG:
                if not plt.fignum_exists(fig.number):
                    return
                plt.pause(0.5)
            else:
                time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
