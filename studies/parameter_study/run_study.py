"""Run a DLR-CMA-ES study stage in parallel and append results to a CSV.

Usage (from studies/parameter_study):
    python run_study.py <stage> [workers] [n_seeds]   (seeds 0..n_seeds-1)
    python run_study.py repro                         (reproducibility check)

Stages: pilot, stage1, stage2, ladder, ladder_ref, psifold, rotpilot.
Results: results/<folder>/<stage>.csv (folder see STAGE_DIR). Redirect the
console output to results/logs/<stage>.log.

While a stage runs, Windows is asked not to enter sleep (only for the
lifetime of this process; no power settings are changed).

Already finished tasks (same key in the CSV) are skipped, so an interrupted
run can be resumed by starting the command again.
"""

import os
import sys

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import csv
import ctypes
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

from study import (NDIM, ROTATED_FUNCTIONS, ladder_configs, psifold_configs,
                   ref_configs, run_single, stage1_configs, stage2_configs,
                   task_key)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
# Result folder per stage (CSV: results/<folder>/<stage>.csv)
STAGE_DIR = {'ladder_ref': 'ladder', 'psifold': 'ladder',
             'rotpilot': 'archiv_rotpilot'}


def build_tasks(stage, n_seeds):
    cfgs = stage1_configs(NDIM)
    if stage == 'pilot':
        keep = {'default', 'tol=0.0001', 'fixed r=50', 'interval=1',
                'pop=136'}
        cfgs = [c for c in cfgs if c[1] in keep]
        seeds = range(2)
    elif stage == 'stage1':
        seeds = range(n_seeds)
    elif stage == 'ladder':
        cfgs = ladder_configs(NDIM)
        seeds = range(n_seeds)
    elif stage == 'ladder_ref':
        cfgs = ref_configs()
        seeds = range(n_seeds)
    elif stage == 'stage2':
        cfgs = stage2_configs()
        seeds = range(n_seeds)
    elif stage == 'psifold':
        cfgs = psifold_configs(NDIM)
        seeds = range(n_seeds)
    elif stage == 'rotpilot':
        keep = {'default', 'fixed r=1', 'fixed r=10', 'tol=0.3', 'rmax=10',
                'interval=1'}
        cfgs = [(g, lab, kw, sc, ROTATED_FUNCTIONS)
                for g, lab, kw, sc, _ in cfgs if lab in keep]
        seeds = range(n_seeds)
    else:
        raise ValueError(stage)
    tasks = []
    for group, label, kwargs, scale, funcs in cfgs:
        for fname in funcs:
            for seed in seeds:
                tasks.append((group, label, kwargs, scale, fname, NDIM, seed))
    return tasks


def load_done(path):
    if not path.exists():
        return set()
    with open(path, newline='') as fh:
        return {f"{r['label']}|{r['function']}|{r['n']}|{r['seed']}"
                for r in csv.DictReader(fh)}


def keep_awake(on):
    """Ask Windows not to sleep while this process runs (no-op elsewhere)."""
    if os.name != 'nt':
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    flags = es_continuous | (es_system_required if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def run_stage(stage, workers, n_seeds):
    folder = RESULTS / STAGE_DIR.get(stage, stage)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{stage}.csv'
    done = load_done(path)
    all_tasks = build_tasks(stage, n_seeds)
    # Manifest for progress.py (avoids importing the library there)
    (folder / f'{stage}_tasks_{n_seeds}seeds.json').write_text(
        json.dumps([[t[1], t[4]] for t in all_tasks]), encoding='utf-8')
    tasks = [t for t in all_tasks if task_key(t) not in done]
    # Expensive functions first for better load balancing
    order = {'Ellipsoid': 0, 'Rosenbrock': 1, 'Rastrigin': 2,
             'Rotated Ellipsoid': 3}
    tasks.sort(key=lambda t: order.get(t[4], 9))
    print(f'{stage}: {len(tasks)} tasks to run ({len(done)} already done), '
          f'{workers} workers', flush=True)

    t0 = perf_counter()
    writer, fh = None, None
    keep_awake(True)
    try:
        _run_pool(tasks, workers, path, t0)
    finally:
        keep_awake(False)
    print(f'done in {perf_counter() - t0:.0f}s', flush=True)


def _run_pool(tasks, workers, path, t0):
    writer, fh = None, None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_single, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as exc:          # keep the study running
                t = futures[fut]
                print(f'FAILED {task_key(t)}: {exc!r}', flush=True)
                continue
            if writer is None:
                new = not path.exists()
                fh = open(path, 'a', newline='')
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    writer.writeheader()
            writer.writerow(row)
            fh.flush()
            print(f'[{i}/{len(tasks)} {perf_counter() - t0:7.0f}s] '
                  f'{row["label"]:24s} {row["function"]:18s} s={row["seed"]} '
                  f'ok={row["success"]} f={row["f_best"]:.2e} '
                  f'evals={row["evals"]:8d} r={row["final_rank"]:3d} '
                  f't={row["wall_time"]:6.1f}s {row["stop"]}', flush=True)
    if fh:
        fh.close()


def repro_check():
    task = ('default', 'default', {}, 1.0, 'Rotated Ellipsoid', 30, 3)
    with ProcessPoolExecutor(max_workers=2) as pool:
        a, b = pool.map(run_single, [task, task])
    keys = ('iterations', 'f_best', 'final_rank', 'mean_rank')
    for k in keys:
        print(f'{k:12s} {a[k]!r:>25} {b[k]!r:>25}')
    print('REPRODUCIBLE' if all(a[k] == b[k] for k in keys)
          else 'NOT REPRODUCIBLE')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    if stage == 'repro':
        repro_check()
    else:
        run_stage(stage, workers, n_seeds)
