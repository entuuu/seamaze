"""Stage 3 pilot: one run per given configuration at n = 300 (seed 0) to
estimate evaluations and run time before launching stage 3.

Usage (from studies/parameter_study):
    python diagnostics/pilot_n300.py
"""

import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(ROOT), str(ROOT / 'variants')]

from concurrent.futures import ProcessPoolExecutor

from study import run_single, stage3_configs

N = 300
WANT = {('tol=0.01, interval=auto', 'Ellipsoid'),
        ('CMA-ES korr.', 'Ellipsoid'),
        ('tol=0.01, interval=auto', 'Rosenbrock')}

if __name__ == '__main__':
    tasks = [(g, lab, kw, sc, f, N, 0)
             for g, lab, kw, sc, funcs in stage3_configs() for f in funcs
             if (lab, f) in WANT]
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        for r in pool.map(run_single, tasks):
            print(f'{r["label"]:26s} {r["function"]:12s} n={r["n"]} '
                  f'Erfolg={r["success"]} f={r["f_best"]:.2e} '
                  f'Evals={r["evals"]} Budget={r["budget"]} '
                  f'Rang={r["final_rank"]} Zeit={r["wall_time"]:.0f}s '
                  f'ms/Iter={r["time_per_iter_ms"]:.1f} {r["stop"]}',
                  flush=True)
