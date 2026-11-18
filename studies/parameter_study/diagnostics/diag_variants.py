"""Compare DLR variants against the library on one seed.

Usage:
    python diag_variants.py ident                 (identity check)
    python diag_variants.py psifold [seed]
"""

import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(ROOT), str(ROOT / 'variants')]

import sys

import numpy as np

from study import FUNCTIONS, TARGET, run_single

N = 100
FULL = {'low_rank_is_adaptive': False, 'low_rank_init_dimension': N,
        'update_interval': 1}
ADAPT = {'update_interval': 1}


def show(label, r):
    ok = 'ja ' if r['success'] else 'nein'
    print(f'{label:44s} Erfolg={ok} Evals={r["evals"]:8d} '
          f'f={r["f_best"]:.3e} Rang={r["final_rank"]:3d} '
          f'Zeit={r["wall_time"]:6.1f}s', flush=True)


def task(label, kwargs, fname, seed):
    return ('diag', label, kwargs, 1.0, fname, N, seed)


if __name__ == '__main__':
    variant = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if variant == 'ident':
        for fname in ['Sphere', 'Rotated Ellipsoid']:
            a = run_single(task('lib', dict(FULL), fname, seed))
            b = run_single(task('copy', {**FULL, '_solver': 'dlr_ident'},
                                fname, seed))
            same = (a['iterations'] == b['iterations']
                    and a['f_best'] == b['f_best'])
            print(f'{fname:18s} Bibliothek it={a["iterations"]} '
                  f'f={a["f_best"]:.6e} | Kopie it={b["iterations"]} '
                  f'f={b["f_best"]:.6e} -> {"IDENTISCH" if same else "ABWEICHUNG"}')
        sys.exit()
    funcs = ['Rotated Ellipsoid', 'Ellipsoid', 'Rosenbrock', 'Sphere',
             'Bent-Cigar', 'Discus']
    print(f'n={N}, seed={seed}, Variante={variant}')
    for fname in funcs:
        for name, kw in [('voller Rang', FULL), ('adaptiv', ADAPT)]:
            show(f'{fname} | {name} | Bibliothek',
                 run_single(task('lib', dict(kw), fname, seed)))
            show(f'{fname} | {name} | {variant}',
                 run_single(task('var', {**kw, '_solver': f'dlr_{variant}'},
                                 fname, seed)))
