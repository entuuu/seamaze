"""Diagnosis: library CMA-ES vs. variant with the step-size path in
original coordinates. Usage: python diag_path.py [seed]"""
import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(ROOT), str(ROOT / 'variants')]

import sys

import numpy as np

from diag_weights import run, N
from study import TARGET
from seamaze.optimizers import CMAES
from cmaes_path_variant import CMAES as CMAESVariant

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
funcs = ['Sphere', 'Bent-Cigar', 'Discus', 'Ellipsoid', 'Rotated Ellipsoid',
         'Rosenbrock']
print(f'n={N}, seed={seed}, Evaluierungen bis f<1e-8 (- = kein Erfolg)')
print(f'{"Funktion":18s} {"Intervall":>9s} {"Bibliothek":>11s} {"Variante":>9s}')
for fname in funcs:
    for interval in (1, None):
        kw = {} if interval is None else {'update_interval': interval}
        out = []
        for cls in (CMAES, CMAESVariant):
            res, _, _ = run(fname, seed, cls, kw, False)
            ok = res['optimal_value'] < TARGET
            out.append(f'{res["iterations"] * 17:d}' if ok else '-')
        print(f'{fname:18s} {str(interval or "auto"):>9s} {out[0]:>11s} '
              f'{out[1]:>9s}', flush=True)
