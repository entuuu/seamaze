"""Create cmaes_path_variant.py: copy of the library CMA-ES in which the
step-size path is rotated back into the original coordinates
(B D^-1 B^T y instead of D^-1 B^T y). Diagnosis only."""
from pathlib import Path

src = Path(__file__).resolve().parents[3] / 'seamaze/optimizers/_cmaes.py'
text = src.read_text(encoding='utf-8')
old = "    elite_mean_step_tr = inv_root_core * latent_step\n"
new = ("    elite_mean_step_tr = ascontiguousarray(basis) @ "
       "(inv_root_core * latent_step)\n")
assert text.count(old) == 1
text = text.replace(old, new)
if 'ascontiguousarray' not in text.split('# %% Internal')[0]:
    text = text.replace('from numpy import (', 'from numpy import (\n    ascontiguousarray,', 1)
Path(__file__).with_name('cmaes_path_variant.py').write_text(text, encoding='utf-8')
print('written')
