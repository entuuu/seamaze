"""Create diagnostic copies of the library DLR-CMA-ES (library untouched).

dlr_variant_ident.py   unchanged copy (identity check)
dlr_variant_psifold.py old diag(psi) is projected into the core update
                       (ext_s += U_aug^T diag(psi) U_aug). At full rank the
                       whole covariance then goes through the same update as
                       in CMA-ES and psi is (nearly) emptied every step.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / 'seamaze' / 'optimizers' / '_dlrcmaes.py'

PATCHES = {
    'ident': [],
    'psifold': [(
        "    ext_s = (basis_proj * core) @ basis_proj.T\n",
        "    ext_s = (basis_proj * core) @ basis_proj.T\n"
        "    ext_s = ext_s + (uhat_aug_tr * psi) @ uhat_aug\n")],
    }

if __name__ == '__main__':
    text = SRC.read_text(encoding='utf-8')
    for name, patches in PATCHES.items():
        out = text
        for old, new in patches:
            assert out.count(old) == 1, (name, old)
            out = out.replace(old, new)
        (HERE / f'dlr_variant_{name}.py').write_text(out, encoding='utf-8')
        print('written', name)
