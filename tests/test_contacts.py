#!/usr/bin/env python3
"""
Module: test_contacts
Purpose: Guard the two contact-classification bugs that made every published
         interaction count wrong, plus the receptor atom count.
Author: Dr. Arli Aditya Parikesit
Date: 2026
References:
    - Bissantz C, Kuhn B, Stahl M (2010) J Med Chem 53:5061-5084
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'scripts'))


def _load(stem):
    """Import a step module whose name starts with a digit."""
    path = ROOT / 'scripts' / f'{stem}.py'
    spec = importlib.util.spec_from_file_location(stem.replace('0', 'x'), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inter = _load('06_interactions')


def _atom(serial, name, ad_type, xyz, resid=1, resname='ALA'):
    """One PDBQT ATOM record, columns exact.

    Name 13-16, resname 18-20, coords 31-54, AutoDock type 78-79 (1-indexed).
    If these drift, element_of falls back to the atom name and the tests below
    pass for the wrong reason.
    """
    x, y, z = xyz
    rec = ('ATOM  ' + f'{serial:5d}' + ' ' + f'{name:<4s}' + ' '
           + f'{resname:>3s}' + ' ' + 'A' + f'{resid:4d}' + '    '
           + f'{x:8.3f}{y:8.3f}{z:8.3f}' + f'{1.00:6.2f}{0.00:6.2f}'
           + '    ' + f'{0.0:6.3f}' + ' ' + f'{ad_type:<2s}')
    assert rec[77:79].strip().upper() == ad_type.strip().upper(), rec
    assert rec[12:16].strip() == name, rec
    return rec


def test_aromatic_carbon_is_element_carbon():
    """AutoDock type A is aromatic carbon, not element 'A'."""
    assert inter.element_of(_atom(1, 'CG', 'A', (0, 0, 0))) == 'C'
    assert inter.element_of(_atom(1, 'C', 'C', (0, 0, 0))) == 'C'
    assert inter.element_of(_atom(1, 'OD1', 'OA', (0, 0, 0))) == 'O'
    assert inter.element_of(_atom(1, 'ND1', 'NA', (0, 0, 0))) == 'N'
    assert inter.element_of(_atom(1, 'HD1', 'HD', (0, 0, 0))) == 'H'
    assert inter.element_of(_atom(1, 'SD', 'SA', (0, 0, 0))) == 'S'
    assert inter.element_of(_atom(1, 'BR1', 'Br', (0, 0, 0))) == 'BR'


def test_aromatic_carbons_are_counted_as_hydrophobic_contacts():
    """Regression: every aromatic carbon used to fall through both branches."""
    rec = inter._collect([_atom(1, 'CG', 'A', (0.0, 0.0, 0.0))], with_residue=True)
    lig = inter._collect([_atom(1, 'C1', 'A', (3.5, 0.0, 0.0))], with_residue=False)
    _, n_hb, n_ph = inter.contacts(rec, lig, hb_cut=3.5, ph_cut=4.0)
    assert n_ph == 1, 'aromatic C...C contact not counted'
    assert n_hb == 0


def test_two_acceptors_facing_each_other_are_not_a_hydrogen_bond():
    """Regression: any N/O...N/O pair under the cutoff used to score as an H-bond."""
    rec = inter._collect([_atom(1, 'O', 'OA', (0.0, 0.0, 0.0))], with_residue=True)
    lig = inter._collect([_atom(1, 'O1', 'OA', (3.0, 0.0, 0.0))], with_residue=False)
    _, n_hb, _ = inter.contacts(rec, lig, hb_cut=3.5, ph_cut=4.0)
    assert n_hb == 0, 'acceptor-acceptor pair counted as a hydrogen bond'


def test_donor_acceptor_pair_is_a_hydrogen_bond():
    """A polar hydrogen on the receptor N makes it a donor."""
    rec = inter._collect([_atom(1, 'N', 'N', (0.0, 0.0, 0.0)),
                          _atom(2, 'HN', 'HD', (0.0, 0.0, 1.0))],
                         with_residue=True)
    assert rec[0]['is_donor'], 'N with an attached HD was not flagged as a donor'
    lig = inter._collect([_atom(1, 'O1', 'OA', (2.9, 0.0, 0.0))], with_residue=False)
    _, n_hb, _ = inter.contacts(rec, lig, hb_cut=3.5, ph_cut=4.0)
    assert n_hb == 1, 'donor-acceptor pair not counted'


def test_empty_pose_raises_instead_of_crashing_in_numpy():
    rec = inter._collect([_atom(1, 'N', 'N', (0.0, 0.0, 0.0))], with_residue=True)
    try:
        inter.contacts(rec, [], hb_cut=3.5, ph_cut=4.0)
    except ValueError:
        return
    raise AssertionError('empty pose should raise ValueError')


def test_receptor_atom_count_excludes_ter_records():
    """Regression: strip_receptor returned ATOM + TER, publishing 2342 for 2341."""
    import pytest
    pytest.importorskip('rdkit')
    import dock
    src = ROOT / 'data/raw/pdb/8B07.pdb'
    if not src.exists():
        pytest.skip('8B07.pdb not fetched')
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        n = dock.strip_receptor(src, Path(td) / 'rec.pdb', chain='A')
    expected = sum(1 for ln in src.read_text().splitlines()
                   if ln.startswith('ATOM') and len(ln) > 21 and ln[21] == 'A')
    assert n == expected, f'counted {n}, expected {expected} ATOM records'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'ok  {name}')
