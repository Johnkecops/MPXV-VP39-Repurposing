#!/usr/bin/env python3
"""
Module: tests/test_pipeline
Purpose: Regression tests for the revision pipeline. These check the claims the
         manuscript makes, not merely that the code runs: if a number in the paper
         stops matching the computed results, a test fails.

Author: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif Nur Muhammad Ansori
Date: 2026
Run:
    conda run -n mpxv-revision python -m pytest tests/ -q
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

MANUSCRIPT = ROOT / 'Biologia-26-manuscript AAP - REV ANMA.docx'


def load(rel):
    p = ROOT / rel
    if not p.exists():
        pytest.skip(f'{rel} not generated yet')
    return p


# --------------------------------------------------------------------------
# Ligand identity (reviewer concern R5)
# --------------------------------------------------------------------------

def test_ligand_names_are_compounds_not_enzymes():
    df = pd.read_csv(load('results/tables/ligand_identity.csv'))
    names = set(df['Correct_compound'])
    assert 'SGC0946' in names and 'SGC8158' in names
    assert 'DOT1L' not in names and 'PRMT7' not in names


def test_every_ligand_has_a_parsable_structure():
    from rdkit import Chem
    df = pd.read_csv(load('results/tables/ligand_identity.csv'))
    assert len(df) == 7
    for _, r in df.iterrows():
        assert Chem.MolFromSmiles(r['Canonical_SMILES']) is not None, r['Correct_compound']


def test_lipinski_verdicts_match_the_manuscript_claim():
    """SAM and SAH pass all four criteria; the three large inhibitors do not."""
    df = pd.read_csv(load('results/tables/ligand_identity.csv')).set_index(
        'Correct_compound')
    assert df.loc['S-adenosyl-L-methionine', 'Lipinski_violations'] == 0
    assert df.loc['S-adenosyl-L-homocysteine', 'Lipinski_violations'] == 0
    assert df.loc['STM957', 'Lipinski_violations'] == 2
    for big in ('SGC0946', 'SGC8158'):
        assert df.loc[big, 'MW'] > 500


# --------------------------------------------------------------------------
# Docking protocol and validation (reviewer concern R4)
# --------------------------------------------------------------------------

def test_redocking_validation_passes():
    proto = json.loads(load('results/data/docking_protocol.json').read_text())
    val = proto['validation']
    assert val['verdict'] == 'PASS'
    assert val['heavy_atom_rmsd_A'] <= val['threshold_A']
    assert val['atoms_compared'] == 27


def test_docking_parameters_are_fully_specified():
    proto = json.loads(load('results/data/docking_protocol.json').read_text())
    for key in ('center_x', 'center_y', 'center_z', 'size_x', 'size_y', 'size_z'):
        assert proto['grid_box'][key] is not None
    for key in ('exhaustiveness', 'num_modes', 'energy_range', 'seed'):
        assert proto['search_parameters'][key] is not None
    assert 'Vina' in proto['software']['AutoDock Vina']


def test_pose_rmsd_pairs_atoms_by_name_not_index():
    """Regression: positional RMSD gave 8.1 A for a pose that is actually 0.8 A."""
    from utils import dock
    wd = ROOT / 'data/processed/docking/validation'
    if not (wd / 'SFG_redock_out.pdbqt').exists():
        pytest.skip('validation run not present')
    rmsd, n = dock.pose_rmsd(wd / 'SFG_crystal.pdb', wd / 'SFG_redock_out.pdbqt')
    assert n == 27
    assert rmsd < 2.0


# --------------------------------------------------------------------------
# Hypothesis testing (reviewer concern R7)
# --------------------------------------------------------------------------

def test_affinity_null_hypothesis_is_rejected():
    t = json.loads(load('results/data/hypothesis_tests.json').read_text())
    assert t['binding_affinity']['anova_p'] < 0.05


def test_ligand_efficiency_reverses_the_affinity_ranking():
    eff = pd.read_csv(load('results/tables/ligand_efficiency.csv')).set_index('Ligand')
    # The three largest inhibitors lead on raw score but trail on efficiency.
    for big in ('STM957', 'SGC0946', 'SGC8158'):
        assert eff.loc[big, 'Rank_by_affinity'] <= 3
        assert eff.loc[big, 'Rank_by_LE'] >= 5


def test_dunnett_control_is_sinefungin():
    t = json.loads(load('results/data/hypothesis_tests.json').read_text())
    for metric in ('binding_affinity', 'ligand_efficiency'):
        assert t[metric]['control'] == 'Sinefungin'
        assert len(t[metric]['comparisons']) == 6


# --------------------------------------------------------------------------
# Flexibility simulations (reviewer concerns R1, R2)
# --------------------------------------------------------------------------

def test_rmsf_profiles_cannot_be_ranked():
    r = json.loads(load('results/data/rmsf_test.json').read_text())
    assert r['p'] > 0.05
    assert r['ranking_supportable'] is False
    assert r['mean_range_A'] < r['smallest_reported_SD_A']


# --------------------------------------------------------------------------
# Structural alignment (reviewer concerns R6, R9)
# --------------------------------------------------------------------------

def test_nsp16_is_the_closer_structural_match():
    align = json.loads(load('results/data/structural_alignment.json').read_text())
    by_target = {a['target']: a for a in align}
    nsp14, nsp16 = by_target['SARS-CoV-2 nsp14'], by_target['SARS-CoV-2 nsp16']
    assert nsp16['equivalent_residue_pairs'] > nsp14['equivalent_residue_pairs']
    assert nsp16['ca_rmsd_over_equivalent_pairs_A'] < nsp14['ca_rmsd_over_equivalent_pairs_A']
    assert nsp16['structure_based_identity_pct'] > nsp14['structure_based_identity_pct']


# --------------------------------------------------------------------------
# Manuscript integrity (harness requirements G, H, I; reviewer concerns R10, R12)
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def manuscript_blocks():
    from docx import Document
    from utils.docx_io import iter_block_items
    if not MANUSCRIPT.exists():
        pytest.skip('manuscript not present')
    return list(iter_block_items(Document(str(MANUSCRIPT))))


def manuscript_text(blocks):
    out = []
    for kind, item in blocks:
        if kind == 'p':
            out.append(item.text)
    return '\n'.join(out)


def test_enzyme_names_no_longer_used_as_compound_labels(manuscript_blocks):
    text = manuscript_text(manuscript_blocks)
    assert 'DOTL1' not in text                      # the reviewer's reported typo
    assert 'SGC0946' in text and 'SGC8158' in text


def test_pdb_accessions_are_present(manuscript_blocks):
    text = manuscript_text(manuscript_blocks)
    for acc in ('8B07', '7TW7', '6YZ1'):
        assert acc in text, acc


def test_limitations_section_exists(manuscript_blocks):
    heads = [i.text.strip().upper() for k, i in manuscript_blocks if k == 'p']
    assert 'LIMITATIONS' in heads
    assert heads.index('LIMITATIONS') < heads.index('CONCLUSION')


def test_no_em_or_en_dashes_in_body_text(manuscript_blocks):
    """Body prose only. The reference list is excluded: en dashes in page ranges
    are standard bibliographic typography, and references are out of scope for the
    humanisation pass."""
    body = []
    for kind, item in manuscript_blocks:
        if kind != 'p':
            continue
        if item.text.strip().upper() == 'REFERENCES':
            break
        body.append(item.text)
    text = '\n'.join(body)
    for ch in ('—', '–'):
        assert ch not in text, f'{ch!r} present in body text'


def test_citations_and_references_agree_both_ways(manuscript_blocks):
    """Harness G, H and I: no orphan citations, no uncited entries, no bare paragraphs."""
    import subprocess
    out = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/10_reference_audit.py'),
         '--docx', str(MANUSCRIPT)],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    report = json.loads((ROOT / 'results/tables/reference_audit.json').read_text())
    assert report['cited_but_not_listed'] == []
    assert report['listed_but_never_cited'] == []
    assert report['year_mismatch_text_vs_list'] == []
    assert report['paragraphs_without_citation'] == []


def test_every_reference_doi_resolved():
    val = json.loads(load('results/data/reference_validation.json').read_text())
    unresolved = [r for r in val if r['doi'] and r['doi_valid'] is False]
    assert unresolved == [], unresolved
