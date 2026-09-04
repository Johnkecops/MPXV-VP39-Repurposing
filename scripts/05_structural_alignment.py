#!/usr/bin/env python3
"""
Module: 05_structural_alignment
Purpose: Full-domain structural superposition and sequence identity for MPXV VP39
         against both SARS-CoV-2 cap methyltransferases.

         Reviewer concern R6: VP39 is a 2'-O-MTase, so its functional counterpart in
         SARS-CoV-2 is nsp16 (in the nsp10-nsp16 complex), not nsp14, which is an
         N7-MTase. Both comparisons are computed here so the nsp14 comparison can be
         retained as a SAM-pocket architecture comparison while the functionally
         matched nsp16 comparison is added.

         Reviewer concern R9: the submitted manuscript reports RMSD over 9 atoms and
         calls it "moderate similarity" against its own 2 A cutoff. Full-domain CE
         superposition over all aligned Ca atoms replaces that.

Author: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif Nur Muhammad Ansori
Date: 2026
Parameters:
    - --pocket-radius: heavy-atom cutoff (A) defining the cofactor pocket (default 5.0)
References:
    - Shindyalov IN, Bourne PE (1998) Protein Eng 11:739-747 (combinatorial extension)
    - Silhan J et al. (2023) Nat Commun 14:2259
    - Krafcikova P et al. (2020) Nat Commun 11:3717
    - Kottur J et al. (2023) PLoS Pathog 19:e1011546
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser, Selection
from Bio.PDB.cealign import CEAligner
from Bio.PDB.Polypeptide import is_aa, index_to_one, three_to_index

warnings.filterwarnings('ignore')

# name -> (pdb id, chain, residue range to keep, co-crystal ligand resname)
# 7TW7/8FRJ are TELSAM(ETV6)-nsp14 chimeras: the fusion tag occupies the low residue
# numbers, the nsp14 N7-MTase core keeps native nsp14 numbering (~288-527).
TARGETS = {
    'MPXV VP39':          ('8B07', 'A', None,        'SFG'),
    'SARS-CoV-2 nsp14':   ('7TW7', 'A', (250, 600),  'SAM'),
    'SARS-CoV-2 nsp16':   ('6YZ1', 'A', None,        'SFG'),
}


def load_chain(pdb_id, chain_id, resid_range):
    """Single-chain, standard-residues-only structure.

    Other chains are removed because CEAligner superposes whole structures: leaving
    the second VP39 copy in 8B07, or nsp10 in the 6YZ1 nsp10-nsp16 complex, lets
    unrelated chains drive the superposition.
    """
    parser = PDBParser(QUIET=True)
    path = Path('data/raw/pdb') / f'{pdb_id}.pdb'
    structure = parser.get_structure(pdb_id, path)
    model = structure[0]
    for ch in [c.id for c in model if c.id != chain_id]:
        model.detach_child(ch)
    chain = model[chain_id]
    drop = []
    for res in chain:
        if not is_aa(res, standard=True):
            drop.append(res.id)
        elif resid_range and not (resid_range[0] <= res.id[1] <= resid_range[1]):
            drop.append(res.id)
    for rid in drop:
        chain.detach_child(rid)
    return structure, chain


def sequence_of(chain):
    seq, resids = [], []
    for res in chain:
        try:
            seq.append(index_to_one(three_to_index(res.get_resname())))
            resids.append(res.id[1])
        except (KeyError, ValueError):
            continue
    return ''.join(seq), resids


def sequence_identity(seq_a, seq_b, mode='global'):
    """Pairwise sequence identity over aligned columns (BLOSUM62, gap -11/-1).

    These two proteins are remote structural homologues with negligible sequence
    similarity, so the sequence-level numbers are reported for completeness only.
    The structure-based identity below is the informative measure.
    """
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = mode
    aln = aligner.align(seq_a, seq_b)[0]
    a, b = aln[0], aln[1]
    cols = sum(1 for x, y in zip(a, b) if x != '-' and y != '-')
    ident = sum(1 for x, y in zip(a, b) if x == y and x != '-')
    return {
        f'{mode}_aligned_columns': cols,
        f'{mode}_sequence_identity_pct': round(100 * ident / cols, 1) if cols else None,
        f'{mode}_alignment_score': round(float(aln.score), 1),
    }


def structure_based_identity(chain_ref, chain_tgt, cutoff=5.0):
    """Identity over structurally equivalent residues after CE superposition.

    A residue pair is structurally equivalent when the two Ca atoms fall within
    `cutoff` of each other and each is the other's nearest Ca. This is the standard
    way to quote identity for remote homologues, and it is defined over the whole
    domain rather than over a handful of hand-picked atoms.
    """
    def ca_list(chain):
        out = []
        for res in chain:
            if is_aa(res, standard=True) and 'CA' in res:
                try:
                    out.append((index_to_one(three_to_index(res.get_resname())),
                                res.id[1], np.asarray(res['CA'].coord)))
                except (KeyError, ValueError):
                    continue
        return out

    ref, tgt = ca_list(chain_ref), ca_list(chain_tgt)
    if not ref or not tgt:
        return {}
    ref_xyz = np.asarray([c for _, _, c in ref])
    tgt_xyz = np.asarray([c for _, _, c in tgt])
    dist = np.linalg.norm(ref_xyz[:, None, :] - tgt_xyz[None, :, :], axis=-1)

    ref_nn = dist.argmin(axis=1)
    tgt_nn = dist.argmin(axis=0)
    pairs = [(i, j) for i, j in enumerate(ref_nn)
             if tgt_nn[j] == i and dist[i, j] <= cutoff]
    if not pairs:
        return {'equivalent_residue_pairs': 0}

    ident = sum(1 for i, j in pairs if ref[i][0] == tgt[j][0])
    rms = float(np.sqrt(np.mean([dist[i, j] ** 2 for i, j in pairs])))
    return {
        'equivalent_residue_pairs': len(pairs),
        'equivalent_pair_cutoff_A': cutoff,
        'structure_based_identity_pct': round(100 * ident / len(pairs), 1),
        'ca_rmsd_over_equivalent_pairs_A': round(rms, 3),
        'coverage_of_reference_pct': round(100 * len(pairs) / len(ref), 1),
    }


def pocket_residues(pdb_id, chain_id, ligand_resname, radius):
    """Residues with any heavy atom within `radius` of the co-crystallised cofactor."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, Path('data/raw/pdb') / f'{pdb_id}.pdb')
    model = structure[0]
    lig = [a for r in model[chain_id] if r.id[0] != ' '
           and r.get_resname() == ligand_resname
           for a in r if a.element != 'H']
    if not lig:
        return []
    lig_xyz = np.asarray([a.coord for a in lig])
    hits = []
    for res in model[chain_id]:
        if not is_aa(res, standard=True):
            continue
        xyz = np.asarray([a.coord for a in res if a.element != 'H'])
        if xyz.size and np.min(np.linalg.norm(
                xyz[:, None, :] - lig_xyz[None, :, :], axis=-1)) <= radius:
            hits.append(f'{res.get_resname()}{res.id[1]}')
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pocket-radius', type=float, default=5.0)
    args = ap.parse_args()

    loaded, seqs, pockets = {}, {}, {}
    for name, (pdb_id, chain_id, rng, lig) in TARGETS.items():
        structure, chain = load_chain(pdb_id, chain_id, rng)
        loaded[name] = (structure, chain, pdb_id)
        seq, resids = sequence_of(chain)
        seqs[name] = seq
        pockets[name] = pocket_residues(pdb_id, chain_id, lig, args.pocket_radius)
        print(f'{name:20s} {pdb_id}:{chain_id}  {len(seq)} residues  '
              f'({resids[0]}-{resids[-1]})  pocket: {len(pockets[name])} residues '
              f'around {lig}')

    ref_name = 'MPXV VP39'
    results = []
    for other in ('SARS-CoV-2 nsp14', 'SARS-CoV-2 nsp16'):
        aligner = CEAligner()
        aligner.set_reference(loaded[ref_name][0])
        aligner.align(loaded[other][0])
        rms = float(aligner.rms)

        # CEAligner transforms the target coordinates in place, so structural
        # equivalence is evaluated after this call.
        rec = {
            'reference': ref_name,
            'target': other,
            'reference_pdb': loaded[ref_name][2],
            'target_pdb': loaded[other][2],
            'ce_rmsd_A': round(rms, 3),
            'reference_residues': len(seqs[ref_name]),
            'target_residues': len(seqs[other]),
            **structure_based_identity(loaded[ref_name][1], loaded[other][1]),
            **sequence_identity(seqs[ref_name], seqs[other], mode='global'),
            **sequence_identity(seqs[ref_name], seqs[other], mode='local'),
            'reference_pocket_residues': pockets[ref_name],
            'target_pocket_residues': pockets[other],
        }
        results.append(rec)
        print(f"\n{ref_name} vs {other}")
        print(f"  CE structural RMSD           : {rec['ce_rmsd_A']:.3f} A")
        print(f"  equivalent residue pairs     : {rec['equivalent_residue_pairs']} "
              f"(Ca within {rec['equivalent_pair_cutoff_A']} A, "
              f"{rec['coverage_of_reference_pct']}% of VP39)")
        print(f"  Ca RMSD over those pairs     : "
              f"{rec['ca_rmsd_over_equivalent_pairs_A']:.3f} A")
        print(f"  structure-based identity     : "
              f"{rec['structure_based_identity_pct']}%")
        print(f"  global sequence identity     : "
              f"{rec['global_sequence_identity_pct']}% over "
              f"{rec['global_aligned_columns']} columns")
        print(f"  cofactor-pocket residues     : {len(pockets[ref_name])} (VP39) vs "
              f"{len(pockets[other])} ({other.split()[-1]})")

    Path('results/data').mkdir(parents=True, exist_ok=True)
    Path('results/data/structural_alignment.json').write_text(
        json.dumps(results, indent=2))

    import pandas as pd
    pd.DataFrame([{k: v for k, v in r.items()
                   if not k.endswith('pocket_residues')} for r in results]).to_csv(
        'results/tables/structural_alignment.csv', index=False)
    print('\n-> results/tables/structural_alignment.csv')


if __name__ == '__main__':
    main()
