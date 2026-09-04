#!/usr/bin/env python3
"""
Module: 03_prepare_and_validate
Purpose: Prepare the MPXV VP39 receptor and all seven ligands, then run the docking
         protocol validation the reviewer asked for: redock the co-crystallised
         sinefungin from 8B07 and report heavy-atom RMSD against the crystal pose.

         Reviewer concern R4. The manuscript reports PyRx; PyRx is a graphical wrapper
         around AutoDock Vina and is not available for macOS arm64. The validation is
         therefore run on AutoDock Vina directly, which is the same engine and the same
         scoring function PyRx invokes, with PyRx's default search parameters held fixed.

Author: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif Nur Muhammad Ansori
Date: 2026
Parameters:
    - --receptor-pdb : source structure (default data/raw/pdb/8B07.pdb)
    - --chain        : receptor chain (default A)
    - --ligand-resname : co-crystallised ligand (default SFG, sinefungin)
    - --rmsd-threshold : pass criterion in Angstrom (default 2.0)
References:
    - Silhan J et al. (2023) Nat Commun 14:2259 (8B07)
    - Eberhardt J et al. (2021) J Chem Inf Model 61:3891-3898
    - Ramirez D, Caballero J (2018) Molecules 23:1038 (docking pose reliability)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import dock  # noqa: E402


def software_versions():
    def v(cmd, first=True):
        try:
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
            return out.strip().splitlines()[0] if first and out.strip() else out.strip()
        except Exception:
            return 'unavailable'
    import rdkit
    return {
        'AutoDock Vina': v('vina --version'),
        'Open Babel': v('obabel -V'),
        'RDKit': rdkit.__version__,
        'Python': sys.version.split()[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--receptor-pdb', default='data/raw/pdb/8B07.pdb')
    ap.add_argument('--chain', default='A')
    ap.add_argument('--ligand-resname', default='SFG')
    ap.add_argument('--rmsd-threshold', type=float, default=2.0)
    ap.add_argument('--workdir', default='data/processed/docking')
    args = ap.parse_args()

    wd = Path(args.workdir)
    (wd / 'ligands').mkdir(parents=True, exist_ok=True)
    (wd / 'validation').mkdir(parents=True, exist_ok=True)

    # --- receptor -----------------------------------------------------------
    rec_pdb = wd / 'VP39_receptor.pdb'
    n_atoms = dock.strip_receptor(args.receptor_pdb, rec_pdb, chain=args.chain)
    rec_pdbqt = dock.receptor_pdbqt(rec_pdb, wd / 'VP39_receptor.pdbqt')
    print(f'receptor: {n_atoms} protein atoms, chain {args.chain} -> {rec_pdbqt.name}')

    # --- grid box defined by the co-crystallised ligand ---------------------
    xtal_pdb = wd / 'validation' / f'{args.ligand_resname}_crystal.pdb'
    n_lig = dock.extract_hetatm(args.receptor_pdb, args.ligand_resname,
                                xtal_pdb, chain=args.chain)
    center = dock.centroid(xtal_pdb)
    box = dock.DEFAULTS['box_size']
    print(f'grid box: centre ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) A, '
          f'{box} x {box} x {box} A, defined on {n_lig} heavy atoms of '
          f'{args.ligand_resname}')

    # --- ligand set ---------------------------------------------------------
    lig_df = pd.read_csv('results/tables/ligand_identity.csv')
    prepared = {}
    for _, row in lig_df.iterrows():
        name = row['Correct_compound'].replace(' ', '_')
        prepared[row['Correct_compound']] = dock.ligand_pdbqt_from_smiles(
            row['Canonical_SMILES'], name, wd / 'ligands')
    print(f'ligands prepared: {len(prepared)}')

    # --- protocol validation: redock crystallographic sinefungin -----------
    val_lig = dock.ligand_pdbqt_from_pdb(xtal_pdb, 'SFG_redock', wd / 'validation')
    out_pdbqt = wd / 'validation' / 'SFG_redock_out.pdbqt'
    modes = dock.vina(rec_pdbqt, val_lig, center, out_pdbqt,
                      wd / 'validation' / 'SFG_redock.log', seed=42)

    rmsd, n_matched = dock.pose_rmsd(xtal_pdb, out_pdbqt)
    verdict = 'PASS' if rmsd <= args.rmsd_threshold else 'FAIL'
    print(f'\nredocking validation: top pose {modes[0][1]:.2f} kcal/mol, '
          f'heavy-atom RMSD {rmsd:.3f} A over {n_matched} atoms vs crystal pose '
          f'-> {verdict} (threshold {args.rmsd_threshold} A)')

    report = {
        'receptor': {'pdb_id': Path(args.receptor_pdb).stem, 'chain': args.chain,
                     'protein_atoms': n_atoms},
        'grid_box': {'center_x': round(float(center[0]), 3),
                     'center_y': round(float(center[1]), 3),
                     'center_z': round(float(center[2]), 3),
                     'size_x': box, 'size_y': box, 'size_z': box,
                     'defined_on': f'{args.ligand_resname} ({n_lig} heavy atoms)'},
        'search_parameters': dict(dock.DEFAULTS, seed=42),
        'validation': {'ligand': args.ligand_resname,
                       'top_pose_affinity_kcal_per_mol': modes[0][1],
                       'heavy_atom_rmsd_A': round(rmsd, 3),
                       'atoms_compared': n_matched,
                       'threshold_A': args.rmsd_threshold,
                       'verdict': verdict},
        'software': software_versions(),
    }
    Path('results/data').mkdir(parents=True, exist_ok=True)
    Path('results/data/docking_protocol.json').write_text(json.dumps(report, indent=2))
    print('-> results/data/docking_protocol.json')


if __name__ == '__main__':
    main()
