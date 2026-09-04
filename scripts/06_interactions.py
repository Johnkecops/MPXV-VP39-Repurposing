#!/usr/bin/env python3
"""
Module: 06_interactions
Purpose: Build the summary table of interacting residues for all seven ligands that
         the reviewer asked for (concern R8), from the top-ranked docked pose.

         The submitted manuscript reports contact counts from BIOVIA Discovery Studio
         and contradicts itself on the STM957 hydrogen-bond count (five in Results, six
         in Discussion). Contacts are recomputed here from the pose coordinates using
         published distance criteria, so a single number is derived from the data rather
         than chosen between two.

Author: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif Nur Muhammad Ansori
Date: 2026
Parameters:
    - --hbond-cutoff       : donor/acceptor heavy-atom distance, A (default 3.5)
    - --hydrophobic-cutoff : carbon-carbon distance, A (default 4.0)
    - --run                : which docking run to profile (default 1)
References:
    - Baker EN, Hubbard RE (1984) Prog Biophys Mol Biol 44:97-179 (H-bond geometry)
    - Bissantz C, Kuhn B, Stahl M (2010) J Med Chem 53:5061-5084 (contact criteria)
    - Salentin S et al. (2015) Nucleic Acids Res 43:W443-W447 (PLIP criteria)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

POLAR = {'N', 'O'}


def parse_receptor(pdbqt):
    """Heavy atoms of the receptor with residue labels."""
    atoms = []
    for line in Path(pdbqt).read_text().splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        el = (line[77:79].strip() or line[12:16].strip()[0]).upper()
        el = el[0] if el[:2] not in ('CL', 'BR') else el[:2]
        if el == 'H':
            continue
        atoms.append({
            'name': line[12:16].strip(),
            'resname': line[17:20].strip(),
            'resid': int(line[22:26]),
            'element': el,
            'xyz': (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        })
    return atoms


def parse_pose(pdbqt, model=1):
    """Heavy atoms of one MODEL in a Vina output file."""
    atoms, current, capture = [], 0, False
    for line in Path(pdbqt).read_text().splitlines():
        if line.startswith('MODEL'):
            current = int(line.split()[1])
            capture = (current == model)
            continue
        if line.startswith('ENDMDL'):
            if capture:
                break
            continue
        if capture and line.startswith(('ATOM', 'HETATM')):
            el = (line[77:79].strip() or line[12:16].strip()[0]).upper()
            el = el[0] if el[:2] not in ('CL', 'BR') else el[:2]
            if el == 'H':
                continue
            atoms.append({'name': line[12:16].strip(), 'element': el,
                          'xyz': (float(line[30:38]), float(line[38:46]),
                                  float(line[46:54]))})
    return atoms


def contacts(rec_atoms, lig_atoms, hb_cut, ph_cut):
    """Classify receptor-ligand contacts by element pair and distance.

    Hydrogen bonds are reported as putative: they are assigned on donor/acceptor
    heavy-atom distance without an explicit angular term, since the rigid-receptor
    PDBQT carries only polar hydrogens.
    """
    r_xyz = np.asarray([a['xyz'] for a in rec_atoms])
    l_xyz = np.asarray([a['xyz'] for a in lig_atoms])
    dist = np.linalg.norm(r_xyz[:, None, :] - l_xyz[None, :, :], axis=-1)

    per_residue = defaultdict(lambda: {'hbond': 0, 'hydrophobic': 0, 'min_dist': 99.0})
    n_hb = n_ph = 0
    cutoff = max(hb_cut, ph_cut)
    for i, j in zip(*np.where(dist <= cutoff)):
        d = float(dist[i, j])
        ra, la = rec_atoms[i], lig_atoms[j]
        key = f"{ra['resname']}{ra['resid']}"
        entry = per_residue[key]
        entry['min_dist'] = min(entry['min_dist'], d)
        if ra['element'] in POLAR and la['element'] in POLAR and d <= hb_cut:
            entry['hbond'] += 1
            n_hb += 1
        elif ra['element'] == 'C' and la['element'] == 'C' and d <= ph_cut:
            entry['hydrophobic'] += 1
            n_ph += 1
    return per_residue, n_hb, n_ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hbond-cutoff', type=float, default=3.5)
    ap.add_argument('--hydrophobic-cutoff', type=float, default=4.0)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--workdir', default='data/processed/docking')
    args = ap.parse_args()

    wd = Path(args.workdir)
    rec = parse_receptor(wd / 'VP39_receptor.pdbqt')
    lig_df = pd.read_csv('results/tables/ligand_identity.csv')

    rows, detail = [], {}
    for _, row in lig_df.iterrows():
        name = row['Correct_compound']
        pose_file = wd / 'runs' / f"{name.replace(' ', '_')}_run{args.run}.pdbqt"
        if not pose_file.exists():
            print(f'  [skip] {name}: {pose_file.name} not found')
            continue
        pose = parse_pose(pose_file)
        per_res, n_hb, n_ph = contacts(rec, pose,
                                       args.hbond_cutoff, args.hydrophobic_cutoff)
        ordered = sorted(per_res.items(), key=lambda kv: kv[1]['min_dist'])
        hb_res = [k for k, v in ordered if v['hbond']]
        rows.append({
            'Ligand': name,
            'Contacting_residues': len(per_res),
            'Hydrogen_bonds': n_hb,
            'H_bond_residues': ', '.join(hb_res),
            'Hydrophobic_contacts': n_ph,
            'Closest_residue': ordered[0][0] if ordered else '-',
            'Closest_distance_A': round(ordered[0][1]['min_dist'], 2) if ordered else None,
            'All_contacting_residues': ', '.join(k for k, _ in ordered),
        })
        detail[name] = {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv)
                            for kk, vv in v.items()} for k, v in per_res.items()}
        print(f'{name:26s} residues {len(per_res):3d}  H-bonds {n_hb:3d}  '
              f'hydrophobic {n_ph:3d}')

    df = pd.DataFrame(rows)
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    df.to_csv('results/tables/interaction_summary.csv', index=False)
    Path('results/data/interaction_detail.json').write_text(json.dumps(detail, indent=2))
    print(f'\ncriteria: H-bond N/O...N/O <= {args.hbond_cutoff} A, '
          f'hydrophobic C...C <= {args.hydrophobic_cutoff} A')
    print('-> results/tables/interaction_summary.csv')


if __name__ == '__main__':
    main()
