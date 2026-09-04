#!/usr/bin/env python3
"""
Module: 04_docking_runs
Purpose: Dock all seven ligands into the MPXV VP39 SAM pocket in three independent
         runs (different random seeds), reproducing the manuscript's design with the
         grid box and search parameters now fully specified.

Author: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif Nur Muhammad Ansori
Date: 2026
Parameters:
    - --runs  : number of independent runs per ligand (default 3)
    - --cpu   : Vina worker threads (default 4; this machine is a 4P+4E Apple M3)
References:
    - Eberhardt J et al. (2021) J Chem Inf Model 61:3891-3898
    - Ramirez D, Caballero J (2018) Molecules 23:1038
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import dock  # noqa: E402

SEEDS = [42, 20240904, 777]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--cpu', type=int, default=4)
    ap.add_argument('--workdir', default='data/processed/docking')
    args = ap.parse_args()

    wd = Path(args.workdir)
    protocol = json.loads(Path('results/data/docking_protocol.json').read_text())
    box = protocol['grid_box']
    center = (box['center_x'], box['center_y'], box['center_z'])

    lig_df = pd.read_csv('results/tables/ligand_identity.csv')
    out_dir = wd / 'runs'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in lig_df.iterrows():
        name = row['Correct_compound']
        stem = name.replace(' ', '_')
        lig = wd / 'ligands' / f'{stem}.pdbqt'
        affinities = []
        for i, seed in enumerate(SEEDS[:args.runs], start=1):
            modes = dock.vina(wd / 'VP39_receptor.pdbqt', lig, center,
                              out_dir / f'{stem}_run{i}.pdbqt',
                              out_dir / f'{stem}_run{i}.log',
                              seed=seed, cpu=args.cpu)
            affinities.append(modes[0][1])
        rows.append({
            'Ligand': name,
            'Manuscript_label': row['Manuscript_label'],
            'HeavyAtoms': int(row['HeavyAtoms']),
            **{f'Run_{i}': a for i, a in enumerate(affinities, start=1)},
        })
        print(f'{name:26s} ' + '  '.join(f'{a:6.2f}' for a in affinities))

    df = pd.DataFrame(rows)
    run_cols = [c for c in df.columns if c.startswith('Run_')]
    df['Mean_affinity'] = df[run_cols].mean(axis=1).round(3)
    df['SD'] = df[run_cols].std(axis=1, ddof=1).round(3)
    df = df.sort_values('Mean_affinity').reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df) + 1))

    Path('results/tables').mkdir(parents=True, exist_ok=True)
    df.to_csv('results/tables/docking_affinities.csv', index=False)
    print('\n' + df[['Rank', 'Ligand', 'Mean_affinity', 'SD']].to_string(index=False))
    print('\n-> results/tables/docking_affinities.csv')


if __name__ == '__main__':
    main()
