#!/usr/bin/env python3
"""
Module: utils.dock
Purpose: Shared receptor/ligand preparation and AutoDock Vina invocation.
Author: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif Nur Muhammad Ansori
Date: 2026
References:
    - Eberhardt J, Santos-Martins D, Tillack AF, Forli S (2021) AutoDock Vina 1.2.0.
      J Chem Inf Model 61:3891-3898
    - O'Boyle NM et al. (2011) Open Babel. J Cheminform 3:33
"""
import subprocess
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# PyRx exposes AutoDock Vina with these defaults; they are held fixed so the
# validation run and the production runs share one parameter set.
DEFAULTS = dict(exhaustiveness=8, num_modes=9, energy_range=3, box_size=25.0)


def run(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True,
                          capture_output=True, text=True, **kw)


def strip_receptor(pdb_in, pdb_out, chain='A'):
    """Keep protein ATOM records of one chain; drop waters, ions and ligands."""
    keep = []
    for line in Path(pdb_in).read_text().splitlines():
        if line.startswith('ATOM') and line[21] == chain:
            keep.append(line)
        elif line.startswith('TER') and len(line) > 21 and line[21] == chain:
            keep.append(line)
    Path(pdb_out).write_text('\n'.join(keep) + '\nEND\n')
    return len(keep)


def extract_hetatm(pdb_in, resname, pdb_out, chain='A'):
    """Pull one heteroatom residue (the co-crystallised ligand) out of a structure."""
    keep = [ln for ln in Path(pdb_in).read_text().splitlines()
            if ln.startswith('HETATM') and ln[17:20].strip() == resname
            and ln[21] == chain and ln[76:78].strip() != 'H']
    if not keep:
        raise ValueError(f'{resname} not found in chain {chain} of {pdb_in}')
    Path(pdb_out).write_text('\n'.join(keep) + '\nEND\n')
    return len(keep)


def centroid(pdb_file):
    xyz = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
           for l in Path(pdb_file).read_text().splitlines()
           if l.startswith(('ATOM', 'HETATM'))]
    return np.asarray(xyz).mean(axis=0)


def receptor_pdbqt(pdb_in, pdbqt_out, ph=7.4):
    """Rigid receptor: add hydrogens at physiological pH, assign Gasteiger charges."""
    run(['obabel', '-ipdb', str(pdb_in), '-opdbqt', '-O', str(pdbqt_out),
         '-xr', '-p', str(ph), '--partialcharge', 'gasteiger'])
    return pdbqt_out


def ligand_pdbqt_from_smiles(smiles, name, out_dir, seed=0xF00D):
    """3D conformer from SMILES: ETKDGv3 embedding, MMFF94s minimisation, then pdbqt."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s', maxIters=2000)
    sdf = out_dir / f'{name}.sdf'
    Chem.SDWriter(str(sdf)).write(mol)
    pdbqt = out_dir / f'{name}.pdbqt'
    run(['obabel', '-isdf', str(sdf), '-opdbqt', '-O', str(pdbqt),
         '--partialcharge', 'gasteiger'])
    return pdbqt


def ligand_pdbqt_from_pdb(pdb_in, name, out_dir, ph=7.4):
    """Prepare a ligand starting from crystallographic coordinates."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdbqt = out_dir / f'{name}.pdbqt'
    run(['obabel', '-ipdb', str(pdb_in), '-opdbqt', '-O', str(pdbqt),
         '-p', str(ph), '--partialcharge', 'gasteiger'])
    return pdbqt


def vina(receptor, ligand, center, out_pdbqt, log, seed,
         size=None, exhaustiveness=None, num_modes=None, energy_range=None, cpu=4):
    size = size or DEFAULTS['box_size']
    cmd = ['vina',
           '--receptor', str(receptor), '--ligand', str(ligand),
           '--center_x', f'{center[0]:.3f}', '--center_y', f'{center[1]:.3f}',
           '--center_z', f'{center[2]:.3f}',
           '--size_x', f'{size}', '--size_y', f'{size}', '--size_z', f'{size}',
           '--exhaustiveness', str(exhaustiveness or DEFAULTS['exhaustiveness']),
           '--num_modes', str(num_modes or DEFAULTS['num_modes']),
           '--energy_range', str(energy_range or DEFAULTS['energy_range']),
           '--seed', str(seed), '--cpu', str(cpu),
           '--out', str(out_pdbqt)]
    res = run(cmd)
    Path(log).write_text(res.stdout)
    return parse_vina_log(res.stdout)


def parse_vina_log(text):
    """Return list of (mode, affinity kcal/mol) from Vina stdout."""
    modes, started = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('mode |'):
            started = True
            continue
        if started:
            parts = s.split()
            if len(parts) >= 2 and parts[0].isdigit():
                modes.append((int(parts[0]), float(parts[1])))
    return modes


def named_heavy_atoms(path, models=False):
    """{atom_name: (x, y, z)} for heavy atoms.

    Vina rebuilds the ligand from a torsion tree, so output atom order does not
    follow input order. PDBQT does preserve the PDB atom names, so poses are matched
    on name rather than on index.
    """
    coords, in_model = {}, not models
    for line in Path(path).read_text().splitlines():
        if models and line.startswith('MODEL'):
            if coords:
                break
            in_model = True
            continue
        if line.startswith('ENDMDL'):
            break
        if in_model and line.startswith(('ATOM', 'HETATM')):
            el = (line[76:78].strip() or line[12:16].strip()[0]).upper()
            if el.startswith('H'):
                continue
            coords[line[12:16].strip()] = (float(line[30:38]),
                                           float(line[38:46]),
                                           float(line[46:54]))
    return coords


def pose_rmsd(reference_pdb, pose_pdbqt):
    """Heavy-atom RMSD between a crystal pose and Vina's top-ranked pose.

    Returns (rmsd, n_atoms_matched). Atoms are paired by PDB atom name.
    """
    ref = named_heavy_atoms(reference_pdb)
    pose = named_heavy_atoms(pose_pdbqt, models=True)
    shared = sorted(set(ref) & set(pose))
    if not shared:
        raise RuntimeError('no atom names in common between crystal and docked pose')
    missing = (set(ref) | set(pose)) - set(shared)
    if missing:
        raise RuntimeError(f'unmatched ligand atoms: {sorted(missing)}')
    a = np.asarray([ref[n] for n in shared])
    b = np.asarray([pose[n] for n in shared])
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean())), len(shared)
