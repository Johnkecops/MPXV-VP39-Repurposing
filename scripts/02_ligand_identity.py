#!/usr/bin/env python3
"""
Module: 02_ligand_identity
Purpose: Resolve the true chemical identity of every docked ligand and compute the
         physicochemical descriptor panel used for the Lipinski/Veber assessment.

         Reviewer concern R5: "DOT1L" and "PRMT7" in the submitted manuscript are the
         human enzymes, not compounds. Kottur et al. (2023) used the DOT1L inhibitor
         SGC0946 and the PRMT7 inhibitor SGC8158, which are separate molecules. Ligand
         structures are taken from the PDB chemical component dictionary of the
         depositions those compounds were actually solved in, so the docked molecule is
         traceable to a primary structural source rather than to a name.

Author: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif Nur Muhammad Ansori
Date: 2026
Parameters:
    - --out: output directory for the ligand table (default results/tables)
References:
    - Lipinski CA et al. (2001) Adv Drug Deliv Rev 46:3-26
    - Veber DF et al. (2002) J Med Chem 45:2615-2623
    - Kottur J et al. (2023) PLoS Pathog 19:e1011546
    - Zilecka E et al. (2024) J Struct Biol X 10:100109
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

RDLogger.DisableLog('rdApp.*')
TIMEOUT = 45
PUBCHEM = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound'

# manuscript_label -> (correct name, PDB chem comp id or None, PubChem CID or None,
#                      source deposition, role)
LIGANDS = [
    ('Sinefungin (control)', 'Sinefungin',  'SFG',   65482,     '8B07',
     'Pan-MTase competitive inhibitor; co-crystallised reference'),
    ('STM957',               'STM957',      'A1IB6', None,      '9FEH',
     'nsp14 MTase inhibitor (Zilecka et al. 2024)'),
    ('DOT1L',                'SGC0946',     'AW2',   56962337,  '8FRJ',
     'DOT1L inhibitor used against nsp14 by Kottur et al. (2023)'),
    ('PRMT7',                'SGC8158',     'MJ7',   None,      '8FRK',
     'PRMT7 inhibitor used against nsp14 by Kottur et al. (2023)'),
    ('SS148',                'SS148',       None,    169494540, '-',
     'nsp10-nsp16 / nsp14 MTase inhibitor (Li et al. 2023)'),
    ('SAM',                  'S-adenosyl-L-methionine', 'SAM', 34755, '7TW7',
     'Native methyl donor; mechanistic reference, not a drug candidate'),
    ('SAH',                  'S-adenosyl-L-homocysteine', 'SAH', 439155, '-',
     'Native reaction by-product; mechanistic reference, not a drug candidate'),
]


def from_pdb_chemcomp(comp_id):
    r = requests.get(f'https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}',
                     timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()
    desc = d.get('rcsb_chem_comp_descriptor', {})
    return {
        'smiles': desc.get('SMILES_stereo') or desc.get('SMILES'),
        'inchikey': desc.get('InChIKey'),
        'pdb_name': d['chem_comp'].get('name'),
    }


def pubchem_by_cid(cid):
    r = requests.get(f'{PUBCHEM}/cid/{cid}/property/'
                     'ConnectivitySMILES,InChIKey,MolecularFormula/JSON', timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()['PropertyTable']['Properties'][0]


class LookupUnavailable(Exception):
    """A lookup could not be completed, as distinct from returning no result."""


def cid_from_inchikey(inchikey):
    """PubChem CID for an InChIKey.

    Returns None only when PubChem answered and had no such compound (404).
    Any other outcome raises LookupUnavailable: swallowing them turned a
    network failure into the published claim "not deposited" in a manuscript
    table, with nothing recording that the network, not PubChem, was the
    source of the answer.
    """
    if not inchikey:
        return None
    try:
        r = requests.get(f'{PUBCHEM}/inchikey/{inchikey}/cids/JSON', timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise LookupUnavailable(f'PubChem unreachable: {type(exc).__name__}')
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise LookupUnavailable(f'PubChem returned HTTP {r.status_code}')
    try:
        return r.json()['IdentifierList']['CID'][0]
    except (ValueError, KeyError, IndexError):
        raise LookupUnavailable('PubChem response not in the expected shape')


def descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'RDKit could not parse: {smiles}')
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    heavy = mol.GetNumHeavyAtoms()

    # Lipinski Rule of Five: MW <= 500, logP <= 5 (an upper bound only -- the
    # manuscript's "logP between 1 and 5" is a misstatement of the rule), HBD <= 5,
    # HBA <= 10. One violation is conventionally tolerated; two or more is a fail.
    lip = {
        'MW<=500': mw <= 500,
        'logP<=5': logp <= 5,
        'HBD<=5': hbd <= 5,
        'HBA<=10': hba <= 10,
    }
    n_viol = sum(1 for v in lip.values() if not v)
    # Veber: rotatable bonds <= 10 and TPSA <= 140 A^2
    veber = rotb <= 10 and tpsa <= 140

    return {
        'MW': round(mw, 2), 'cLogP': round(logp, 2), 'HBD': hbd, 'HBA': hba,
        # TPSA_A2, matching 08_admet.py: the two manuscript tables previously
        # labelled the same quantity 'TPSA' and 'TPSA_A2'.
        'RotB': rotb, 'TPSA_A2': round(tpsa, 1), 'HeavyAtoms': heavy,
        'Lipinski_violations': n_viol,
        'Lipinski_failed_criteria': ', '.join(k for k, v in lip.items() if not v) or 'none',
        'Veber_pass': veber,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/tables')
    args = ap.parse_args()

    rows = []
    for label, name, comp, cid, dep, role in LIGANDS:
        rec = {'Manuscript_label': label, 'Correct_compound': name,
               'PDB_chem_comp': comp or '-', 'Source_deposition': dep, 'Role': role}

        smiles = inchikey = None
        if comp:
            info = from_pdb_chemcomp(comp)
            smiles, inchikey = info['smiles'], info['inchikey']
            rec['PDB_chem_comp_name'] = info['pdb_name']
        lookup_error = None
        if cid is None:
            try:
                cid = cid_from_inchikey(inchikey)
            except LookupUnavailable as exc:
                lookup_error = str(exc)
            time.sleep(0.25)        # between the two PubChem calls, not after both
        if smiles is None and cid:
            p = pubchem_by_cid(cid)
            smiles, inchikey = p['ConnectivitySMILES'], p['InChIKey']

        if lookup_error:
            # Distinct from 'not deposited': we did not get an answer. Writing
            # 'not deposited' here would publish a claim about PubChem's
            # contents that rests on the state of the network that day.
            rec['PubChem_CID'] = 'lookup unavailable'
            rec['PubChem_lookup_error'] = lookup_error
            print(f'  [warn] {name}: {lookup_error}')
        else:
            rec['PubChem_CID'] = cid if cid else 'not deposited'
        rec['Canonical_SMILES'] = smiles
        rec['InChIKey'] = inchikey
        if not smiles:
            raise SystemExit(
                f'no SMILES for {name}: the PDB chem comp carried none and the '
                f'PubChem lookup did not supply one'
                + (f' ({lookup_error})' if lookup_error else ''))
        rec.update(descriptors(smiles))
        rows.append(rec)
        print(f"{label:22s} -> {name:26s} CID {str(rec['PubChem_CID']):>12s} "
              f"MW {rec['MW']:7.2f}  Lipinski viol {rec['Lipinski_violations']}")
        time.sleep(0.25)   # NCBI courtesy rate limit

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'ligand_identity.csv', index=False)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    # SMILES first, then the name: that is the order `obabel -ismi` and every
    # other .smi reader expects. Reversed, the name is parsed as the structure.
    df[['Canonical_SMILES', 'Correct_compound']].to_csv(
        'data/processed/ligands.smi', sep='\t', index=False, header=False)
    (out / 'ligand_identity.json').write_text(json.dumps(rows, indent=2))
    print(f'\n-> {out}/ligand_identity.csv  ({len(df)} ligands)')


if __name__ == '__main__':
    main()
