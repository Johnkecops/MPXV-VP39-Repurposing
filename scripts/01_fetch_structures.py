#!/usr/bin/env python3
"""
Module: 01_fetch_structures
Purpose: Download every PDB entry used in the revision and record provenance
         (title, method, resolution, primary citation) for the Methods section.
Author: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif Nur Muhammad Ansori
Date: 2026
Parameters:
    - --out: destination directory for coordinate files (default data/raw/pdb)
References:
    - Silhan et al. 2023 Nat Commun 14 (VP39, 8B07)
    - Kottur et al. 2023 PLoS Pathog 19:e1011546 (nsp14 7TW7/8FRJ/8FRK)
    - Krafcikova et al. 2020 Nat Commun 11:3717 (nsp10-nsp16, 6YZ1)
"""
import argparse
import json
from pathlib import Path

import requests

ENTRIES = {
    '8B07': 'MPXV VP39 2\'-O-MTase in complex with sinefungin (docking receptor)',
    '7TW7': 'SARS-CoV-2 nsp14 N7-MTase-TELSAM with SAM (manuscript comparison)',
    '6YZ1': 'SARS-CoV-2 nsp10-nsp16 2\'-O-MTase with sinefungin (added comparison)',
    '8FRJ': 'nsp14-TELSAM with SGC0946 (source of the "DOT1L" ligand)',
    '8FRK': 'nsp14-TELSAM with SGC8158 (source of the "PRMT7" ligand)',
    '9FEH': 'nsp14 with STM957-series inhibitor (source of the STM957 ligand)',
}
TIMEOUT = 60


def fetch(pdb_id, out_dir):
    pdb_id = pdb_id.upper()
    # Newer/large depositions have no legacy PDB format; fall back to mmCIF.
    path = out_dir / f'{pdb_id}.pdb'
    if not path.exists():
        r = requests.get(f'https://files.rcsb.org/download/{pdb_id}.pdb', timeout=TIMEOUT)
        if r.status_code == 404:
            path = out_dir / f'{pdb_id}.cif'
            r = requests.get(f'https://files.rcsb.org/download/{pdb_id}.cif', timeout=TIMEOUT)
        r.raise_for_status()
        path.write_bytes(r.content)

    meta = requests.get(
        f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}', timeout=TIMEOUT).json()
    cite = meta.get('rcsb_primary_citation', {}) or {}
    info = meta.get('rcsb_entry_info', {}) or {}
    res = info.get('resolution_combined') or []
    return {
        'pdb_id': pdb_id,
        'file': str(path),
        'title': meta.get('struct', {}).get('title'),
        'method': [m['method'] for m in meta.get('exptl', [])],
        'resolution_A': res[0] if res else None,
        'citation_title': cite.get('title'),
        'citation_journal': cite.get('rcsb_journal_abbrev'),
        'citation_year': cite.get('year'),
        'citation_doi': cite.get('pdbx_database_id_doi'),
        'citation_pubmed': cite.get('pdbx_database_id_pub_med'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/raw/pdb')
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for pdb_id, role in ENTRIES.items():
        rec = fetch(pdb_id, out_dir)
        rec['role'] = role
        records.append(rec)
        print(f"{rec['pdb_id']}  {rec['resolution_A']} A  {rec['title'][:64]}")

    Path('results/data').mkdir(parents=True, exist_ok=True)
    Path('results/data/structure_provenance.json').write_text(
        json.dumps(records, indent=2))
    print(f'\n{len(records)} entries -> results/data/structure_provenance.json')


if __name__ == '__main__':
    main()
