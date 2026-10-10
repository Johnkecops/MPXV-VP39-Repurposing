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
import time
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
    # Check BOTH cached names before downloading: testing only for .pdb meant a
    # cif-only entry re-requested the .pdb (404) and re-downloaded the whole
    # .cif on every single run.
    pdb_path = out_dir / f'{pdb_id}.pdb'
    cif_path = out_dir / f'{pdb_id}.cif'
    path = next((p for p in (pdb_path, cif_path) if p.exists()), None)
    if path is None:
        path = pdb_path
        r = requests.get(f'https://files.rcsb.org/download/{pdb_id}.pdb', timeout=TIMEOUT)
        if r.status_code == 404:
            path = cif_path
            r = requests.get(f'https://files.rcsb.org/download/{pdb_id}.cif', timeout=TIMEOUT)
        r.raise_for_status()
        path.write_bytes(r.content)

    meta_r = requests.get(
        f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}', timeout=TIMEOUT)
    # Without this, a 503 maintenance page or a captive-portal redirect raised a
    # bare JSONDecodeError naming neither the URL nor the status code.
    meta_r.raise_for_status()
    try:
        meta = meta_r.json()
    except ValueError as exc:
        raise RuntimeError(f'RCSB returned a non-JSON body for {pdb_id} '
                           f'(HTTP {meta_r.status_code})') from exc
    cite = meta.get('rcsb_primary_citation', {}) or {}
    info = meta.get('rcsb_entry_info', {}) or {}
    res = info.get('resolution_combined') or []
    return {
        'pdb_id': pdb_id,
        'file': str(path),
        'title': meta.get('struct', {}).get('title'),
        'method': [m.get('method') for m in (meta.get('exptl') or [])],
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

    records, failed = [], []
    for pdb_id, role in ENTRIES.items():
        # One transient failure used to abort main() and discard every record
        # already fetched, including the metadata the Methods needs. Collect
        # what succeeded, write it, and report the rest with a non-zero exit.
        try:
            rec = fetch(pdb_id, out_dir)
        except (requests.RequestException, RuntimeError) as exc:
            failed.append((pdb_id, f'{type(exc).__name__}: {exc}'))
            print(f'{pdb_id}  FAILED  {type(exc).__name__}: {exc}')
            continue
        rec['role'] = role
        records.append(rec)
        # title is Optional in the RCSB schema; subscripting None raised a
        # bare TypeError here.
        title = rec['title'] or '(no title in RCSB record)'
        print(f"{rec['pdb_id']}  {rec['resolution_A']} A  {title[:64]}")
        time.sleep(0.25)        # courtesy pause between entries

    Path('results/data').mkdir(parents=True, exist_ok=True)
    Path('results/data/structure_provenance.json').write_text(
        json.dumps(records, indent=2))
    print(f'\n{len(records)} entries -> results/data/structure_provenance.json')
    if failed:
        raise SystemExit('could not fetch: '
                         + '; '.join(f'{i} ({e})' for i, e in failed))


if __name__ == '__main__':
    main()
