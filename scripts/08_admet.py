#!/usr/bin/env python3
"""
Module: 08_admet
Purpose: ADMET / developability assessment of the seven docked ligands (reviewer
         concern R3: the submitted manuscript reports no ADMET assessment at all).

         Two layers are computed:
           1. A deterministic physicochemical and structural-alert panel from RDKit.
              These are rule-based calculations, reproducible exactly, with published
              provenance for every threshold.
           2. Optional learned predictions from TorchDrug, when that library is
              importable. TorchDrug ships model architectures and MoleculeNet loaders
              rather than pretrained ADMET predictors, so this layer trains small
              classifiers on the MoleculeNet BBBP and ClinTox tasks. It is treated as
              indicative only and is reported separately from the rule-based panel.

         The distinction matters: rule-based descriptors are exact, whereas learned
         predictions on ~2000-compound training sets carry wide uncertainty and none
         of these ligands resemble the training chemistry closely.

Author: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif Nur Muhammad Ansori
Date: 2026
Parameters:
    - --skip-learned : compute the rule-based panel only
References:
    - Lipinski CA et al. (2001) Adv Drug Deliv Rev 46:3-26
    - Veber DF et al. (2002) J Med Chem 45:2615-2623
    - Ghose AK, Viswanadhan VN, Wendoloski JJ (1999) J Comb Chem 1:55-68
    - Egan WJ, Merz KM, Baldwin JJ (2000) J Med Chem 43:3867-3877 (Egan egg)
    - Baell JB, Holloway GA (2010) J Med Chem 53:2719-2740 (PAINS)
    - Brenk R et al. (2008) ChemMedChem 3:435-444 (unwanted fragments)
    - Daina A, Michielin O, Zoete V (2017) Sci Rep 7:42717 (SwissADME rules)
    - Ertl P, Schuffenhauer A (2009) J Cheminform 1:8 (synthetic accessibility)
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog('rdApp.*')


def build_catalog():
    params = FilterCatalogParams()
    for cat in (FilterCatalogParams.FilterCatalogs.PAINS,
                FilterCatalogParams.FilterCatalogs.BRENK,
                FilterCatalogParams.FilterCatalogs.NIH):
        params.AddCatalog(cat)
    return FilterCatalog(params)


def rule_panel(smiles, catalog):
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    mr = Crippen.MolMR(mol)
    heavy = mol.GetNumHeavyAtoms()
    arom = len([r for r in mol.GetRingInfo().AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)])
    fcsp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    lipinski = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    veber = rotb <= 10 and tpsa <= 140
    ghose = (160 <= mw <= 480 and -0.4 <= logp <= 5.6
             and 40 <= mr <= 130 and 20 <= heavy <= 70)
    egan = tpsa <= 131.6 and logp <= 5.88
    # Lipinski-style oral absorption proxy (Daina et al. 2017): TPSA and logP
    # jointly place a compound inside or outside the "BOILED-Egg" white region.
    passive_gi = 71 <= tpsa <= 142 and -1 <= logp <= 6

    alerts = catalog.GetMatches(mol)
    alert_names = sorted({m.GetDescription() for m in alerts})

    return {
        'MW': round(mw, 2), 'cLogP': round(logp, 2), 'TPSA_A2': round(tpsa, 1),
        'HBD': hbd, 'HBA': hba, 'RotB': rotb, 'MolarRefractivity': round(mr, 1),
        'HeavyAtoms': heavy, 'AromaticRings': arom, 'FractionCsp3': round(fcsp3, 3),
        'Lipinski_violations': lipinski,
        'Veber_pass': veber, 'Ghose_pass': ghose, 'Egan_pass': egan,
        'Passive_GI_absorption_likely': passive_gi,
        'Structural_alerts_n': len(alert_names),
        'Structural_alerts': '; '.join(alert_names[:4]) if alert_names else 'none',
    }


def learned_panel(smiles_list, names):
    """Train small MoleculeNet classifiers with TorchDrug; indicative only."""
    try:
        import torch
        from torchdrug import data, datasets, models, tasks, core
    except Exception as exc:                       # noqa: BLE001
        return None, f'TorchDrug unavailable ({type(exc).__name__}: {exc})'

    try:
        results = {n: {} for n in names}
        for ds_name, loader, task_key in (
                ('BBBP', datasets.BBBP, 'p_np'),
                ('ClinTox', datasets.ClinTox, 'CT_TOX')):
            ds = loader('data/processed/moleculenet',
                        node_feature='pretrain', edge_feature='pretrain')
            train, valid, test = ds.split()
            model = models.GIN(input_dim=ds.node_feature_dim,
                               hidden_dims=[128, 128], short_cut=True,
                               batch_norm=True, readout='mean')
            task = tasks.PropertyPrediction(
                model, task=[task_key], criterion='bce',
                metric=('auroc',), verbose=0)
            optimiser = torch.optim.Adam(task.parameters(), lr=1e-3)
            solver = core.Engine(task, train, valid, test,
                                 optimiser, batch_size=64, log_interval=1e9)
            solver.train(num_epoch=5)
            metric = solver.evaluate('valid')

            mols = data.PackedMolecule.from_smiles(
                smiles_list, node_feature='pretrain', edge_feature='pretrain')
            with torch.no_grad():
                pred = torch.sigmoid(task.predict({'graph': mols})).squeeze(-1)
            for n, v in zip(names, pred.tolist()):
                results[n][f'{ds_name}_prob'] = round(float(v), 3)
                results[n][f'{ds_name}_valid_auroc'] = round(
                    float(list(metric.values())[0]), 3)
        return results, None
    except Exception as exc:                       # noqa: BLE001
        return None, f'TorchDrug run failed ({type(exc).__name__}: {exc})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-learned', action='store_true')
    args = ap.parse_args()

    lig = pd.read_csv('results/tables/ligand_identity.csv')
    catalog = build_catalog()

    rows = []
    for _, r in lig.iterrows():
        rows.append({'Ligand': r['Correct_compound'],
                     **rule_panel(r['Canonical_SMILES'], catalog)})
    df = pd.DataFrame(rows)

    note = 'TorchDrug layer skipped by request'
    if not args.skip_learned:
        learned, note = learned_panel(list(lig['Canonical_SMILES']),
                                      list(lig['Correct_compound']))
        if learned:
            df = df.merge(
                pd.DataFrame([{'Ligand': k, **v} for k, v in learned.items()]),
                on='Ligand', how='left')
            note = 'TorchDrug GIN classifiers trained on MoleculeNet BBBP and ClinTox'

    Path('results/tables').mkdir(parents=True, exist_ok=True)
    df.to_csv('results/tables/admet.csv', index=False)
    Path('results/data/admet_provenance.json').write_text(json.dumps(
        {'rule_based': 'RDKit descriptors + PAINS/BRENK/NIH filter catalogues',
         'learned_layer': note}, indent=2))

    cols = ['Ligand', 'MW', 'cLogP', 'TPSA_A2', 'Lipinski_violations', 'Veber_pass',
            'Ghose_pass', 'Egan_pass', 'Passive_GI_absorption_likely',
            'Structural_alerts_n']
    print(df[cols].to_string(index=False))
    print(f'\nlearned layer: {note}')
    print('-> results/tables/admet.csv')


if __name__ == '__main__':
    main()
