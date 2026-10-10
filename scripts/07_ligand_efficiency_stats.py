#!/usr/bin/env python3
"""
Module: 07_ligand_efficiency_stats
Purpose: Test the hypotheses the Introduction states but never tests (reviewer concern
         R7), using ligand efficiency rather than raw docking score.

         The Introduction sets a null hypothesis of "no significant difference in
         binding affinity ... compared to known ligands" and an alternative of a
         significant difference, then reports no test at all. Raw Vina scores also
         scale with molecular size, so the three largest ligands are flattered by the
         comparison: ligand efficiency normalises affinity per heavy atom and is the
         appropriate metric for ranking chemically dissimilar compounds.

Author: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif Nur Muhammad Ansori
Date: 2026
Parameters:
    - --alpha : family-wise significance level (default 0.05)
References:
    - Kenny PW (2019) The nature of ligand efficiency. J Cheminform 11:8.
      doi:10.1186/s13321-019-0330-2 (reviewer-supplied reference). Note that Kenny
      argues LE is defined against an arbitrary reference state (1 M standard
      concentration) and that efficiency rankings are therefore not scale-free; LE is
      used here as a size-normalised comparator, not as an absolute quality score.
    - Hopkins AL, Groom CR, Alex A (2004) Drug Discov Today 9:430-431 (original LE)
    - Dunnett CW (1955) J Am Stat Assoc 50:1096-1121
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CONTROL = 'Sinefungin'
# Dunnett's multivariate-t integration underflows to 0.0 well before the true
# p-value is zero. Below this the result is reported as an upper bound.
P_FLOOR = 1e-16


def cohens_d(a, b):
    """Hedges-corrected standardised mean difference for small samples."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                 / (na + nb - 2))
    if sp == 0:
        return 0.0
    d = (np.mean(a) - np.mean(b)) / sp
    j = 1 - 3 / (4 * (na + nb) - 9)          # Hedges' small-sample correction
    return float(d * j)


#  The interval that belongs beside a Dunnett p-value is Dunnett's own
#  simultaneous interval: it uses the residual pooled across all k groups
#  (df = N - k) and the multiplicity-adjusted critical value. A plain
#  two-sample t interval pools only the two groups compared (df = 2n - 2) and
#  applies no adjustment, so it can exclude zero for a comparison the p-value
#  calls non-significant. scipy computes the correct one, so there is nothing
#  to hand-roll here -- see DunnettResult.confidence_interval.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()

    dock = pd.read_csv('results/tables/docking_affinities.csv')
    run_cols = [c for c in dock.columns if c.startswith('Run_')]
    # With one run per ligand every within-group variance is undefined: ddof=1
    # gives NaN, which propagates silently into the F statistic, Hedges' g and
    # the p-values, and `NaN < alpha` is False, so nothing flags it.
    if len(run_cols) < 2:
        raise SystemExit(f'need at least 2 runs per ligand for a variance '
                         f'estimate, found {len(run_cols)}: {run_cols}')

    # --- ligand efficiency --------------------------------------------------
    # LE = -dG / N_heavy  (kcal/mol per heavy atom)
    le_cols = []
    for c in run_cols:
        col = f'LE_{c}'
        dock[col] = -dock[c] / dock['HeavyAtoms']
        le_cols.append(col)
    dock['Mean_LE'] = dock[le_cols].mean(axis=1).round(4)
    dock['SD_LE'] = dock[le_cols].std(axis=1, ddof=1).round(4)

    lig = pd.read_csv('results/tables/ligand_identity.csv')[
        ['Correct_compound', 'cLogP']].rename(columns={'Correct_compound': 'Ligand'})
    dock = dock.merge(lig, on='Ligand', how='left')
    # A left join on a drifted ligand name yields cLogP = NaN and then
    # LELP = NaN in the published table, with no error anywhere.
    missing = dock.loc[dock['cLogP'].isna(), 'Ligand'].tolist()
    if missing:
        raise SystemExit(f'no cLogP for {missing}: ligand names disagree between '
                         f'docking_affinities.csv and ligand_identity.csv')
    # LELP = logP / LE: penalises efficiency bought with lipophilicity
    dock['LELP'] = (dock['cLogP'] / dock['Mean_LE']).round(2)

    groups_aff = {r['Ligand']: np.asarray([r[c] for c in run_cols], float)
                  for _, r in dock.iterrows()}
    groups_le = {r['Ligand']: np.asarray([r[c] for c in le_cols], float)
                 for _, r in dock.iterrows()}

    results = {'alpha': args.alpha, 'n_runs_per_ligand': len(run_cols)}

    for label, groups, unit in (('binding_affinity', groups_aff, 'kcal/mol'),
                                ('ligand_efficiency', groups_le, 'kcal/mol/heavy atom')):
        names = list(groups)
        arrays = [groups[n] for n in names]
        f, p = stats.f_oneway(*arrays)
        # Levene's test guards the equal-variance assumption behind ANOVA and Dunnett
        lev_stat, lev_p = stats.levene(*arrays)

        ctrl = groups[CONTROL]
        treat_names = [n for n in names if n != CONTROL]
        d = stats.dunnett(*[groups[n] for n in treat_names], control=ctrl,
                          alternative='two-sided')

        ci = d.confidence_interval(confidence_level=1 - args.alpha)

        comparisons = []
        for n, stat, pv, lo, hi in zip(treat_names, d.statistic, d.pvalue,
                                       ci.low, ci.high):
            comparisons.append({
                'ligand': n,
                'mean_difference_vs_control': round(float(np.mean(groups[n])
                                                          - np.mean(ctrl)), 4),
                'ci_simultaneous_low': round(float(lo), 4),
                'ci_simultaneous_high': round(float(hi), 4),
                'hedges_g': round(cohens_d(groups[n], ctrl), 3),
                'dunnett_t': round(float(stat), 3),
                # .3g, not round(pv, 4): rounding collapsed p = 1.3e-09 to 0.0.
                # Where the integration itself underflows, the value is floored
                # and flagged, because a p-value of exactly zero is not a
                # reportable quantity -- it is an upper bound.
                'dunnett_p': max(float(f'{pv:.3g}'), P_FLOOR),
                'dunnett_p_is_upper_bound': bool(pv < P_FLOOR),
                'significant': bool(pv < args.alpha),
            })

        results[label] = {
            'unit': unit,
            'anova_F': round(float(f), 3),
            'anova_p': float(f'{p:.3g}'),
            'levene_p': float(f'{lev_p:.3g}'),
            'control': CONTROL,
            'comparisons': comparisons,
        }

        print(f'\n=== {label} ({unit}) ===')
        print(f'one-way ANOVA: F = {f:.3f}, p = {p:.3g}   '
              f"(Levene p = {lev_p:.3g})")
        print(f'Dunnett vs {CONTROL}, alpha = {args.alpha}:')
        for c in comparisons:
            mark = '*' if c['significant'] else ' '
            print(f"  {mark} {c['ligand']:26s} diff {c['mean_difference_vs_control']:+.4f} "
                  f"[{c['ci_simultaneous_low']:+.4f}, "
                  f"{c['ci_simultaneous_high']:+.4f}]  "
                  f"g = {c['hedges_g']:+.2f}  "
                  f"p {'<' if c['dunnett_p_is_upper_bound'] else '='} "
                  f"{c['dunnett_p']:.3g}")

    out = dock[['Ligand', 'Manuscript_label', 'HeavyAtoms', 'Mean_affinity', 'SD',
                'Mean_LE', 'SD_LE', 'cLogP', 'LELP']].copy()
    out['Rank_by_affinity'] = out['Mean_affinity'].rank(method='min').astype(int)
    out['Rank_by_LE'] = out['Mean_LE'].rank(method='min', ascending=False).astype(int)
    out = out.sort_values('Rank_by_LE')
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    Path('results/data').mkdir(parents=True, exist_ok=True)
    out.to_csv('results/tables/ligand_efficiency.csv', index=False)
    Path('results/data/hypothesis_tests.json').write_text(json.dumps(results, indent=2))

    print('\n=== ranking: raw affinity vs ligand efficiency ===')
    print(out[['Ligand', 'HeavyAtoms', 'Mean_affinity', 'Rank_by_affinity',
               'Mean_LE', 'Rank_by_LE']].to_string(index=False))
    print('\n-> results/tables/ligand_efficiency.csv')
    print('-> results/data/hypothesis_tests.json')


if __name__ == '__main__':
    main()
