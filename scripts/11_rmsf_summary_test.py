#!/usr/bin/env python3
"""
Module: 11_rmsf_summary_test
Purpose: Test whether the CABS-flex RMSF means in the submitted Table 3 can support a
         stability ranking (reviewer concern R2).

         The per-residue RMSF profiles were not retained, so the test is computed from
         the reported summary statistics. A one-way ANOVA F statistic is recoverable
         exactly from group means, standard deviations and group size, which is
         sufficient to answer the question the reviewer actually asked: are these seven
         values distinguishable from one another?

Author: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif Nur Muhammad Ansori
Date: 2026
Parameters:
    - --n : residues per RMSF profile (283 modelled residues in 8B07 chain A)
References:
    - Kuriata A et al. (2018) Nucleic Acids Res 46:W338-W343 (CABS-flex 2.0)
    - Cohen J (1988) Statistical Power Analysis for the Behavioral Sciences
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Values as reported in Table 3 of the submitted manuscript, relabelled to the
# corrected compound names.
RMSF = [
    ('STM957',                     1.059, 1.036),
    ('SGC0946',                    0.939, 0.978),
    ('SGC8158',                    1.048, 1.162),
    ('Sinefungin',                 1.021, 1.506),
    ('S-adenosyl-L-homocysteine',  0.983, 0.849),
    ('S-adenosyl-L-methionine',    1.017, 1.020),
    ('SS148',                      0.939, 0.840),
]


def anova_from_summary(means, sds, n):
    """One-way ANOVA F and p from group summary statistics with equal group sizes."""
    k = len(means)
    means, sds = np.asarray(means, float), np.asarray(sds, float)
    grand = means.mean()
    ss_between = n * ((means - grand) ** 2).sum()
    ss_within = (n - 1) * (sds ** 2).sum()
    df_b, df_w = k - 1, k * (n - 1)
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    f = ms_b / ms_w
    p = stats.f.sf(f, df_b, df_w)
    eta_sq = ss_between / (ss_between + ss_within)
    return {'F': round(float(f), 4), 'df_between': df_b, 'df_within': df_w,
            'p': float(f'{p:.4g}'), 'eta_squared': round(float(eta_sq), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=283)
    args = ap.parse_args()

    names = [r[0] for r in RMSF]
    means = [r[1] for r in RMSF]
    sds = [r[2] for r in RMSF]

    res = anova_from_summary(means, sds, args.n)
    spread = max(means) - min(means)
    res.update({
        'residues_per_profile': args.n,
        'mean_range_A': round(spread, 3),
        'smallest_reported_SD_A': min(sds),
        'largest_reported_SD_A': max(sds),
        'spread_as_fraction_of_smallest_SD': round(spread / min(sds), 3),
        'ranking_supportable': bool(res['p'] < 0.05 and spread > min(sds)),
    })

    print(f'one-way ANOVA on RMSF means (n = {args.n} residues per profile)')
    print(f"  F({res['df_between']}, {res['df_within']}) = {res['F']}, p = {res['p']}")
    print(f"  eta^2 = {res['eta_squared']}")
    print(f'  full range of means      : {spread:.3f} A')
    print(f'  smallest reported SD     : {min(sds):.3f} A')
    print(f'  range / smallest SD      : {spread / min(sds):.3f}')
    print(f"  ranking supportable      : {res['ranking_supportable']}")

    pd.DataFrame({'Complex': names, 'Mean_RMSF_A': means,
                  'SD_A': sds}).to_csv('results/tables/rmsf_summary.csv', index=False)
    Path('results/data/rmsf_test.json').write_text(json.dumps(res, indent=2))
    print('\n-> results/tables/rmsf_summary.csv')


if __name__ == '__main__':
    main()
