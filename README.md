# MPXV VP39 Repurposing: Revision Pipeline

Computational pipeline supporting the revision of the manuscript *Repurposing SARS-CoV-2
Methyltransferase Inhibitors Against Monkeypox Virus VP39: A Molecular Docking, Ligand
Efficiency and Protein Flexibility Study*, submitted to Biologia / Acta Scientifica.

Every number reported in the manuscript is produced by a script in this repository and
written to `results/`. Nothing in the manuscript text is typed from memory: the applier
(`12_revise_manuscript.py`) reads `results/` at build time.

Authors: Arres Mehezkeil Handoko, Arli Aditya Parikesit, Advent Roan Widiyono, Andrew
Medha, Emanuel Theodorus Sudiman, Felice Surya, Veronica Irina, Moh. Royhan Afnani, Arif
Nur Muhammad Ansori.

## Scope

The target is monkeypox virus VP39, a 2'-O-methyltransferase, solved with the pan-MTase
inhibitor sinefungin (PDB 8B07). Seven ligands are docked into the SAM pocket: sinefungin
(control), SAM, SAH, SS148, STM957, SGC0946 and SGC8158. The revision addresses thirteen
reviewer concerns.

Four results carry the revision:

1. **Docking protocol validated.** Redocking the co-crystallised sinefungin reproduces the
   crystal pose at 0.815 A heavy-atom RMSD over 27 atoms (threshold 2.0 A). Grid box,
   exhaustiveness, seed and software versions are recorded in
   `results/data/docking_protocol.json`.
2. **Ligand efficiency reverses the affinity ranking.** STM957, SGC8158 and SGC0946 lead on
   raw Vina score but are all significantly *less* efficient than sinefungin: they carry 38
   to 42 heavy atoms against 27. Ligand efficiency ANOVA F = 296.8, p = 6.25e-14; Dunnett
   comparisons in `results/data/hypothesis_tests.json`.
3. **The flexibility ranking is withdrawn.** CABS-flex 2.0 simulates the protein chain
   alone, so no ligand was present in any of the seven runs. An ANOVA recovered from the
   reported summary statistics gives F(6, 1974) = 0.579, p = 0.747, eta squared 0.0018: the
   seven RMSF profiles are statistically indistinguishable. RMSF is re-presented as
   intrinsic backbone flexibility of the ligand-free domain.
4. **nsp16 added alongside nsp14.** VP39 is a 2'-O-MTase, so nsp16 is the functionally
   matched SARS-CoV-2 enzyme, and it is also the structurally closer one: 166 equivalent
   residues (58.7 percent coverage, Ca RMSD 2.017 A) against 93 for nsp14 (32.9 percent,
   2.414 A). The nsp14 comparison is retained, reframed as a SAM-pocket architecture
   comparison rather than functional equivalence.

Docking scores are approximations, not binding free energies. No result here has been
validated in vitro; the ADMET panel is rule-based rather than predictive. See the
Limitations section of the manuscript.

## Layout

```
.

├── scripts/            steps 01 to 08, 11 (structures, docking, statistics)
├── data/raw/pdb/       downloaded coordinates (gitignored)
├── data/processed/     receptor/ligand PDBQT, docking runs (gitignored)
├── results/tables/     CSV/JSON feeding the manuscript tables
├── results/data/       provenance and full statistical output
├── environments/       conda environment and pinned requirements
└── tests/              regression tests over the reported numbers
```

## Pipeline

Steps run in numeric order; each writes to `results/` and reads only from earlier steps.

| Step | File | Purpose | Concern |
| --- | --- | --- | --- |
| 01 | `scripts/01_fetch_structures.py` | Download 8B07, 7TW7, 6YZ1, 8FRJ, 8FRK; record method, resolution, citation | R4 |
| 02 | `scripts/02_ligand_identity.py` | Resolve true compound identity (SGC0946, SGC8158, not DOT1L/PRMT7); PubChem CIDs, SMILES, descriptors | R5 |
| 03 | `scripts/03_prepare_and_validate.py` | Prepare receptor and ligands; redock sinefungin and report RMSD | R4 |
| 04 | `scripts/04_docking_runs.py` | Three seeded runs per ligand into the SAM pocket | R4 |
| 05 | `scripts/05_structural_alignment.py` | Full-domain CE superposition of VP39 against nsp14 and nsp16 | R6, R9 |
| 06 | `scripts/06_interactions.py` | Recompute contacts from pose coordinates; interaction summary for all seven ligands | R8 |
| 07 | `scripts/07_ligand_efficiency_stats.py` | ANOVA plus Dunnett against sinefungin, on affinity and on ligand efficiency | R7 |
| 08 | `scripts/08_admet.py` | RDKit descriptor and structural-alert panel (Lipinski, Veber, Ghose, Egan, PAINS/BRENK/NIH) | R3 |

| 09 | `scripts/11_rmsf_summary_test.py` | ANOVA recovered from the reported RMSF summary statistics | R1, R2 |


Revised prose lives in `utils/revision_content.py`, keyed by block index, so it can be read
and edited without touching document-manipulation code.

## Environment

AutoDock Vina and Open Babel come from conda-forge; neither installs reliably from PyPI on
macOS arm64.

```bash
conda env create -f environments/environment.yml
conda activate mpxv-revision
```

Reported results were produced with Python 3.11.16, AutoDock Vina 1.2.7, Open Babel 3.2.1,
RDKit 2026.03.1, Biopython 1.88 and SciPy 1.16.3, on macOS arm64 (Apple M3, 4P+4E; docking
uses `--cpu 4`).

TorchDrug is deliberately absent. It does not build on Apple silicon and ships model
architectures rather than pretrained ADMET predictors, so ADMET is a rule-based RDKit panel
with published thresholds instead of a learned model. See `ERRORS.md`.

## Running

```bash
conda run -n mpxv-revision python scripts/01_fetch_structures.py
conda run -n mpxv-revision python scripts/02_ligand_identity.py
# ... steps 03 to 08, 11 in scripts/; 09, 10, 12, 13 in utils/
```

Steps 01, 02 and 09 make network calls (RCSB, PubChem, CrossRef, NCBI). Step 09 uses the
CrossRef polite pool and needs `--mailto`. Docking (03, 04) is the only expensive stage:
about 21 production runs plus one validation redock.

## Tests

```bash
conda run -n mpxv-revision python -m pytest tests/ -q
```

The tests assert the manuscript's claims, not merely that the code runs: ligand identities,
the redocking RMSD, the direction of the efficiency reversal, the unrankability of the RMSF
profiles, nsp16 being the closer match, and manuscript integrity (PDB accessions present,
Limitations section before Conclusion, no em/en dashes in body prose, citations and
reference list agreeing both ways).

**Current status: 15 passed, 2 failed.** Both failures are path drift, not analysis errors.
The scripts were written with shared helpers under `scripts/` and steps 
under `scripts/`



## Key references

- Silhan J et al. (2023) *Nat Commun* 14 — VP39 with sinefungin, PDB 8B07
- Kottur J et al. (2023) *PLoS Pathog* 19:e1011546 — nsp14 inhibitors, PDB 7TW7/8FRJ/8FRK
- Krafcikova P et al. (2020) *Nat Commun* 11:3717 — nsp10-nsp16, PDB 6YZ1
- Eberhardt J et al. (2021) *J Chem Inf Model* 61:3891-3898 — AutoDock Vina 1.2.0
- Kenny PW (2019) *J Cheminform* 11:8 — the nature of ligand efficiency
- Kuriata A et al. (2018) *Nucleic Acids Res* 46:W338-W343 — CABS-flex 2.0
