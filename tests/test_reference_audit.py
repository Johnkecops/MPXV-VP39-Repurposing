#!/usr/bin/env python3
"""
Module: test_reference_audit
Purpose: Guard the citation-parsing and reference-matching bugs that let the
         audit report a clean bill of health for a manuscript it had not fully
         read. Synthetic strings only: no .docx is opened.
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.docx_io import leading_surname_year, section_of  # noqa: E402


def _load_audit():
    path = ROOT / 'utils' / '10_reference_audit.py'
    spec = importlib.util.spec_from_file_location('ref_audit', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load_audit()


# --- leading_surname_year -------------------------------------------------

def test_single_letter_surname_is_matched():
    """Regression: a `+` quantifier never matched a one-letter surname, so the
    ('C', '2020') DROP_REFERENCES key could not fire."""
    entry = ('C S, S. DK, Ragunathan V, Tiwari P, A. S, P BD (2020) '
             'Molecular docking study.')
    assert leading_surname_year(entry) == ('C', '2020')


def test_ordinary_surname_and_year():
    assert leading_surname_year('Silhan J et al. (2023) Nat Commun 14') == (
        'Silhan', '2023')


# --- section_of -----------------------------------------------------------

def test_numbered_and_punctuated_headings_are_recognised():
    """Regression: exact-match-only left the reference list empty, and every
    downstream 'no problems found' assertion then passed vacuously."""
    for text in ('REFERENCES', 'REFERENCES:', '4. REFERENCES', '4 REFERENCES',
                 '4.1) REFERENCES', 'REFERENCES.'):
        assert section_of(text) == 'REFERENCES', text
    assert section_of('MATERIAL AND METHODS') == 'MATERIAL AND METHODS'
    assert section_of('SOME OTHER PARAGRAPH') is None


# --- parse_intext ---------------------------------------------------------

def test_page_numbers_are_not_read_as_the_citation_year():
    """Regression: years[-1] turned '2020, pp. 1234-1240' into year 1240."""
    hits = audit.parse_intext('As shown (Smith et al., 2020, pp. 1234-1240).')
    assert ('Smith', '2020') in hits, hits
    assert not any(y == '1240' for _, y in hits), hits


def test_eg_prefixed_citation_is_not_dropped():
    """Regression: the prefix regex required whitespace, but real usage is
    '(e.g., Smith et al., 2020)', so the citation vanished silently."""
    assert ('Smith', '2020') in audit.parse_intext(
        'Several groups report this (e.g., Smith et al., 2020).')
    assert ('Jones', '2019') in audit.parse_intext(
        'See also (see Jones 2019).')


def test_multiple_citations_in_one_parenthetical():
    hits = audit.parse_intext('(Smith et al., 2020; Jones & Lee, 2019)')
    assert ('Smith', '2020') in hits and ('Jones', '2019') in hits, hits


# --- the both-ways reference check ---------------------------------------

def test_uncited_entry_by_a_two_entry_author_is_detected():
    """Regression: keying on surname alone meant citing Wang (2023) marked the
    uncited Wang (2022) entry as cited, so the check held in one direction."""
    listed = [
        {'surname': 'Wang', 'year': '2023', 'entry': 'Wang D-P (2023) a.'},
        {'surname': 'Wang', 'year': '2022', 'entry': 'Wang Y (2022) b.'},
    ]
    cited = {('Wang', '2023'): ['INTRODUCTION']}

    not_cited = [r for r in listed if (r['surname'], r['year']) not in cited]
    assert [r['year'] for r in not_cited] == ['2022'], not_cited

    # the old surname-only rule found nothing
    old = [r for r in listed if r['surname'] not in {s for s, _ in cited}]
    assert old == [], 'surname-only rule should have been the lenient one'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'ok  {name}')
