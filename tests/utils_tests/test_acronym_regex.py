"""The acronym regex in find_acronym_statute must stay linear.

`regex_check_acronym` used to be:

    ([A-Z]+[a-z]{0,1}\\.*\\s*,*)*((A|a)(c|C)(t|T))*\\s*

`[A-Z]+` sits inside a group the outer `*` repeats, so a run of capitals can be
split between the two quantifiers in exponentially many ways. On a long
capitalised token that does not satisfy the whole pattern the engine tries all
of them: 28 capitals took over a minute locally and doubled every two further
characters. CPython's `re` holds the GIL in C while matching, so the hang cannot
be interrupted from another thread - the worker is wedged.

These tests pin both halves of the fix: it must be fast, and it must accept
exactly what it accepted before.
"""

import itertools
import pathlib
import re
import time

import pytest

SAFE_PATTERN = r"(?:[A-Z][a-z]?\.*\s*,*)*(?:[Aa][cC][tT])*\s*"
VULNERABLE_PATTERN = r"([A-Z]+[a-z]{0,1}\.*\s*,*)*((A|a)(c|C)(t|T))*\s*"

# Statute strings the NER pipeline actually produces, and near-misses.
REAL_INPUTS = [
    "IPC", "CrPC", "Cr.P.C.", "I.P.C.", "IT Act", "IPC Act", "NDPS Act",
    "AIR", "SEBI Act", "A", "Ac", "Act", "act", "ACT", "IPC ", "IPC,",
    "I.P.C. Act ", "IPCAct", "", "Art", "Article 21", "Indian Penal Code",
    "CPC, 1908", "S. 302", "TheAct", "Foo bar", "iPc",
]


def _source_pattern():
    """The literal the shipped function uses.

    Read from the file rather than imported: importing the module pulls in
    spacy, transformers and a model download, none of which this test needs.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    source = (root / "opennyai" / "ner" / "InLegalNER" / "postprocessing_utils.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r'regex_check_acronym\s*=\s*r"([^"]+)"', source)
    assert match, "could not find regex_check_acronym in postprocessing_utils.py"
    return match.group(1)


def test_shipped_pattern_has_no_nested_quantifier():
    """The `[A-Z]+`-inside-`*` construct must not come back."""
    pattern = _source_pattern()
    assert "[A-Z]+" not in pattern, (
        f"nested quantifier is back in regex_check_acronym: {pattern!r}"
    )


def test_pathological_input_completes_quickly():
    """A long capitalised token must not wedge the process.

    This is the exact shape reached via
    InLegalNER.__call__ -> pro_statute_coref_resol -> create_statute_clusters
    -> find_acronym_statute when a STATUTE entity is a mis-segmented heading or
    an OCR artifact.
    """
    pattern = re.compile(_source_pattern())
    hostile = "A" * 5000 + "1"

    start = time.perf_counter()
    assert pattern.fullmatch(hostile) is None
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, (
        f"5000 capitals took {elapsed:.2f}s; the pattern is backtracking again"
    )


@pytest.mark.parametrize("text", REAL_INPUTS)
def test_behaviour_matches_the_original_pattern(text):
    """The fix is a speed fix, not a behaviour change."""
    shipped = re.fullmatch(_source_pattern(), text) is not None
    original = re.fullmatch(VULNERABLE_PATTERN, text) is not None
    assert shipped == original, (
        f"{text!r}: original accepted={original}, shipped accepts={shipped}"
    )


def test_behaviour_matches_exhaustively_on_short_strings():
    """Every string up to length 4 over a representative alphabet agrees.

    Length is capped because the ORIGINAL pattern is the slow one - going much
    further would make this test itself hang, which is the whole point.
    """
    alphabet = "ABacpt., \t1"
    shipped = re.compile(_source_pattern())
    original = re.compile(VULNERABLE_PATTERN)

    checked = 0
    for length in range(0, 5):
        for tup in itertools.product(alphabet, repeat=length):
            text = "".join(tup)
            checked += 1
            assert (shipped.fullmatch(text) is not None) == (
                original.fullmatch(text) is not None
            ), f"disagreement on {text!r}"

    assert checked > 10000
