"""
loaders.py — MorpHIN data-file loaders
========================================
Each function reads exactly one file format and returns a plain Python
structure.  No analysis logic lives here.

Functions
---------
load_irregular_mapping  IRREGULAR_VERB_MAPPING       → {surface: lemma}
load_suffix_rules       newFormat* rule files         → sorted list of rules
load_suffix_analysis    SUFFIX_ANALYSIS               → {morpheme: [feat_dict]}
load_aux_verbs          VERB_AUXILIARY_LIST           → set of roots
load_flag_map           VERB_STEM/SUFFIX_FLAG_MAP     → {morpheme: flag}
"""

from __future__ import annotations

from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ── Low-level helper ──────────────────────────────────────────────────────────

def _open(path: str):
    return open(path, encoding="utf-8")


# ── Public loaders ────────────────────────────────────────────────────────────

def load_irregular_mapping(path: str) -> dict[str, str]:
    """
    IRREGULAR_VERB_MAPPING  →  {surface_form: lemma}
    e.g. 'गया' → 'जा',  'किया' → 'कर',  'लिया' → 'ले'
    """
    mapping: dict[str, str] = {}
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                mapping[parts[0].strip()] = parts[1].strip()
    return mapping


def load_suffix_rules(
    *paths: str,
    paradigms: frozenset[str] = frozenset({"verb", "copula"}),
) -> list[tuple[str, str, str]]:
    """
    Parse one or more suffix-replacement-rule files.
    Format per line:  paradigm, suffix, add_back,,,,priority

    Returns list of (paradigm, suffix, add_back) sorted by suffix length DESC
    so that longer (more specific) suffixes are tried first.
    """
    rules: list[tuple[str, str, str]] = []
    seen:  set[tuple[str, str, str]]  = set()
    INVALID_ADDBACKS = {"ै", "ो"}

    for path in paths:
        with _open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                paradigm = parts[0].strip()
                if paradigm not in paradigms:
                    continue
                suffix   = parts[1].strip()
                add_back = parts[2].strip()
                if add_back in INVALID_ADDBACKS:
                    continue  # skip entirely
                if not suffix or add_back.lower() == "nil":
                    continue
                key = (paradigm, suffix, add_back)
                if key not in seen:
                    seen.add(key)
                    rules.append(key)

    rules.sort(key=lambda r: len(r[1]), reverse=True)
    return rules


def load_suffix_analysis(path: str) -> dict[str, list[dict[str, str]]]:
    """
    SUFFIX_ANALYSIS  →  {morpheme_string: [feature_dict, …]}

    Morphemes ending with '$' are word-final markers.
    Morphemes without '$' are internal (can appear anywhere in suffix).
    The caller may inspect both sets separately.
    """
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            paradigm = parts[0]
            if paradigm not in ("verb", "copula"):
                continue
            morpheme = parts[1]
            feat: dict[str, str] = {}
            for fs in parts[2:]:
                fs = fs.strip()
                if not fs:
                    continue
                kv = fs.split(":")
                if len(kv) == 2:
                    feat[kv[0].strip()] = kv[1].strip()
            if feat:
                result[morpheme].append(feat)
    return dict(result)


def load_aux_verbs(path: str) -> set[str]:
    """VERB_AUXILIARY_LIST  →  set of auxiliary/vector verb roots."""
    aux: set[str] = set()
    with _open(path) as fh:
        for line in fh:
            w = line.strip()
            if w and not w.startswith("//"):
                aux.add(w)
    return aux


def load_flag_map(path: str, target_section: str = "#verb") -> dict[str, str]:
    """
    Generic loader for VERB_STEM_FLAG_MAP / VERB_SUFFIX_FLAG_MAP.
    Reads tab-delimited  morpheme → flag  within the target section.
    """
    flags: dict[str, str] = {}
    in_target = False
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("#"):
                in_target = (line == target_section)
                continue
            if in_target:
                parts = line.split("\t")
                if len(parts) == 2:
                    flags[parts[0].strip()] = parts[1].strip()
    return flags
