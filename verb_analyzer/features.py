"""
features.py — Feature extraction for Hindi verb suffixes
==========================================================
Pure functions; no file I/O, no class state.

Public API
----------
extract_features(suffix, suffix_flag_map)  → list[VerbFeatures]
    Returns every legitimate morphological reading for the given suffix.
    Multiple readings arise for ambiguous bare suffixes (े, ए, ें).
    Ordered most-specific → least-specific.

_infer_irregular_features(word)            → VerbFeatures
    Pattern-match a word from IRREGULAR_VERB_MAPPING to recover features.
    (Prefixed _ because callers normally go through VerbAnalyzer.analyze.)

Priority order (the critical design decision)
---------------------------------------------
1. Infinitive lookup           — ना/ने/नी are completely unambiguous, bail early.
2. Full-suffix set membership  — checked against the five frozensets BEFORE any
                                 morpheme scan.  Prevents sub-morpheme flags
                                 (e.g. ए → y → perfective) from mis-tagging
                                 compound suffixes like एगा (future) or इए (imperative).
3. Morpheme flag scan          — VERB_SUFFIX_FLAG_MAP fills gaps for suffixes
                                 not covered by the sets.
4. Terminal morpheme scan      — recovers gender/number from the suffix ending.
"""

from __future__ import annotations

from .models import VerbFeatures
from .feature_tables import (
    _TERMINAL_GN,
    _FUTURE_GN,
    _PERSON_PREFIXES,
    _FUTURE_SUFFIXES,
    _IMPERFECTIVE_SUFFIXES,
    _PERFECTIVE_SUFFIXES,
    _IMPERATIVE_SUFFIXES,
    _SUBJUNCTIVE_SUFFIXES,
    _INFINITIVE_MAP,
    _FLAG_TO_FEATURE,
)


def extract_features(suffix: str, suffix_flag_map: dict[str, str]) -> list[VerbFeatures]:
    """
    Return a list of VerbFeatures interpretations for the given suffix.
    Multiple interpretations arise when a suffix is morphologically ambiguous
    (e.g. bare 'े' can be perfective-masc-pl OR subjunctive-2nd).

    The list is ordered from most-specific to least-specific.
    """
    suffix = suffix.strip()

    # ── 1. Infinitive (unambiguous) ───────────────────────────────────────────
    if suffix in _INFINITIVE_MAP:
        return [VerbFeatures(**_INFINITIVE_MAP[suffix])]

    f = VerbFeatures()
    extra_interpretations: list[VerbFeatures] = []

    # ── 2. Person ─────────────────────────────────────────────────────────────
    for morph, person in _PERSON_PREFIXES:
        if suffix == morph or suffix.endswith(morph) or suffix.startswith(morph):
            f.person = person
            break

    # ── 3. Tense / Aspect / Mood ──────────────────────────────────────────────
    # Full-suffix set membership runs FIRST (see module docstring for why).
    if suffix in _FUTURE_SUFFIXES:
        f.tense = "future"
    elif suffix in _IMPERFECTIVE_SUFFIXES:
        f.aspect = "imperfective"
    elif suffix in _IMPERATIVE_SUFFIXES:
        f.mood = "imperative"
    elif suffix in _SUBJUNCTIVE_SUFFIXES:
        f.mood = "subjunctive"
    elif suffix in _PERFECTIVE_SUFFIXES:
        f.aspect = "perfective"
    else:
        # Fall back to morpheme-level flag scan
        for morph, flag in suffix_flag_map.items():
            if morph and morph in suffix:
                feat = _FLAG_TO_FEATURE.get(flag, {})
                for k, v in feat.items():
                    if getattr(f, k) is None:
                        setattr(f, k, v)

    # Imperative is always 2nd person
    if f.mood == "imperative":
        f.person = f.person or "2"

    # ── 4. Gender / Number ────────────────────────────────────────────────────
    if f.tense == "future":
        for morph, gn in _FUTURE_GN:
            if suffix.endswith(morph):
                f.gender = gn["gender"]
                f.number = gn["number"]
                break
    else:
        if f.aspect == "imperfective":
            _IMP_GN = {
                "ता": ("M", "S"), "ती": ("F", "S"),
                "ते": ("M", "P"), "तीं": ("F", "P"),
            }
            if suffix in _IMP_GN:
                f.gender, f.number = _IMP_GN[suffix]
        else:
            # Terminal morpheme scan (longest match first)
            for morph, gn in _TERMINAL_GN:
                if suffix.endswith(morph):
                    f.gender = f.gender or gn.get("gender")
                    f.number = f.number or gn.get("number")
                    break

    # ── 5. Verbal type ────────────────────────────────────────────────────────
    if f.verbal_type is None:
        if f.tense or f.mood:
            f.verbal_type = "finite"
        elif f.aspect in ("perfective", "imperfective"):
            f.verbal_type = "participle"

    # ── 6. Ambiguous bare suffixes — generate extra interpretations ───────────
    # 'े' alone: could be perfective-M-P  OR  subjunctive-2-S/P
    if suffix == "े" and f.verbal_type == "participle":
        alt = VerbFeatures(mood="subjunctive", person="2", verbal_type="finite")
        extra_interpretations.append(alt)

    # 'ें' alone: could be subjunctive-S or subjunctive-P
    if suffix == "ें":
        alt1 = VerbFeatures(mood="subjunctive", number="S", verbal_type="finite")
        alt2 = VerbFeatures(mood="subjunctive", number="P", verbal_type="finite")
        extra_interpretations.extend([alt1, alt2])

    # 'ए' alone: perfective-M-P  OR  subjunctive-3-S
    if suffix == "ए":
        alt = VerbFeatures(mood="subjunctive", person="3", number="S", verbal_type="finite")
        extra_interpretations.append(alt)

    return [f] + extra_interpretations


def _infer_irregular_features(word: str) -> VerbFeatures:
    """
    Pattern-match an irregular surface form to infer its feature bundle.
    All forms in IRREGULAR_VERB_MAPPING are perfective, future, or imperative.
    """
    f = VerbFeatures()

    # Future forms: लूंगा, लूंगी, लेंगे, लेंगीं, दूंगी, देंगे …
    if any(word.endswith(s) for s in ("गा", "गी", "गे", "गीं")):
        f.tense = "future"
        f.verbal_type = "finite"
        if word.endswith("गीं"):
            f.gender, f.number = "F", "P"
        elif word.endswith("गी"):
            f.gender, f.number = "F", "S"
        elif word.endswith("गे"):
            f.gender, f.number = "M", "P"
        elif word.endswith("गा"):
            f.gender, f.number = "M", "S"
        return f

    # Imperative forms: लीजिए, लीजिये, दीजिए …
    if any(word.endswith(s) for s in ("जिए", "जिये", "लीजिए", "लीजिये")):
        f.mood, f.person, f.verbal_type = "imperative", "2", "finite"
        return f

    # Perfective forms (default)
    f.aspect = "perfective"
    f.verbal_type = "participle"
    if word.endswith("ईं") or word.endswith("यीं"):
        f.gender, f.number = "F", "P"
    elif word.endswith("ई") or word.endswith("यी"):
        f.gender, f.number = "F", "S"
    elif word.endswith("ये") or word.endswith("ए"):
        f.gender, f.number = "M", "P"
    elif word.endswith("या"):
        f.gender, f.number = "M", "S"
    elif word.endswith("ी"):
        f.gender, f.number = "F", "S"
    elif word.endswith("ीं"):
        f.gender, f.number = "F", "P"
    return f
