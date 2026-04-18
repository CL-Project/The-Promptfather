"""
verb_analyzer.py — Hindi Verb Morphological Analyzer
======================================================
Backed by MorpHIN data files:
  IRREGULAR_VERB_MAPPING          suppletive forms (गया→जा, किया→कर …)
  newFormatSuffixReplacementRules suffix-stripping rules → stem recovery
  newFormatUniqueSuffixReplacementRules  additional unique verb rules
  SUFFIX_ANALYSIS                 morpheme → feature bundle
  VERB_SUFFIX_FLAG_MAP            suffix morpheme → aspect/tense flag
  VERB_STEM_FLAG_MAP              stem morpheme  → compound-verb flag
  VERB_AUXILIARY_LIST             vector/auxiliary verb roots

Public API
----------
  va = VerbAnalyzer(data_dir='path/to/morphin_files')
  analyses = va.analyze('खाया')          # → list[VerbAnalysis]
  print(va.summarize('खाएगा'))           # human-readable
  results = va.analyze_batch(word_list)  # → dict[str, list[VerbAnalysis]]
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerbFeatures:
    """Morphosyntactic feature bundle for a verb form."""
    aspect:       Optional[str] = None  # imperfective | perfective
    tense:        Optional[str] = None  # future | past | present
    mood:         Optional[str] = None  # imperative | subjunctive | conditional
    verbal_type:  Optional[str] = None  # finite | participle | infinitive | conjunctive
    person:       Optional[str] = None  # 1 | 2 | 3
    number:       Optional[str] = None  # S | P
    gender:       Optional[str] = None  # M | F | N
    case:         Optional[str] = None  # D (direct) | O (oblique)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def __str__(self) -> str:
        d = self.to_dict()
        return ", ".join(f"{k}:{v}" for k, v in d.items()) if d else "(no features)"

    def merge(self, other: "VerbFeatures") -> "VerbFeatures":
        """Return a new VerbFeatures with fields from other overriding self."""
        d = {**self.to_dict(), **other.to_dict()}
        return VerbFeatures(**d)


@dataclass
class VerbAnalysis:
    """A single morphological analysis of an inflected verb form."""
    surface:    str
    lemma:      str
    suffix:     str
    paradigm:   str
    features:   VerbFeatures
    irregular:  bool = False
    auxiliary:  Optional[str] = None   # vector/aux verb root detected in stem
    aux_flag:   Optional[str] = None   # MorpHIN flag for that component

    def __str__(self) -> str:
        tag = "[IRREG] " if self.irregular else ""
        aux = f"  compound({self.auxiliary})" if self.auxiliary else ""
        return (f"{tag}lemma={self.lemma!r:<12} "
                f"suf={self.suffix!r:<10} "
                f"{self.features}{aux}")


# ══════════════════════════════════════════════════════════════════════════════
# File loaders
# ══════════════════════════════════════════════════════════════════════════════

def _open(path: str):
    return open(path, encoding="utf-8")


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
    seen: set[tuple[str, str, str]] = set()

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


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

# ── Terminal morpheme → gender/number (from SUFFIX_ANALYSIS) ─────────────────
# Ordered longest-first so the greedy match works correctly.
_TERMINAL_GN: list[tuple[str, dict]] = [
    ("यीं", {"gender": "F", "number": "P"}),
    ("यी",  {"gender": "F", "number": "S"}),
    ("ये",  {"gender": "M", "number": "P"}),
    ("या",  {"gender": "M", "number": "S"}),
    ("ईं",  {"gender": "F", "number": "P"}),
    ("ई",   {"gender": "F", "number": "S"}),
    ("ुईं", {"gender": "F", "number": "P"}),
    ("ुई",  {"gender": "F", "number": "S"}),
    ("ुए",  {"gender": "M", "number": "P"}),
    ("ुआ",  {"gender": "M", "number": "S"}),
    ("ीं",  {"gender": "F", "number": "P"}),
    ("ी",   {"gender": "F", "number": "S"}),
    ("े",   {"gender": "M", "number": "P"}),
    ("ा",   {"gender": "M", "number": "S"}),
]

# ── Future gender/number (from suffix ending) ────────────────────────────────
_FUTURE_GN: list[tuple[str, dict]] = [
    ("गीं", {"gender": "F", "number": "P"}),
    ("गी",  {"gender": "F", "number": "S"}),
    ("गे",  {"gender": "M", "number": "P"}),
    ("गा",  {"gender": "M", "number": "S"}),
]

# ── Person markers ────────────────────────────────────────────────────────────
_PERSON_PREFIXES: list[tuple[str, str]] = [
    # First person
    ("ूंगा", "1"), ("ूँगा", "1"), ("ूंगी", "1"), ("ूँगी", "1"),
    ("ऊंगा", "1"), ("ऊँगा", "1"), ("ऊंगी", "1"), ("ऊँगी", "1"),
    ("ूं", "1"),   ("ूँ", "1"),   ("ऊं", "1"),   ("ऊँ", "1"),
    ("एंगीं", "1"), ("एँगीं", "1"),
    # Second person
    ("ोगे", "2"), ("ओगे", "2"), ("ोगी", "2"), ("ओगी", "2"),
    ("ोंगे", "2"), ("ोंगी", "2"),
    ("ो", "2"),   ("ओ", "2"),
    ("िए", "2"),  ("िये", "2"), ("इए", "2"), ("इये", "2"),
    ("जिये", "2"), ("जिए", "2"),
]

# ── Suffix membership sets for aspect / mood ─────────────────────────────────
_FUTURE_SUFFIXES = frozenset([
    "ूंगा",  "ूँगा",  "ूंगी",  "ूँगी",
    "ऊंगा",  "ऊँगा",  "ऊंगी",  "ऊँगी",
    "ेगा",   "ेगी",   "ेंगे",  "ेंगी",
    "एगा",   "एगी",   "एंगे",  "एंगी",  "एंगीं",  "एँगे",  "एँगीं",
    "येगा",  "येगी",  "येंगे", "येंगी",
    "ोगा",   "ोगी",   "ोगे",   "ओगे",   "ओगी",
    "ोंगे",  "ोंगी",
])

_IMPERFECTIVE_SUFFIXES = frozenset(["ता", "ती", "ते", "तीं"])

_PERFECTIVE_SUFFIXES = frozenset([
    "या",  "यी",  "ये",  "यीं",  "यें",
    "ई",   "ईं",
    "िया",
    "ुआ",  "ुए",  "ुई",  "ुईं",
    "ुये", "ुयी", "ुयीं",
    "ा",   "ी",   "े",   "ीं",
])

_IMPERATIVE_SUFFIXES = frozenset([
    "ो",  "ओ",   "िए",  "िये", "इए",  "इये",
    "जिये", "जिए",
])

_SUBJUNCTIVE_SUFFIXES = frozenset([
    "ए", "एँ", "ये", "यें",
    "ूँ", "ऊँ", "ूं", "ऊं",
    "ों", "ें",
])

_INFINITIVE_MAP = {
    "ना": {"verbal_type": "infinitive", "gender": "M", "number": "S"},
    "ने": {"verbal_type": "infinitive", "gender": "M", "number": "S", "case": "O"},
    "नी": {"verbal_type": "infinitive", "gender": "F", "number": "S"},
}

# ── Flag → feature mapping (from VERB_SUFFIX_FLAG_MAP) ───────────────────────
_FLAG_TO_FEATURE: dict[str, dict] = {
    "t":  {"aspect": "imperfective"},
    "g":  {"tense": "future"},
    "na": {"verbal_type": "infinitive"},
    "ne": {"verbal_type": "infinitive", "case": "O"},
    "ni": {"verbal_type": "infinitive", "gender": "F"},
    "y":  {"aspect": "perfective"},    # ya / ye / e / i perfective markers
}


def extract_features(suffix: str, suffix_flag_map: dict[str, str]) -> list[VerbFeatures]:
    """
    Return a list of VerbFeatures interpretations for the given suffix.
    Multiple interpretations arise when a suffix is morphologically ambiguous
    (e.g. bare 'े' can be perfective-masc-pl OR subjunctive-2nd).

    The list is ordered from most-specific to least-specific.
    """
    suffix = suffix.strip()

    # ── Infinitive (unambiguous) ──────────────────────────────────────────────
    if suffix in _INFINITIVE_MAP:
        return [VerbFeatures(**_INFINITIVE_MAP[suffix])]

    f = VerbFeatures()
    extra_interpretations: list[VerbFeatures] = []

    # ── Person ────────────────────────────────────────────────────────────────
    for morph, person in _PERSON_PREFIXES:
        if suffix == morph or suffix.endswith(morph) or suffix.startswith(morph):
            f.person = person
            break

    # ── Tense / Aspect / Mood ─────────────────────────────────────────────────
    # Priority: full-suffix set membership runs FIRST.
    # This prevents sub-morpheme flags (e.g. 'ए'→y→perfective) from
    # mis-tagging compound suffixes like 'एगा' (future) or 'इए' (imperative).
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
        # Fall back to morpheme-level flag scan for anything not in the sets
        for morph, flag in suffix_flag_map.items():
            if morph and morph in suffix:
                feat = _FLAG_TO_FEATURE.get(flag, {})
                for k, v in feat.items():
                    if getattr(f, k) is None:
                        setattr(f, k, v)

    # Imperative is always 2nd person
    if f.mood == "imperative":
        f.person = f.person or "2"

    # ── Gender / Number ───────────────────────────────────────────────────────
    if f.tense == "future":
        for morph, gn in _FUTURE_GN:
            if suffix.endswith(morph):
                f.gender = gn["gender"]
                f.number = gn["number"]
                break
    else:
        # Imperfective: gender/number encoded directly in suffix
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

    # ── Verbal type ──────────────────────────────────────────────────────────
    if f.verbal_type is None:
        if f.tense or f.mood:
            f.verbal_type = "finite"
        elif f.aspect in ("perfective", "imperfective"):
            f.verbal_type = "participle"

    # ── Ambiguous bare suffixes — generate extra interpretations ─────────────
    # 'े' alone: could be perfective-M-P  OR subjunctive-2-S/P
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


# ══════════════════════════════════════════════════════════════════════════════
# Irregular form feature inference
# ══════════════════════════════════════════════════════════════════════════════

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

    # Perfective forms
    f.aspect = "perfective"
    f.verbal_type = "participle"
    if word.endswith("ईं") or word.endswith("यीं"):
        f.gender, f.number = "F", "P"
    elif word.endswith("ई") or word.endswith("यी"):
        f.gender, f.number = "F", "S"
    elif word.endswith("ये") or word.endswith("ए") or word.endswith("ये"):
        f.gender, f.number = "M", "P"
    elif word.endswith("या") or word.endswith("या"):
        f.gender, f.number = "M", "S"
    elif word.endswith("ी"):
        f.gender, f.number = "F", "S"
    elif word.endswith("ीं"):
        f.gender, f.number = "F", "P"
    return f


# ══════════════════════════════════════════════════════════════════════════════
# Main Analyzer
# ══════════════════════════════════════════════════════════════════════════════

class VerbAnalyzer:
    """
    Hindi verb morphological analyzer.

    Parameters
    ----------
    data_dir : str
        Directory containing the MorpHIN data files.
    verb_lexicon : set[str] | None
        Optional set of known verb stems.  When supplied, analyses whose
        recovered lemma is NOT in the lexicon are flagged with
        `lexicon_verified = False` (but still returned — the caller decides
        whether to filter).
    """

    # Default filenames (relative to data_dir)
    _FILES = {
        "irregular":    "IRREGULAR_VERB_MAPPING",
        "rules_main":   "newFormatSuffixReplacementRules",
        "rules_unique": "newFormatUniqueSuffixReplacementRules",
        "suffix_analysis": "SUFFIX_ANALYSIS",
        "aux_list":     "VERB_AUXILIARY_LIST",
        "stem_flags":   "VERB_STEM_FLAG_MAP",
        "suffix_flags": "VERB_SUFFIX_FLAG_MAP",
    }

    def __init__(
        self,
        data_dir: str = ".",
        verb_lexicon: Optional[set[str]] = None,
    ):
        d = data_dir
        p = lambda name: os.path.join(d, self._FILES[name])

        self.irregulars:      dict[str, str]              = load_irregular_mapping(p("irregular"))
        self.suffix_rules:    list[tuple[str, str, str]]  = load_suffix_rules(
            p("rules_main"), p("rules_unique")
        )
        self.suffix_analysis: dict[str, list[dict]]       = load_suffix_analysis(p("suffix_analysis"))
        self.aux_verbs:       set[str]                    = load_aux_verbs(p("aux_list"))
        self.stem_flags:      dict[str, str]              = load_flag_map(p("stem_flags"),  "#verb")
        self.suffix_flags:    dict[str, str]              = load_flag_map(p("suffix_flags"), "#verb")
        self.verb_lexicon:    Optional[set[str]]          = verb_lexicon

        # Sort aux verbs longest-first for greedy matching
        self._aux_sorted = sorted(self.aux_verbs, key=len, reverse=True)
        # Sort stem-flag morphemes longest-first
        self._stem_flags_sorted = sorted(self.stem_flags, key=len, reverse=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, word: str) -> list[VerbAnalysis]:
        """
        Morphologically analyze a single Hindi word.

        Returns a list of VerbAnalysis objects.  Empty list means the word
        doesn't look like a verb according to this analyzer's resources.
        Results are ordered: irregulars first, then longest-suffix matches.
        """
        word = word.strip()
        results: list[VerbAnalysis] = []
        seen: set[tuple] = set()  # (lemma, suffix, paradigm) dedup key

        # ── Step 1: Irregular lookup (suppletive stems) ───────────────────────
        if word in self.irregulars:
            lemma    = self.irregulars[word]
            features = _infer_irregular_features(word)
            aux      = self._detect_aux(lemma)
            results.append(VerbAnalysis(
                surface=word, lemma=lemma,
                suffix="(irregular)", paradigm="verb",
                features=features, irregular=True,
                auxiliary=aux[0] if aux else None,
                aux_flag=aux[1]  if aux else None,
            ))
            # Don't return early — the same word might also match suffix rules
            # (rare but possible for highly regular irregulars)

        # ── Step 2: Suffix replacement rules ─────────────────────────────────
        for paradigm, suffix, add_back in self.suffix_rules:
            if not word.endswith(suffix):
                continue

            # Recover the stem / lemma candidate
            stem_base = word[: len(word) - len(suffix)]
            lemma     = stem_base + add_back   # add_back may be empty string

            if not lemma:
                continue

            # Lexicon check (optional, non-blocking)
            lex_ok = (self.verb_lexicon is None) or (lemma in self.verb_lexicon)

            # Build all feature interpretations for this suffix
            feat_list = extract_features(suffix, self.suffix_flags)

            for features in feat_list:
                key = (lemma, suffix, paradigm, str(features))
                if key in seen:
                    continue
                seen.add(key)

                aux = self._detect_aux(lemma)
                results.append(VerbAnalysis(
                    surface=word, lemma=lemma,
                    suffix=suffix, paradigm=paradigm,
                    features=features, irregular=False,
                    auxiliary=aux[0] if aux else None,
                    aux_flag=aux[1]  if aux else None,
                ))

        return results

    def analyze_batch(self, words: list[str]) -> dict[str, list[VerbAnalysis]]:
        """Analyze a list of words. Returns {word: [VerbAnalysis, …]}."""
        return {w: self.analyze(w) for w in words}

    def summarize(self, word: str) -> str:
        """Return a human-readable multi-line summary of all analyses."""
        analyses = self.analyze(word)
        if not analyses:
            return f"'{word}'  →  no verb analysis found"
        lines = [f"Analyses for  '{word}'  ({len(analyses)} result(s)):"]
        for i, a in enumerate(analyses, 1):
            lines.append(f"  {i:>2}.  {a}")
        return "\n".join(lines)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _detect_aux(self, stem: str) -> Optional[tuple[str, str]]:
        """
        Detect auxiliary/vector verb component in a recovered stem.
        Returns (morpheme, flag) or None.

        Checks VERB_STEM_FLAG_MAP morphemes first (longer matches win),
        then falls back to VERB_AUXILIARY_LIST.
        """
        # Stem-flag map (compound verb analysis)
        for morph in self._stem_flags_sorted:
            if morph in stem:
                return (morph, self.stem_flags[morph])
        # Aux verb list (direct root match)
        for aux in self._aux_sorted:
            if stem == aux or stem.endswith(aux):
                return (aux, "aux")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Integration helper — drop-in for analyzer.py
# ══════════════════════════════════════════════════════════════════════════════

def make_verb_analyzer(data_dir: str, verb_lexicon=None) -> VerbAnalyzer:
    """
    Factory that can be called from an existing analyzer.py:

        from verb_analyzer import make_verb_analyzer
        verb_analyzer = make_verb_analyzer(DATA_DIR)

        # In your main analyze() function:
        verb_results = verb_analyzer.analyze(token)
    """
    return VerbAnalyzer(data_dir=data_dir, verb_lexicon=verb_lexicon)


# ══════════════════════════════════════════════════════════════════════════════
# CLI demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    va = VerbAnalyzer(data_dir=data_dir)

    demo = [
        # ── Irregular suppletive forms ──
        "गया",   "गई",   "गए",
        "किया",  "किये", "की",
        "लिया",  "लिए",  "दिया",
        # ── Perfective (regular) ──
        "खाया",  "खाई",  "खाए",  "खाये",
        "पिया",                           # iया rule: stem पी
        # ── Imperfective (habitual participle) ──
        "खाता",  "खाती", "खाते",
        # ── Future ──
        "खाएगा", "खाएगी", "खाएंगे",
        "खाऊंगा", "खाऊंगी",
        # ── Infinitive ──
        "खाना",  "खाने", "खानी",
        # ── Imperative / Subjunctive ──
        "खाओ",   "खाइए",  "खाए",
        # ── Compound with aux component ──
        "खा",    "रह",    "सक",
    ]

    print("=" * 70)
    print("  Hindi Verb Analyzer — demo")
    print("=" * 70)
    for w in demo:
        print()
        print(va.summarize(w))

    print()
    print("=" * 70)
    print(f"Loaded {len(va.irregulars)} irregulars, "
          f"{len(va.suffix_rules)} suffix rules, "
          f"{len(va.aux_verbs)} auxiliaries.")
