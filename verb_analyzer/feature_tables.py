"""
feature_tables.py — Static linguistic tables
==============================================
All read-only lookup tables consumed by features.py.
Editing linguistic coverage (adding new suffixes, fixing gender assignment, …)
should only require touching this file.

Exports
-------
_TERMINAL_GN          suffix ending → {gender, number}  (imperfective / perfective)
_FUTURE_GN            suffix ending → {gender, number}  (future forms)
_PERSON_PREFIXES      suffix substring → person string
_FUTURE_SUFFIXES      frozenset of unambiguous future suffixes
_IMPERFECTIVE_SUFFIXES
_PERFECTIVE_SUFFIXES
_IMPERATIVE_SUFFIXES
_SUBJUNCTIVE_SUFFIXES
_INFINITIVE_MAP       suffix → full VerbFeatures kwargs  (unambiguous)
_FLAG_TO_FEATURE      MorpHIN flag → partial feature dict
"""
import unicodedata

_nfc = lambda s: unicodedata.normalize('NFC', s)

# ── Terminal morpheme → gender/number ────────────────────────────────────────
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

_FUTURE_SUFFIXES: frozenset[str] = frozenset(_nfc(s) for s in [
    "ूंगा",  "ूँगा",  "ूंगी",  "ूँगी",
    "ऊंगा",  "ऊँगा",  "ऊंगी",  "ऊँगी",
    "ेगा",   "ेगी",   "ेंगे",  "ेंगी",
    "एगा",   "एगी",   "एंगे",  "एंगी",  "एंगीं",  "एँगे",  "एँगीं",
    "येगा",  "येगी",  "येंगे", "येंगी",
    "ोगा",   "ोगी",   "ोगे",   "ओगे",   "ओगी",
    "ोंगे",  "ोंगी",
])

_IMPERFECTIVE_SUFFIXES: frozenset[str] = frozenset(_nfc(s) for s in ["ता", "ती", "ते", "तीं"])

_PERFECTIVE_SUFFIXES: frozenset[str] = frozenset(_nfc(s) for s in [
    "या",  "यी",  "ये",  "यीं",  "यें",
    "ई",   "ईं",
    "िया",
    "ुआ",  "ुए",  "ुई",  "ुईं",
    "ुये", "ुयी", "ुयीं",
    "ा",   "ी",   "े",   "ीं",
])

_IMPERATIVE_SUFFIXES: frozenset[str] = frozenset(_nfc(s) for s in [
    "ो",  "ओ",   "िए",  "िये", "इए",  "इये",
    "जिये", "जिए",
])

_SUBJUNCTIVE_SUFFIXES: frozenset[str] = frozenset(_nfc(s) for s in [
    "ए", "एँ", "ये", "यें",
    "ूँ", "ऊँ", "ूं", "ऊं",
    "ों", "ें",
])

# ── Infinitive (unambiguous forms → full feature bundle) ─────────────────────
_INFINITIVE_MAP: dict[str, dict] = {
    "ना": {"verbal_type": "infinitive", "gender": "M", "number": "S"},
    "ने": {"verbal_type": "infinitive", "gender": "M", "number": "S", "case": "O"},
    "नी": {"verbal_type": "infinitive", "gender": "F", "number": "S"},
}

# ── MorpHIN flag → partial feature dict ──────────────────────────────────────
_FLAG_TO_FEATURE: dict[str, dict] = {
    "t":  {"aspect": "imperfective"},
    "g":  {"tense": "future"},
    "na": {"verbal_type": "infinitive"},
    "ne": {"verbal_type": "infinitive", "case": "O"},
    "ni": {"verbal_type": "infinitive", "gender": "F"},
    "y":  {"aspect": "perfective"},    # ya / ye / e / i perfective markers
}

_AMBIGUOUS_SUFFIXES: dict[str, list[dict]] = {
    "ए": [
        {"aspect": "perfective", "verbal_type": "participle", "gender": "M", "number": "P"},
        {"mood": "subjunctive", "verbal_type": "finite"},
        {"mood": "subjunctive", "verbal_type": "finite", "person": "3", "number": "S"},
    ],
    "े": [
        {"aspect": "perfective", "verbal_type": "participle", "gender": "M", "number": "P"},
        {"mood": "subjunctive", "verbal_type": "finite", "person": "2"},
    ],
    "ें": [
        {"mood": "subjunctive", "verbal_type": "finite", "number": "S"},
        {"mood": "subjunctive", "verbal_type": "finite", "number": "P"},
    ],
}
