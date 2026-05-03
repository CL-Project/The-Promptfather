"""
disambiguator.py — Context-based morphological disambiguator for Hindi
=======================================================================
Takes a raw Hindi sentence and returns a disambiguated token list by
filtering each token's candidate analyses using postposition, quantifier,
and verb-agreement signals extracted from the surrounding context.

Public API
----------
    from disambiguator import Disambiguator

    d = Disambiguator(
            noun_lexicon_path='Nouns/data/noun_lexicon_expanded.tsv',
            verb_data_dir='path/to/morphin_files',
        )
    result = d.disambiguate('लड़की ने खाना खाया')
    for tok in result:
        print(tok)

    # Slot in a gender model for out-of-lexicon words:
    d = Disambiguator(..., gender_model=my_model)

Design notes
------------
* Pure filter — never generates new analyses, only removes invalid candidates
  from what the existing analyzers already produced.
* Stateless per sentence — no cross-sentence memory.
* Noun signals are applied in strict priority order:
    genitive-gender > postposition-case > quantifier-number > verb-agreement
* Verb signals: negation mood constraint (नहीं → ¬IMP, मत → IMP-only).
* Clitics (भी, ही, तो) are treated as transparent for postposition and
  quantifier lookback — the skip logic finds the nearest non-clitic neighbour.
  Character-fused clitics are not handled (requires upstream segmentation).
* Compound postpositions are detected in a longest-match-first pre-pass;
  all tokens inside a matched compound are excluded from the single-token
  oblique rule to prevent double-firing.
* Ergative construction (ने → object agreement) is NOT yet implemented.
  Any unresolvable token is returned with ambiguous=True.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Noun analyzer imports
# ---------------------------------------------------------------------------
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_NOUNS_DIR = os.path.join(_HERE, "Nouns")
if _NOUNS_DIR not in sys.path:
    sys.path.insert(0, _NOUNS_DIR)

from Nouns.analyzer import load_lexicon, analyze          # Nouns/analyzer.py
from Nouns.noun_paradigm_templates import PARADIGM_TABLES  # Nouns/noun_paradigm_templates.py

# ---------------------------------------------------------------------------
# Verb analyzer imports
# ---------------------------------------------------------------------------
from verb_analyzer import VerbAnalyzer  # verb_analyzer/__init__.py


# ===========================================================================
# Clitics
# ===========================================================================
# Space-separated clitics that attach to noun phrases but are transparent for
# postposition/quantifier detection.  When scanning for a governed noun,
# positions occupied by these tokens are skipped.
# Note: character-fused clitics (e.g. "लड़कीभी") are not handled here.
_CLITICS: frozenset[str] = frozenset({
    "भी",   # also / even
    "ही",   # only / just (emphatic)
    "तो",   # then / as for (topic marker)
})


# ===========================================================================
# Single-token postpositions
# ===========================================================================

# Full inventory — used to detect when a token is *any* kind of postposition
# (needed for the "prev token is also a postposition" guard).
_POSTPOSITIONS: frozenset[str] = frozenset({
    "को",      # dative / accusative
    "में",     # locative
    "से",      # ablative / instrumental
    "ने",      # ergative
    "पर",      # locative / on
    "तक",      # up to / until
    "के",      # genitive masculine
    "की",      # genitive feminine
    "का",      # genitive masculine singular
    "द्वारा",  # by (formal instrumental)
    "पास",     # near / with
    "साथ",     # with / together
    "बिना",    # without
    "जैसा",    # like / as (M SG)
    "जैसी",    # like / as (F)
    "जैसे",    # like / as (M OBL/PL)
    "बाद",     # after
    "पहले",    # before
    "आगे",     # ahead / in front of
    "पीछे",    # behind
    "ऊपर",     # above
    "नीचे",    # below
    "अंदर",    # inside
    "बाहर",    # outside
    "बीच",     # between / among
    "सामने",   # in front of / facing
    "खिलाफ",   # against
    "अनुसार",  # according to
    "अलावा",   # besides / except
    "दौरान",   # during
    "बावजूद",  # despite
    "विरुद्ध",  # against (formal)
    "तरफ",     # towards
    "ओर",      # towards / direction
    "तरह",     # like / in the manner of
    "जगह",     # instead of / in place of
    "बजाय",    # instead of
    "खातिर",   # for the sake of
    "लिए",     # for / in order to
    "बारे",    # about (always in के बारे में)
    "रूप",     # form (as in के रूप में)
    "आधार",    # basis (as in के आधार पर)
    "नाम",     # name (as in के नाम पर)
    "वजह",     # reason / cause (as in की वजह से)
    "तुलना",   # comparison (as in की तुलना में)
    "मुकाबले", # compared to
    "बदले",    # in exchange (as in के बदले में)
    "बराबर",   # equal to
})

# Subset that specifically signals OBLIQUE case on the *governed* noun.
_OBL_POSTPOSITIONS: frozenset[str] = frozenset({
    "को", "में", "से", "ने", "पर", "तक",
    "के", "की", "का", "द्वारा",
    "पास", "साथ", "बिना",
    "बाद", "पहले", "आगे", "पीछे",
    "ऊपर", "नीचे", "अंदर", "बाहर",
    "बीच", "सामने", "खिलाफ", "अनुसार",
    "अलावा", "दौरान", "बावजूद", "विरुद्ध",
    "तरफ", "ओर", "तरह", "जगह", "बजाय", "खातिर",
})


# ===========================================================================
# Compound postpositions
# ===========================================================================
# Keys are tuples of whitespace-separated tokens exactly as they appear.
# Values are the case signal emitted on the governed noun (always "OBL").
#
# Detection uses a longest-match-first pre-pass in _build_signal_map.
# All tokens inside a matched compound are flagged so the single-token
# oblique rule does not double-fire on them.
#
# The governed noun is the nearest non-clitic token to the LEFT of the
# compound's first token (computed via _prev_content_pos).

_COMPOUND_POSTPOSITIONS: dict[tuple[str, ...], str] = {

    # ── के + X  (2-token, genitive masculine base) ───────────────────────────
    ("के", "लिए"):     "OBL",   # for / in order to
    ("के", "साथ"):     "OBL",   # with / together with
    ("के", "बाद"):     "OBL",   # after
    ("के", "ऊपर"):     "OBL",   # above / over
    ("के", "नीचे"):    "OBL",   # below / under
    ("के", "अंदर"):    "OBL",   # inside
    ("के", "बाहर"):    "OBL",   # outside
    ("के", "आगे"):     "OBL",   # ahead of / in front of
    ("के", "पीछे"):    "OBL",   # behind
    ("के", "पास"):     "OBL",   # near / with (possession)
    ("के", "बीच"):     "OBL",   # between / among
    ("के", "खिलाफ"):   "OBL",   # against
    ("के", "बिना"):    "OBL",   # without
    ("के", "द्वारा"):  "OBL",   # by / through (formal)
    ("के", "अनुसार"):  "OBL",   # according to
    ("के", "अलावा"):   "OBL",   # besides / except for
    ("के", "सामने"):   "OBL",   # in front of / facing
    ("के", "दौरान"):   "OBL",   # during
    ("के", "बावजूद"):  "OBL",   # despite
    ("के", "विरुद्ध"): "OBL",   # against (formal/legal)
    ("के", "बजाय"):    "OBL",   # instead of
    ("के", "बराबर"):   "OBL",   # equal to / on a par with

    # ── की + X  (2-token, genitive feminine base) ────────────────────────────
    ("की", "तरफ"):     "OBL",   # towards
    ("की", "ओर"):      "OBL",   # towards / in the direction of
    ("की", "तरह"):     "OBL",   # like / in the manner of
    ("की", "जगह"):     "OBL",   # instead of / in place of
    ("की", "बजाय"):    "OBL",   # instead of (variant)
    ("की", "खातिर"):   "OBL",   # for the sake of

    # ── से + X  (2-token, ablative base) ─────────────────────────────────────
    ("से", "पहले"):    "OBL",   # before
    ("से", "दूर"):     "OBL",   # far from
    ("से", "ज़्यादा"): "OBL",   # more than
    ("से", "ज्यादा"):  "OBL",   # more than (unaccented variant)
    ("से", "कम"):      "OBL",   # less than
    ("से", "लेकर"):    "OBL",   # from … (up to)
    ("से", "अलग"):     "OBL",   # different from / apart from
    ("से", "परे"):     "OBL",   # beyond
    ("से", "बाहर"):    "OBL",   # outside of

    # ── को + X  (2-token, dative base) ───────────────────────────────────────
    ("को", "छोड़कर"):  "OBL",   # except for / leaving aside
    ("को", "लेकर"):    "OBL",   # concerning / regarding

    # ── 3-token compounds ─────────────────────────────────────────────────────
    # Listed before any 2-token prefix so the pre-pass (which sorts longest-
    # first) always prefers the full compound over a partial match.
    ("के", "बारे", "में"):    "OBL",   # about / regarding
    ("के", "बदले", "में"):    "OBL",   # in exchange for
    ("के", "आधार", "पर"):    "OBL",   # on the basis of
    ("के", "रूप", "में"):     "OBL",   # in the form of / as (role)
    ("के", "नाम", "पर"):     "OBL",   # in the name of
    ("के", "साथ", "साथ"):    "OBL",   # alongside / at the same time as
    ("के", "मुकाबले", "में"): "OBL",   # compared to / in comparison with
    ("के", "विरोध", "में"):   "OBL",   # in opposition to
    ("की", "वजह", "से"):     "OBL",   # because of / due to
    ("की", "तुलना", "में"):   "OBL",   # in comparison with
}

# Pre-sorted longest-first so the pre-pass always prefers maximal matches.
_COMPOUND_POSTPOSITIONS_SORTED: list[tuple[str, ...]] = sorted(
    _COMPOUND_POSTPOSITIONS, key=len, reverse=True
)


# ===========================================================================
# Quantifiers
# ===========================================================================

_SINGULAR_QUANTIFIERS: frozenset[str] = frozenset({
    "एक",        # one
    "इस",        # this (proximal singular demonstrative)
    "उस",        # that (distal singular demonstrative)
    "किस",       # which (interrogative singular)
    "हर",        # every (distributive singular)
    "प्रत्येक",  # each
})

_PLURAL_QUANTIFIERS: frozenset[str] = frozenset({
    "सब",    # all
    "सभी",   # all (more formal)
    "कई",    # several / many
    "कुछ",   # some / a few
    "बहुत",  # many (when followed by plural noun)
    "अनेक",  # numerous
    "ये",    # these (proximal plural demonstrative)
    "वे",    # those (distal plural demonstrative)
    "इन",    # oblique of ये
    "उन",    # oblique of वे
})


# ===========================================================================
# Genitive agreement signals
# ===========================================================================
# का/की/के agree in gender (and for का, number) with the noun they MODIFY
# (the head noun to their right).  This emits a forward-looking gender signal.
#
#   का → M, SG    (masculine singular head)
#   की → F, None  (feminine head; number not determined by की alone)
#   के → M, None  (masculine head, OBL or PL — number ambiguous from के alone)

_GENITIVE_AGREEMENT: dict[str, tuple[str, Optional[str]]] = {
    "का": ("M", "SG"),
    "की": ("F", None),
    "के": ("M", None),
}


# ===========================================================================
# Negation
# ===========================================================================
# Negation words constrain the mood of verbs in the same clause.
#   नहीं / न / ना  →  declarative negation  →  verb is NOT imperative
#   मत              →  prohibitive negation   →  verb MUST BE imperative
#
# Scope note: treated as a sentence-level signal — will misfire in complex
# multi-clause sentences. Acceptable for typical simple/compound sentences.

_NEGATION: dict[str, str] = {
    "नहीं": "nahi",   # standard declarative negation
    "न":    "nahi",   # literary / formal (same constraint)
    "ना":   "nahi",   # colloquial variant
    "मत":   "mat",    # prohibitive — verb must be imperative
}


# ===========================================================================
# Misc module-level helpers
# ===========================================================================

# Verb number codes from VerbAnalyzer → internal 'SG'/'PL'
_VERB_NUM_MAP: dict[str, str] = {"S": "SG", "P": "PL"}


def _prev_content_pos(tokens: list[str], i: int) -> int:
    """
    Return the index of the nearest non-clitic token strictly before
    position ``i``, skipping over _CLITICS entries.  Returns -1 if none.

    Example: tokens = ["लड़की", "भी", "ने"]
        _prev_content_pos(tokens, 2)  →  0   (skips "भी" at index 1)
    """
    j = i - 1
    while j >= 0 and tokens[j] in _CLITICS:
        j -= 1
    return j


def _next_content_pos(tokens: list[str], i: int) -> int:
    """
    Return the index of the nearest non-clitic token strictly after
    position ``i``, skipping over _CLITICS entries.
    Returns len(tokens) if none.

    Example: tokens = ["का", "भी", "घर"]
        _next_content_pos(tokens, 0)  →  2   (skips "भी" at index 1)
    """
    j = i + 1
    while j < len(tokens) and tokens[j] in _CLITICS:
        j += 1
    return j


def _reading_key(c: NounCandidate) -> tuple:
    """Canonical key for a NounCandidate — used to count distinct readings."""
    return (c.lemma, c.gender, c.number, c.case)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class NounCandidate:
    """One morphological reading of a noun token."""
    lemma:      str
    gender:     str          # 'M' | 'F'
    number:     str          # 'SG' | 'PL'
    case:       str          # 'DIR' | 'OBL'
    paradigm:   str          # e.g. 'M1', 'F3'
    confidence: str          # 'certain' | 'heuristic' | 'predicted'
    source:     str = "lexicon"   # 'lexicon' | 'gender_model'


@dataclass
class VerbCandidate:
    """One morphological reading of a verb token (thin wrapper over VerbAnalysis)."""
    lemma:       str
    suffix:      str
    paradigm:    str
    gender:      Optional[str]   # 'M' | 'F' | None
    number:      Optional[str]   # 'S' | 'P' | None
    person:      Optional[str]   # '1' | '2' | '3' | None
    aspect:      Optional[str]
    tense:       Optional[str]
    mood:        Optional[str]
    verbal_type: Optional[str]
    confidence:  str
    irregular:   bool = False


@dataclass
class TokenResult:
    """Disambiguated result for a single token."""
    token:            str
    position:         int
    noun_candidates:  list[NounCandidate] = field(default_factory=list)
    verb_candidates:  list[VerbCandidate] = field(default_factory=list)
    ambiguous:        bool = False
    signals_applied:  list[str] = field(default_factory=list)

    @property
    def has_analysis(self) -> bool:
        return bool(self.noun_candidates or self.verb_candidates)

    @property
    def resolved_noun(self) -> Optional[NounCandidate]:
        if len(self.noun_candidates) == 1:
            return self.noun_candidates[0]
        return None

    @property
    def resolved_verb(self) -> Optional[VerbCandidate]:
        if len(self.verb_candidates) == 1:
            return self.verb_candidates[0]
        return None

    def __str__(self) -> str:
        parts = [f"[{self.position}] '{self.token}'"]
        if self.signals_applied:
            parts.append(f"  signals: {', '.join(self.signals_applied)}")
        if self.noun_candidates:
            for c in self.noun_candidates:
                parts.append(
                    f"  NOUN  lemma={c.lemma!r:<12} "
                    f"{c.paradigm} {c.gender} {c.number} {c.case} conf={c.confidence}"
                )
        if self.verb_candidates:
            for c in self.verb_candidates:
                feats = " ".join(
                    f"{k}={v}" for k, v in {
                        "asp": c.aspect, "tns": c.tense, "mood": c.mood,
                        "gen": c.gender, "num": c.number, "per": c.person,
                    }.items() if v
                )
                parts.append(
                    f"  VERB  lemma={c.lemma!r:<12} suf={c.suffix!r:<8} "
                    f"{feats} conf={c.confidence}"
                )
        if not self.has_analysis:
            parts.append("  (no analysis)")
        if self.ambiguous:
            parts.append("  ⚠ ambiguous — multiple readings remain")
        return "\n".join(parts)


# ===========================================================================
# Gender model protocol (slot for external model)
# ===========================================================================

@runtime_checkable
class GenderModel(Protocol):
    """
    Minimal interface for a pluggable gender prediction model.

    The model receives a single Hindi word string and returns:
        'M'  — predicted masculine
        'F'  — predicted feminine
        None — model abstains (no prediction)
    """
    def predict(self, word: str) -> Optional[str]: ...


# ===========================================================================
# Signal map (internal)
# ===========================================================================

@dataclass
class _SignalMap:
    """
    All context signals collected for a sentence, indexed by token position.

    oblique[i]      = True          → token i must be oblique case
    number[i]       = 'SG' | 'PL'  → token i's number is constrained
    gender[i]       = 'M'  | 'F'   → token i's gender is constrained
                                      (sourced from genitive agreement)
    verb_agreement  = [(gender, number), ...]  distilled from verb tokens
    negation_mood   = 'nahi' | 'mat' | None   sentence-level negation type
    """
    oblique:        dict[int, bool]        = field(default_factory=dict)
    number:         dict[int, str]         = field(default_factory=dict)
    gender:         dict[int, str]         = field(default_factory=dict)
    verb_agreement: list[tuple[str, str]]  = field(default_factory=list)
    negation_mood:  Optional[str]          = None


# ===========================================================================
# Main disambiguator
# ===========================================================================

class Disambiguator:
    """
    Context-based morphological disambiguator.

    Parameters
    ----------
    noun_lexicon_path : str
        Path to noun_lexicon_expanded.tsv (or noun_lexicon_final.tsv).
    verb_data_dir : str
        Directory containing MorpHIN verb data files.
    gender_model : GenderModel | None
        Optional pluggable gender predictor for out-of-lexicon consonant-final
        nouns.  Must implement GenderModel.predict(word) → 'M' | 'F' | None.
    """

    def __init__(
        self,
        noun_lexicon_path: str,
        verb_data_dir: str,
        gender_model: Optional[GenderModel] = None,
    ):
        self._lex_lookup, self._lex_display, self._lex_confidence = load_lexicon(
            noun_lexicon_path
        )
        self._paradigm_tables = PARADIGM_TABLES
        self._verb_analyzer = VerbAnalyzer(data_dir=verb_data_dir)
        self._gender_model = gender_model

    # =========================================================================
    # Step 1 — Tokenize
    # =========================================================================

    def _tokenize(self, sentence: str) -> list[str]:
        """
        Split on whitespace and NFC-normalise each token so that comparisons
        against the postposition / quantifier sets are reliable.
        """
        return [
            unicodedata.normalize('NFC', tok)
            for tok in sentence.split()
            if tok.strip()
        ]

    # =========================================================================
    # Step 2 — Run analyzers on every token
    # =========================================================================

    def _analyze_token_nouns(self, token: str) -> list[NounCandidate]:
        raw = analyze(
            token,
            self._lex_lookup,
            self._lex_display,
            self._lex_confidence,
            self._paradigm_tables,
        )
        return [
            NounCandidate(
                lemma=r["lemma"],
                gender=r["gender"],
                number=r["number"],
                case=r["case"],
                paradigm=r["paradigm"],
                confidence=r["confidence"],
            )
            for r in raw
        ]

    def _analyze_token_verbs(self, token: str) -> list[VerbCandidate]:
        raw = self._verb_analyzer.analyze(token)
        candidates = []
        for a in raw:
            f = a.features
            candidates.append(VerbCandidate(
                lemma=a.lemma,
                suffix=a.suffix,
                paradigm=a.paradigm,
                gender=f.gender,
                number=f.number,
                person=f.person,
                aspect=f.aspect,
                tense=f.tense,
                mood=f.mood,
                verbal_type=f.verbal_type,
                confidence=a.confidence,
                irregular=a.irregular,
            ))
        return candidates

    # =========================================================================
    # Step 3 — Build context signal map
    # =========================================================================

    def _build_signal_map(
        self,
        tokens: list[str],
        verb_candidates_by_pos: dict[int, list[VerbCandidate]],
    ) -> _SignalMap:
        """
        Two-pass signal collection.

        Pass 1 — Compound postpositions (longest-match-first pre-pass).
            Scans for every entry in _COMPOUND_POSTPOSITIONS_SORTED.
            On a match, marks the governed noun as oblique and flags all
            positions inside the compound so Pass 2 skips them entirely.
            Non-overlapping matches in the same sentence are all detected.

        Pass 2 — Single-token signals (skips positions flagged in Pass 1).
            • Single-token oblique postpositions — with clitic-transparent
              lookback via _prev_content_pos.
            • Genitive agreement (का/की/के) — emits a gender (and for का,
              number) signal on the nearest non-clitic token to the right.
              Tokens already consumed by a compound (e.g. "के" in "के बारे में")
              are skipped, preventing a spurious gender signal on "बारे".
            • Quantifier number signals — also clitic-transparent lookahead.
            • Negation mood constraint — first negation word wins.
            • Verb agreement features — distilled from all verb candidates.
        """
        signals = _SignalMap()
        n = len(tokens)

        # ── Pass 1: compound postposition detection ───────────────────────────
        in_compound: set[int] = set()

        for compound in _COMPOUND_POSTPOSITIONS_SORTED:   # longest first
            clen = len(compound)
            for start in range(n - clen + 1):
                if tuple(tokens[start : start + clen]) != compound:
                    continue
                # Reject if any position is already claimed by a longer compound
                if any(j in in_compound for j in range(start, start + clen)):
                    continue
                # Governed noun: nearest non-clitic token to the left
                noun_pos = _prev_content_pos(tokens, start)
                if noun_pos >= 0:
                    signals.oblique[noun_pos] = True
                for j in range(start, start + clen):
                    in_compound.add(j)

        # ── Pass 2: single-token signals ─────────────────────────────────────
        for i, tok in enumerate(tokens):

            # All signals for tokens inside a compound are already handled
            if i in in_compound:
                continue

            # ── Oblique postposition ───────────────────────────────────────────
            if tok in _OBL_POSTPOSITIONS:
                noun_pos = _prev_content_pos(tokens, i)
                # Guard: don't mark another postposition as oblique
                if noun_pos >= 0 and tokens[noun_pos] not in _POSTPOSITIONS:
                    signals.oblique[noun_pos] = True

            # ── Genitive agreement → gender (and number for का) ───────────────
            # का/की/के agree with their HEAD noun (the noun to their right).
            # setdefault preserves any stronger signal already present.
            if tok in _GENITIVE_AGREEMENT:
                gender_sig, number_sig = _GENITIVE_AGREEMENT[tok]
                head_pos = _next_content_pos(tokens, i)
                if head_pos < n:
                    signals.gender.setdefault(head_pos, gender_sig)
                    if number_sig is not None:
                        signals.number.setdefault(head_pos, number_sig)

            # ── Quantifier → number of following content token ────────────────
            # Quantifier signals override genitive number signals (stronger).
            if tok in _SINGULAR_QUANTIFIERS:
                head_pos = _next_content_pos(tokens, i)
                if head_pos < n:
                    signals.number[head_pos] = "SG"
            elif tok in _PLURAL_QUANTIFIERS:
                head_pos = _next_content_pos(tokens, i)
                if head_pos < n:
                    signals.number[head_pos] = "PL"

            # ── Negation → sentence-level verb mood constraint ─────────────────
            # First negation word wins; multiple negations in one simple clause
            # are uncommon and the first is almost always the relevant one.
            if tok in _NEGATION and signals.negation_mood is None:
                signals.negation_mood = _NEGATION[tok]

            # ── Verb agreement features ────────────────────────────────────────
            for vc in verb_candidates_by_pos.get(i, []):
                if vc.gender and vc.number:
                    num = _VERB_NUM_MAP.get(vc.number)
                    if num:
                        signals.verb_agreement.append((vc.gender, num))

        return signals

    # =========================================================================
    # Step 4a — Disambiguate noun candidates
    # =========================================================================

    def _disambiguate_noun(
        self,
        pos: int,
        candidates: list[NounCandidate],
        signals: _SignalMap,
    ) -> tuple[list[NounCandidate], list[str]]:
        """
        Filter noun candidates using signals from the signal map.

        Rules are applied in strict priority order; each rule only commits
        if at least one candidate survives the filter (safe-filter pattern).

            Rule 0 — genitive gender agreement  (strongest)
            Rule 1 — postposition → oblique case
            Rule 2 — quantifier / genitive → number
            Rule 3 — verb agreement             (weakest; DIR-case only)

        Returns (filtered_candidates, list_of_signal_labels_applied).
        """
        applied: list[str] = []

        if not candidates:
            return candidates, applied

        # ── Rule 0: Genitive gender agreement ─────────────────────────────────
        # Example: "की किताब" → किताब must be F
        # Example: "का घर"   → घर must be M SG
        if pos in signals.gender:
            target_gender = signals.gender[pos]
            filtered = [c for c in candidates if c.gender == target_gender]
            if filtered:
                candidates = filtered
                applied.append(f"genitive→{target_gender}")

        # ── Rule 1: Postposition → oblique case ───────────────────────────────
        if signals.oblique.get(pos):
            filtered = [c for c in candidates if c.case == "OBL"]
            if filtered:
                candidates = filtered
                applied.append("postposition→OBL")

        # ── Rule 2: Quantifier / genitive number ──────────────────────────────
        if pos in signals.number:
            target_num = signals.number[pos]
            filtered = [c for c in candidates if c.number == target_num]
            if filtered:
                candidates = filtered
                applied.append(f"quantifier→{target_num}")

        # ── Rule 3: Verb agreement (DIR-case, non-oblique tokens only) ────────
        if signals.verb_agreement:
            has_dir = any(c.case == "DIR" for c in candidates)
            if has_dir and not signals.oblique.get(pos):
                for (v_gender, v_number) in signals.verb_agreement:
                    filtered = [
                        c for c in candidates
                        if c.gender == v_gender and c.number == v_number
                    ]
                    if filtered:
                        candidates = filtered
                        applied.append(f"verb_agreement→{v_gender}{v_number}")
                        break

        return candidates, applied

    # =========================================================================
    # Step 4b — Disambiguate verb candidates
    # =========================================================================

    def _disambiguate_verb(
        self,
        candidates: list[VerbCandidate],
        signals: _SignalMap,
    ) -> tuple[list[VerbCandidate], list[str]]:
        """
        Filter verb candidates using sentence-level signals.

        Currently handles negation mood constraints:
            नहीं / न / ना  →  exclude imperative mood readings
            मत             →  keep only imperative mood readings

        Scope caveat: negation_mood is treated as sentence-level. In sentences
        with multiple clauses the signal may constrain the wrong verb. This is
        an acknowledged limitation; a proper fix requires dependency parsing.

        Returns (filtered_candidates, list_of_signal_labels_applied).
        """
        applied: list[str] = []

        if not candidates:
            return candidates, applied

        # ── Negation mood constraint ──────────────────────────────────────────
        if signals.negation_mood == "nahi":
            # नहीं/न/ना: declarative negation — imperative reading impossible
            filtered = [c for c in candidates if c.mood != "IMP"]
            if filtered:
                candidates = filtered
                applied.append("negation(नहीं)→¬IMP")

        elif signals.negation_mood == "mat":
            # मत: prohibitive — verb must be in imperative form
            filtered = [c for c in candidates if c.mood == "IMP"]
            if filtered:
                candidates = filtered
                applied.append("negation(मत)→IMP")

        return candidates, applied

    # =========================================================================
    # Step 5 — Gender model for out-of-lexicon words
    # =========================================================================

    def _apply_gender_model(
        self,
        token: str,
        existing_noun_candidates: list[NounCandidate],
    ) -> list[NounCandidate]:
        """
        If this token has no noun candidates AND a gender model is loaded,
        ask the model for a predicted gender and synthesize candidates.

        Paradigm is inferred from the token's final character:
            consonant-final + M → M2,  consonant-final + F → F3
        (the two most frequent consonant-final paradigm classes.)

        Synthesized candidates include all (SG/PL) × (DIR/OBL) combinations;
        the rule-based filters in steps 4a/4b prune them normally.
        """
        if existing_noun_candidates or self._gender_model is None:
            return existing_noun_candidates

        predicted_gender = self._gender_model.predict(token)
        if predicted_gender not in ("M", "F"):
            return existing_noun_candidates

        last = token[-1] if token else ""
        _ENDING_CLASS: dict[str, tuple[str, str]] = {
            "ा": ("M1", "F2"),
            "े": ("M1", "F1"),
            "ी": ("M3", "F1"),
            "ि": ("M3", "F4"),
            "ू": ("M4", "F5"),
            "ु": ("M4", "F5"),
        }
        pair = _ENDING_CLASS.get(last, ("M2", "F3"))
        paradigm = pair[0] if predicted_gender == "M" else pair[1]

        synth: list[NounCandidate] = []
        for number in ("SG", "PL"):
            for case in ("DIR", "OBL"):
                synth.append(NounCandidate(
                    lemma=token,
                    gender=predicted_gender,
                    number=number,
                    case=case,
                    paradigm=paradigm,
                    confidence="predicted",
                    source="gender_model",
                ))
        return synth

    # =========================================================================
    # Step 6 — Public entry point
    # =========================================================================

    def disambiguate(self, sentence: str) -> list[TokenResult]:
        """
        Tokenize, analyse, and disambiguate a Hindi sentence.

        Returns a list of TokenResult objects in token order.
        Each TokenResult carries the surviving candidates (noun and/or verb)
        after all context signals have been applied.

        Tokens with no analysis (function words, punctuation, unknown words)
        are included with empty candidate lists and has_analysis=False.
        """
        # ── Step 1 ────────────────────────────────────────────────────────────
        tokens = self._tokenize(sentence)

        # ── Step 2 ────────────────────────────────────────────────────────────
        noun_cands_by_pos: dict[int, list[NounCandidate]] = {}
        verb_cands_by_pos: dict[int, list[VerbCandidate]] = {}

        for i, tok in enumerate(tokens):
            noun_cands_by_pos[i] = self._analyze_token_nouns(tok)
            verb_cands_by_pos[i] = self._analyze_token_verbs(tok)

        # ── Step 3 ────────────────────────────────────────────────────────────
        signal_map = self._build_signal_map(tokens, verb_cands_by_pos)

        # ── Steps 4 & 5 ───────────────────────────────────────────────────────
        results: list[TokenResult] = []

        for i, tok in enumerate(tokens):
            noun_cands = noun_cands_by_pos[i]
            verb_cands = verb_cands_by_pos[i]

            # Step 5: synthesize candidates for OOV words (must run before 4a)
            noun_cands = self._apply_gender_model(tok, noun_cands)

            # Step 4a: filter noun candidates with context signals
            noun_cands, noun_signals = self._disambiguate_noun(
                i, noun_cands, signal_map
            )

            # Step 4b: filter verb candidates (negation mood, etc.)
            verb_cands, verb_signals = self._disambiguate_verb(
                verb_cands, signal_map
            )

            # Merge signal labels; verb-specific ones are prefixed for clarity
            applied = noun_signals + [f"verb:{s}" for s in verb_signals]

            # Determine ambiguity — more than one distinct reading remaining
            noun_ambiguous = len({_reading_key(c) for c in noun_cands}) > 1
            verb_ambiguous = len({
                (c.lemma, c.gender or "", c.number or "",
                 c.aspect or "", c.tense or "")
                for c in verb_cands
            }) > 1
            is_ambiguous = noun_ambiguous or verb_ambiguous

            results.append(TokenResult(
                token=tok,
                position=i,
                noun_candidates=noun_cands,
                verb_candidates=verb_cands,
                ambiguous=is_ambiguous,
                signals_applied=applied,
            ))

        return results


# ===========================================================================
# Convenience function
# ===========================================================================

def disambiguate_sentence(
    sentence: str,
    noun_lexicon_path: str,
    verb_data_dir: str,
    gender_model: Optional[GenderModel] = None,
    verbose: bool = True,
) -> list[TokenResult]:
    """One-shot helper: build a Disambiguator, run it, and optionally print."""
    d = Disambiguator(
        noun_lexicon_path=noun_lexicon_path,
        verb_data_dir=verb_data_dir,
        gender_model=gender_model,
    )
    results = d.disambiguate(sentence)
    if verbose:
        print(f"\nSentence: {sentence!r}")
        print("─" * 60)
        for r in results:
            print(r)
            print()
    return results


# ===========================================================================
# CLI demo
# ===========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python disambiguator.py <noun_lexicon.tsv> <verb_data_dir> [sentence]")
        sys.exit(1)

    lex_path = sys.argv[1]
    verb_dir = sys.argv[2]
    sentence = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else None

    d = Disambiguator(noun_lexicon_path=lex_path, verb_data_dir=verb_dir)

    if sentence:
        results = d.disambiguate(sentence)
        print(f"\nSentence: {sentence!r}")
        print("─" * 60)
        for r in results:
            print(r)
    else:
        print(f"Hindi Disambiguator  (lexicon: {lex_path}, verbs: {verb_dir})")
        print("Type a sentence and press Enter.  'q' to quit.\n")
        while True:
            try:
                line = input("› ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                break
            results = d.disambiguate(line)
            print()
            for r in results:
                print(r)
            print()