"""
disambiguator.py — Context-based morphological disambiguator for Hindi
=======================================================================
Takes a raw Hindi sentence and returns a disambiguated token list by
scoring each token's candidate analyses using postposition, quantifier,
genitive-agreement, verb-agreement, bigram-prior, and coordination
signals extracted from the surrounding context.

Public API
----------
    from disambiguator import Disambiguator, BigramModel

    # Optional: train a bigram model from HDTB treebank files
    bigram = BigramModel.from_treebank(
        path='path/to/hdtb/',
        lexicon_lookup=lex_lookup,   # from load_lexicon()
    )

    d = Disambiguator(
            noun_lexicon_path='Nouns/data/noun_lexicon_expanded.tsv',
            verb_data_dir='path/to/morphin_files',
            bigram_model=bigram,         # optional
            gender_model=my_model,       # optional
        )
    result = d.disambiguate('लड़की ने खाना खाया')
    for tok in result:
        print(tok)

Design notes
------------
* Scoring model — candidates are scored across all signals rather than
  hard-filtered in a fixed priority chain.  Each signal adds or subtracts
  weight; the top-scoring candidates are returned.  This means a weak
  but convergent combination of signals can out-rank a single weak signal,
  and a clearly wrong candidate (negative total) is pruned even if a
  stricter filter would have kept it.  Signal weights:

      Genitive gender    ±8   (explicit morphological agreement)
      Oblique case       ±4   (postposition forces case)
      Number             +2   (quantifier / genitive number)
      Nearest verb agr   +1   (weak; subject may be dropped)
      Bigram prior       0–2  (scaled P(paradigm_j | paradigm_i))

  Negative weights are only applied for genitive gender and oblique case,
  which are near-certain when they fire.  All other mismatches are omitted
  (no penalty) to avoid over-pruning.

* Bigram model — trained on consecutive noun paradigm class pairs from
  HDTB treebank data.  Lemmas are looked up in the classified lexicon
  for M1–F5 assignment; unknown lemmas fall back to gen+case coarse classes
  (M2 / F3 as the most frequent consonant-final representatives).

* Nearest-subject verb agreement — agreement features are taken only from
  the first verb to the right of the noun in question, not from all verbs
  in the sentence.  This prevents a distant verb from constraining a noun
  that it cannot plausibly agree with.

* Coordination — a post-processing pass propagates case across और-joined
  noun phrases: if one conjunct has a single resolved case and the other is
  ambiguous, the resolved case is inherited.

* Clitics (भी, ही, तो) are transparent for postposition and quantifier
  lookback/lookahead.

* Compound postpositions are detected longest-match-first before single-
  token signals to prevent double-firing.

* Ergative ने → object agreement and cross-sentence memory are not
  implemented.  Any unresolvable token is returned with ambiguous=True.
"""

from __future__ import annotations

import os
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))
_NOUNS_DIR = os.path.join(_HERE, "Nouns")
if _NOUNS_DIR not in sys.path:
    sys.path.insert(0, _NOUNS_DIR)

from Nouns.analyzer import load_lexicon, analyze, normalize_hindi
from Nouns.noun_paradigm_templates import PARADIGM_TABLES
from verb_analyzer import VerbAnalyzer


# ===========================================================================
# Module-level linguistic constants  (unchanged from original)
# ===========================================================================

_CLITICS: frozenset[str] = frozenset({"भी", "ही", "तो"})

_POSTPOSITIONS: frozenset[str] = frozenset({
    "को", "में", "से", "ने", "पर", "तक",
    "के", "की", "का", "द्वारा",
    "पास", "साथ", "बिना",
    "जैसा", "जैसी", "जैसे",
    "बाद", "पहले", "आगे", "पीछे",
    "ऊपर", "नीचे", "अंदर", "बाहर",
    "बीच", "सामने", "खिलाफ", "अनुसार",
    "अलावा", "दौरान", "बावजूद", "विरुद्ध",
    "तरफ", "ओर", "तरह", "जगह", "बजाय", "खातिर",
    "लिए", "बारे", "रूप", "आधार", "नाम",
    "वजह", "तुलना", "मुकाबले", "बदले", "बराबर",
})

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

_COMPOUND_POSTPOSITIONS: dict[tuple[str, ...], str] = {
    ("के", "लिए"): "OBL",    ("के", "साथ"): "OBL",
    ("के", "बाद"): "OBL",    ("के", "ऊपर"): "OBL",
    ("के", "नीचे"): "OBL",   ("के", "अंदर"): "OBL",
    ("के", "बाहर"): "OBL",   ("के", "आगे"): "OBL",
    ("के", "पीछे"): "OBL",   ("के", "पास"): "OBL",
    ("के", "बीच"): "OBL",    ("के", "खिलाफ"): "OBL",
    ("के", "बिना"): "OBL",   ("के", "द्वारा"): "OBL",
    ("के", "अनुसार"): "OBL", ("के", "अलावा"): "OBL",
    ("के", "सामने"): "OBL",  ("के", "दौरान"): "OBL",
    ("के", "बावजूद"): "OBL", ("के", "विरुद्ध"): "OBL",
    ("के", "बजाय"): "OBL",   ("के", "बराबर"): "OBL",
    ("की", "तरफ"): "OBL",    ("की", "ओर"): "OBL",
    ("की", "तरह"): "OBL",    ("की", "जगह"): "OBL",
    ("की", "बजाय"): "OBL",   ("की", "खातिर"): "OBL",
    ("से", "पहले"): "OBL",   ("से", "दूर"): "OBL",
    ("से", "ज़्यादा"): "OBL", ("से", "ज्यादा"): "OBL",
    ("से", "कम"): "OBL",     ("से", "लेकर"): "OBL",
    ("से", "अलग"): "OBL",    ("से", "परे"): "OBL",
    ("से", "बाहर"): "OBL",
    ("को", "छोड़कर"): "OBL", ("को", "लेकर"): "OBL",
    ("के", "बारे", "में"): "OBL",    ("के", "बदले", "में"): "OBL",
    ("के", "आधार", "पर"): "OBL",    ("के", "रूप", "में"): "OBL",
    ("के", "नाम", "पर"): "OBL",     ("के", "साथ", "साथ"): "OBL",
    ("के", "मुकाबले", "में"): "OBL", ("के", "विरोध", "में"): "OBL",
    ("की", "वजह", "से"): "OBL",     ("की", "तुलना", "में"): "OBL",
}

_COMPOUND_POSTPOSITIONS_SORTED: list[tuple[str, ...]] = sorted(
    _COMPOUND_POSTPOSITIONS, key=len, reverse=True
)

_SINGULAR_QUANTIFIERS: frozenset[str] = frozenset({
    "एक", "इस", "उस", "किस", "हर", "प्रत्येक",
})

_PLURAL_QUANTIFIERS: frozenset[str] = frozenset({
    "सब", "सभी", "कई", "कुछ", "बहुत", "अनेक",
    "ये", "वे", "इन", "उन",
})

_GENITIVE_AGREEMENT: dict[str, tuple[str, Optional[str]]] = {
    "का": ("M", "SG"),
    "की": ("F", None),
    "के": ("M", None),
}

_NEGATION: dict[str, str] = {
    "नहीं": "nahi",
    "न":    "nahi",
    "ना":   "nahi",
    "मत":   "mat",
}

_VERB_NUM_MAP: dict[str, str] = {"S": "SG", "P": "PL"}


# ===========================================================================
# Bigram model
# ===========================================================================

class BigramModel:
    """
    Noun paradigm bigram model trained on HDTB treebank data.

    Models P(paradigm_j | paradigm_i) over consecutive noun tokens
    within each sentence.  Used as a soft prior (+0 to +2 weight)
    in the candidate scoring step.

    Training
    --------
    For each noun token (CPOS == 'NN'), the paradigm class is determined:
      1. Lexicon lookup: nukta-stripped NFC lemma → M1–F5 from load_lexicon()
      2. Fallback: gen + case from the feature string → M2 or F3
         (the most frequent consonant-final classes for each gender)
      3. 'UNK' if neither is available — excluded from bigram counts

    Smoothing
    ---------
    Additive (Laplace-like) smoothing with α = 0.1 prevents zero probabilities
    for paradigm pairs that were not observed in the training corpus.
    """

    _ALPHA = 0.1   # smoothing constant

    def __init__(self) -> None:
        self._counts: dict[str, Counter] = defaultdict(Counter)
        self._vocab:  set[str]           = set()

    @classmethod
    def from_treebank(
        cls,
        path: str,
        lexicon_lookup: dict[str, str],
    ) -> "BigramModel":
        """
        Parse HDTB-format .dat files at *path* (single file or directory)
        and return a trained BigramModel.

        Parameters
        ----------
        path : str
            An HDTB .dat file, or a directory tree of .dat files.
        lexicon_lookup : dict
            Nukta-stripped NFC lemma → paradigm class, as returned by
            load_lexicon().  Used for priority-1 class assignment.
        """
        model = cls()
        for fp in _collect_treebank_files(path):
            for sent_classes in _parse_hdtb_noun_classes(fp, lexicon_lookup):
                model._train_sentence(sent_classes)
        return model

    def _train_sentence(self, classes: list[str]) -> None:
        """Record (prev, curr) paradigm class pairs for one sentence."""
        for prev_cls, curr_cls in zip(classes, classes[1:]):
            if prev_cls == "UNK" or curr_cls == "UNK":
                continue
            self._counts[prev_cls][curr_cls] += 1
            self._vocab.update((prev_cls, curr_cls))

    def score(self, prev_class: str, curr_class: str) -> float:
        """
        Return P(curr_class | prev_class) with additive smoothing.

        Returns 0.0 for an unseen *prev_class* so that an unresolved
        previous noun contributes nothing to the current position's score.
        """
        counts = self._counts.get(prev_class)
        if not counts:
            return 0.0
        vocab_size = max(len(self._vocab), 1)
        total      = sum(counts.values()) + self._ALPHA * vocab_size
        numerator  = counts.get(curr_class, 0) + self._ALPHA
        return numerator / total


# ---------------------------------------------------------------------------
# Treebank helpers (used by BigramModel.from_treebank)
# ---------------------------------------------------------------------------

def _collect_treebank_files(path: str) -> list[str]:
    """Return [path] if a file, or all .dat files under path if a directory."""
    if os.path.isfile(path):
        return [path]
    found = []
    for root, _, fnames in os.walk(path):
        for fn in sorted(fnames):
            if fn.endswith(".dat"):
                found.append(os.path.join(root, fn))
    return sorted(found)


def _parse_hdtb_feats(feat_str: str) -> dict[str, str]:
    """
    Parse 'cat-n|gen-f|num-sg|pers-3|case-o|...' into a plain dict.
    Splits on the first '-' only so hyphenated values survive.
    """
    result: dict[str, str] = {}
    for part in feat_str.split("|"):
        idx = part.find("-")
        if idx == -1:
            continue
        result[part[:idx]] = part[idx + 1:]
    return result


# gen+case → coarse paradigm class for lemmas not in our lexicon.
# M2 and F3 are the largest consonant-final classes and the safest fallback.
_GEN_CASE_FALLBACK: dict[tuple[str, str], str] = {
    ("m", "d"): "M2",
    ("m", "o"): "M2",
    ("f", "d"): "F3",
    ("f", "o"): "F3",
}


def _parse_hdtb_noun_classes(
    filepath: str,
    lexicon_lookup: dict[str, str],
) -> list[list[str]]:
    """
    Parse one HDTB .dat file.  Returns a list of sentences, where each
    sentence is the ordered list of paradigm class strings for its NN tokens.

    Sentences are delimited by blank lines.  Lines starting with '#' are
    skipped.  Only tokens with CPOS == 'NN' contribute a class string.

    HDTB column layout (0-indexed, tab-separated):
        0  tok_id   1  surface   2  lemma   3  pos   4  CPOS
        5  feats    6  head      7  deprel  8  _     9  _
    """
    sentences: list[list[str]] = []
    current:   list[str]       = []

    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")

            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue

            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 6:
                continue

            if cols[4].strip() != "NN":
                continue

            lemma    = unicodedata.normalize("NFC", cols[2].strip())
            stripped = normalize_hindi(lemma)

            # Priority 1: exact lexicon lookup
            if stripped in lexicon_lookup:
                current.append(lexicon_lookup[stripped])
                continue

            # Priority 2: gen + case coarse fallback
            feats = _parse_hdtb_feats(cols[5].strip())
            key   = (feats.get("gen", ""), feats.get("case", ""))
            current.append(_GEN_CASE_FALLBACK.get(key, "UNK"))

    if current:
        sentences.append(current)

    return sentences


# ===========================================================================
# Module-level helper functions
# ===========================================================================

def _prev_content_pos(tokens: list[str], i: int) -> int:
    """
    Index of the nearest non-clitic token strictly before position i.
    Returns -1 if none exists.
    """
    j = i - 1
    while j >= 0 and tokens[j] in _CLITICS:
        j -= 1
    return j


def _next_content_pos(tokens: list[str], i: int) -> int:
    """
    Index of the nearest non-clitic token strictly after position i.
    Returns len(tokens) if none exists.
    """
    j = i + 1
    while j < len(tokens) and tokens[j] in _CLITICS:
        j += 1
    return j


def _reading_key(c: "NounCandidate") -> tuple:
    """Canonical key for a NounCandidate — used to count distinct readings."""
    return (c.lemma, c.gender, c.number, c.case)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class NounCandidate:
    """One morphological reading of a noun token."""
    lemma:      str
    gender:     str    # 'M' | 'F'
    number:     str    # 'SG' | 'PL'
    case:       str    # 'DIR' | 'OBL'
    paradigm:   str    # e.g. 'M1', 'F3'
    confidence: str    # 'certain' | 'heuristic' | 'predicted'
    source:     str = "lexicon"   # 'lexicon' | 'gender_model'


@dataclass
class VerbCandidate:
    """One morphological reading of a verb token."""
    lemma:       str
    suffix:      str
    paradigm:    str
    gender:      Optional[str]
    number:      Optional[str]
    person:      Optional[str]
    aspect:      Optional[str]
    tense:       Optional[str]
    mood:        Optional[str]
    verbal_type: Optional[str]
    confidence:  str
    irregular:   bool = False


@dataclass
class TokenResult:
    """Disambiguated result for a single token."""
    token:           str
    position:        int
    noun_candidates: list[NounCandidate] = field(default_factory=list)
    verb_candidates: list[VerbCandidate] = field(default_factory=list)
    ambiguous:       bool = False
    signals_applied: list[str] = field(default_factory=list)

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
        for c in self.noun_candidates:
            parts.append(
                f"  NOUN  lemma={c.lemma!r:<12} "
                f"{c.paradigm} {c.gender} {c.number} {c.case} conf={c.confidence}"
            )
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
# Gender model protocol
# ===========================================================================

@runtime_checkable
class GenderModel(Protocol):
    """
    Minimal interface for a pluggable gender prediction model.
    Returns 'M', 'F', or None (abstain).
    """
    def predict(self, word: str) -> Optional[str]: ...


# ===========================================================================
# Signal map
# ===========================================================================

@dataclass
class _SignalMap:
    """
    All context signals for a sentence, indexed by token position.

    oblique[i]               True           → token i is forced oblique
    number[i]                'SG' | 'PL'   → token i's number is constrained
    gender[i]                'M'  | 'F'    → token i's gender is constrained
    verb_agreement_by_pos[v] [(g, n), …]   → agreement pairs from verb at v
    verb_positions           [v, …]        → sorted positions with verb candidates
    negation_mood            'nahi'|'mat'  → sentence-level negation
    """
    oblique:               dict[int, bool]               = field(default_factory=dict)
    number:                dict[int, str]                = field(default_factory=dict)
    gender:                dict[int, str]                = field(default_factory=dict)
    verb_agreement_by_pos: dict[int, list[tuple[str, str]]] = field(default_factory=dict)
    verb_positions:        list[int]                     = field(default_factory=list)
    negation_mood:         Optional[str]                 = None


# ===========================================================================
# Main disambiguator
# ===========================================================================

class Disambiguator:
    """
    Context-based morphological disambiguator for Hindi.

    Parameters
    ----------
    noun_lexicon_path : str
        Path to noun_lexicon_expanded.tsv (or noun_lexicon_final.tsv).
    verb_data_dir : str
        Directory containing MorpHIN verb data files.
    gender_model : GenderModel | None
        Optional pluggable gender predictor for OOV consonant-final nouns.
    bigram_model : BigramModel | None
        Optional trained noun paradigm bigram model.  When supplied,
        adds a soft prior of up to +2 weight per candidate in the scorer.
        Train with BigramModel.from_treebank() before passing in.
    """

    def __init__(
        self,
        noun_lexicon_path: str,
        verb_data_dir: str,
        gender_model: Optional[GenderModel] = None,
        bigram_model: Optional[BigramModel] = None,
    ) -> None:
        self._lex_lookup, self._lex_display, self._lex_confidence = load_lexicon(
            noun_lexicon_path
        )
        self._paradigm_tables = PARADIGM_TABLES
        self._verb_analyzer   = VerbAnalyzer(data_dir=verb_data_dir)
        self._gender_model    = gender_model
        self._bigram          = bigram_model

    # =========================================================================
    # Step 1 — Tokenize
    # =========================================================================

    def _tokenize(self, sentence: str) -> list[str]:
        return [
            unicodedata.normalize("NFC", tok)
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
        return [
            VerbCandidate(
                lemma=a.lemma,
                suffix=a.suffix,
                paradigm=a.paradigm,
                gender=a.features.gender,
                number=a.features.number,
                person=a.features.person,
                aspect=a.features.aspect,
                tense=a.features.tense,
                mood=a.features.mood,
                verbal_type=a.features.verbal_type,
                confidence=a.confidence,
                irregular=a.irregular,
            )
            for a in self._verb_analyzer.analyze(token)
        ]

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

        Pass 1 — Compound postpositions (longest-match-first).
            Governed noun (nearest non-clitic token to the left) is flagged
            oblique.  All positions inside the compound are marked so Pass 2
            does not double-fire on them.

        Pass 2 — Single-token signals.
            • Oblique postpositions → oblique signal on left neighbour
            • Genitive agreement (का/की/के) → gender (and number for का)
              signal on the nearest non-clitic token to the right
            • Quantifier → number signal on nearest right neighbour
            • Negation → sentence-level mood constraint on verbs
            • Verb agreement → stored per verb position (not sentence-global)
        """
        signals = _SignalMap()
        n       = len(tokens)

        # ── Pass 1: compound postpositions ────────────────────────────────────
        in_compound: set[int] = set()

        for compound in _COMPOUND_POSTPOSITIONS_SORTED:
            clen = len(compound)
            for start in range(n - clen + 1):
                if tuple(tokens[start : start + clen]) != compound:
                    continue
                if any(j in in_compound for j in range(start, start + clen)):
                    continue
                noun_pos = _prev_content_pos(tokens, start)
                if noun_pos >= 0:
                    signals.oblique[noun_pos] = True
                for j in range(start, start + clen):
                    in_compound.add(j)

        # ── Pass 2: single-token signals ──────────────────────────────────────
        for i, tok in enumerate(tokens):

            if i in in_compound:
                continue

            # Oblique postposition
            if tok in _OBL_POSTPOSITIONS:
                noun_pos = _prev_content_pos(tokens, i)
                if noun_pos >= 0 and tokens[noun_pos] not in _POSTPOSITIONS:
                    signals.oblique[noun_pos] = True

            # Genitive agreement → gender / number of right neighbour
            if tok in _GENITIVE_AGREEMENT:
                gender_sig, number_sig = _GENITIVE_AGREEMENT[tok]
                head_pos = _next_content_pos(tokens, i)
                if head_pos < n:
                    signals.gender.setdefault(head_pos, gender_sig)
                    if number_sig is not None:
                        signals.number.setdefault(head_pos, number_sig)

            # Quantifier → number of right neighbour (overrides genitive number)
            if tok in _SINGULAR_QUANTIFIERS:
                head_pos = _next_content_pos(tokens, i)
                if head_pos < n:
                    signals.number[head_pos] = "SG"
            elif tok in _PLURAL_QUANTIFIERS:
                head_pos = _next_content_pos(tokens, i)
                if head_pos < n:
                    signals.number[head_pos] = "PL"

            # Negation
            if tok in _NEGATION and signals.negation_mood is None:
                signals.negation_mood = _NEGATION[tok]

            # Verb agreement — stored per position, not as a flat list.
            # Only positions that actually produce agreement pairs are recorded.
            verb_cands = verb_candidates_by_pos.get(i, [])
            if verb_cands:
                pairs = [
                    (vc.gender, _VERB_NUM_MAP[vc.number])
                    for vc in verb_cands
                    if vc.gender and vc.number and vc.number in _VERB_NUM_MAP
                ]
                if pairs:
                    signals.verb_agreement_by_pos[i] = pairs
                    signals.verb_positions.append(i)

        signals.verb_positions.sort()
        return signals

    # =========================================================================
    # Step 4a — Score noun candidates
    # =========================================================================

    def _score_noun_candidates(
        self,
        pos: int,
        candidates: list[NounCandidate],
        signals: _SignalMap,
        prev_paradigm: Optional[str],
    ) -> tuple[list[NounCandidate], list[str]]:
        """
        Assign a score to each candidate based on all available context
        signals, then return the subset tied at the maximum score.

        Signal weights
        --------------
        Genitive gender    ±8  — explicit morphological agreement; very reliable
        Oblique case       ±4  — postposition forces case; very reliable
        Number             +2  — quantifier / genitive number; medium reliability
        Nearest verb agr   +1  — only from the first verb to the right of pos;
                                  only applied to DIR-case candidates that are
                                  not already forced oblique
        Bigram prior       0–2 — P(paradigm | prev_paradigm) scaled by 2

        Negative weights are applied only for genitive and case signals,
        which are near-certain.  All other signal mismatches contribute 0
        (no penalty) to avoid over-pruning on weaker signals.

        If every candidate scores 0 (no signal fired), all candidates are
        returned unchanged — correct behaviour for function words and
        tokens with no surrounding context.
        """
        applied: list[str] = []
        if not candidates:
            return candidates, applied

        scores = [0.0] * len(candidates)

        # ── Genitive gender  (±8) ─────────────────────────────────────────────
        if pos in signals.gender:
            tg = signals.gender[pos]
            applied.append(f"genitive→{tg}")
            for i, c in enumerate(candidates):
                scores[i] += 8.0 if c.gender == tg else -8.0

        # ── Oblique case  (±4) ────────────────────────────────────────────────
        if signals.oblique.get(pos):
            applied.append("postposition→OBL")
            for i, c in enumerate(candidates):
                scores[i] += 4.0 if c.case == "OBL" else -4.0

        # ── Number  (+2 for match, 0 for mismatch) ────────────────────────────
        if pos in signals.number:
            tn = signals.number[pos]
            applied.append(f"quantifier→{tn}")
            for i, c in enumerate(candidates):
                if c.number == tn:
                    scores[i] += 2.0

        # ── Nearest verb agreement  (+1, DIR candidates only) ─────────────────
        # Find the first verb position strictly to the right of pos.
        # Only applies when the token is not already forced oblique, since
        # an oblique NP cannot be the subject that the verb agrees with.
        nearest_verb: Optional[int] = None
        for vpos in signals.verb_positions:   # already sorted
            if vpos > pos:
                nearest_verb = vpos
                break

        if nearest_verb is not None and not signals.oblique.get(pos):
            has_dir = any(c.case == "DIR" for c in candidates)
            if has_dir:
                for vg, vn in signals.verb_agreement_by_pos.get(nearest_verb, []):
                    applied.append(f"verb_agreement→{vg}{vn}")
                    for i, c in enumerate(candidates):
                        if c.case == "DIR" and c.gender == vg and c.number == vn:
                            scores[i] += 1.0
                    break   # first consistent agreement pair wins

        # ── Bigram prior  (0 to +2) ───────────────────────────────────────────
        # P(curr_paradigm | prev_paradigm) scaled to [0, 2].
        # prev_paradigm is None at the start of the sentence or after an
        # ambiguous noun, in which case BigramModel.score() returns 0.0 (no-op).
        if self._bigram is not None:
            for i, c in enumerate(candidates):
                p = self._bigram.score(prev_paradigm or "", c.paradigm)
                scores[i] += p * 2.0

        # ── Select top-scoring candidates ─────────────────────────────────────
        if all(s == 0.0 for s in scores):
            # No signal fired — return all candidates unchanged
            return candidates, applied

        max_score = max(scores)
        top = [c for c, s in zip(candidates, scores) if s == max_score]
        return (top if top else candidates), applied

    # =========================================================================
    # Step 4b — Disambiguate verb candidates
    # =========================================================================

    def _disambiguate_verb(
        self,
        candidates: list[VerbCandidate],
        signals: _SignalMap,
    ) -> tuple[list[VerbCandidate], list[str]]:
        """
        Filter verb candidates using sentence-level negation mood constraints.

            नहीं / न / ना  →  exclude imperative readings
            मत             →  keep only imperative readings
        """
        applied: list[str] = []
        if not candidates:
            return candidates, applied

        if signals.negation_mood == "nahi":
            filtered = [c for c in candidates if c.mood != "IMP"]
            if filtered:
                candidates = filtered
                applied.append("negation(नहीं)→¬IMP")

        elif signals.negation_mood == "mat":
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
        If this token has no noun candidates and a gender model is loaded,
        ask the model for a predicted gender and synthesize candidates.

        Paradigm is inferred from the token's final character:
            consonant-final + M → M2,  consonant-final + F → F3
        """
        if existing_noun_candidates or self._gender_model is None:
            return existing_noun_candidates

        predicted_gender = self._gender_model.predict(token)
        if predicted_gender not in ("M", "F"):
            return existing_noun_candidates

        last = token[-1] if token else ""
        _ENDING_CLASS: dict[str, tuple[str, str]] = {
            "ा": ("M1", "F2"), "े": ("M1", "F1"),
            "ी": ("M3", "F1"), "ि": ("M3", "F4"),
            "ू": ("M4", "F5"), "ु": ("M4", "F5"),
        }
        pair    = _ENDING_CLASS.get(last, ("M2", "F3"))
        paradigm = pair[0] if predicted_gender == "M" else pair[1]

        return [
            NounCandidate(
                lemma=token, gender=predicted_gender,
                number=num, case=case,
                paradigm=paradigm, confidence="predicted",
                source="gender_model",
            )
            for num  in ("SG", "PL")
            for case in ("DIR", "OBL")
        ]

    # =========================================================================
    # Step 6 — Coordination post-processing
    # =========================================================================

    def _apply_coordination(self, results: list[TokenResult]) -> list[TokenResult]:
        """
        Propagate case across और-coordinated noun phrases.

        For each 'और' token, find the nearest noun-bearing token on each side
        (skipping clitics and non-noun tokens).  If one conjunct has a single
        resolved case and the other is case-ambiguous, filter the ambiguous
        conjunct to match the resolved case.

        Signal label 'coordination→CASE' is appended to signals_applied for
        any token whose candidates are narrowed.

        Both-resolved and both-ambiguous cases are left unchanged.
        """
        n = len(results)

        for i, r in enumerate(results):
            if r.token != "और":
                continue

            # Nearest noun-bearing position to the left
            left_pos: Optional[int] = None
            for j in range(i - 1, -1, -1):
                if results[j].token not in _CLITICS and results[j].noun_candidates:
                    left_pos = j
                    break

            # Nearest noun-bearing position to the right
            right_pos: Optional[int] = None
            for j in range(i + 1, n):
                if results[j].token not in _CLITICS and results[j].noun_candidates:
                    right_pos = j
                    break

            if left_pos is None or right_pos is None:
                continue

            left  = results[left_pos]
            right = results[right_pos]

            left_cases  = {c.case for c in left.noun_candidates}
            right_cases = {c.case for c in right.noun_candidates}

            def _propagate(
                source: TokenResult,
                target: TokenResult,
                target_case: str,
            ) -> TokenResult:
                """Return a new TokenResult with candidates filtered to target_case."""
                filtered = [c for c in target.noun_candidates if c.case == target_case]
                if not filtered:
                    return target   # no survivors — leave unchanged
                noun_ambig = len({_reading_key(c) for c in filtered}) > 1
                verb_ambig = len({
                    (c.lemma, c.gender or "", c.number or "",
                     c.aspect or "", c.tense or "")
                    for c in target.verb_candidates
                }) > 1
                return TokenResult(
                    token=target.token,
                    position=target.position,
                    noun_candidates=filtered,
                    verb_candidates=target.verb_candidates,
                    ambiguous=noun_ambig or verb_ambig,
                    signals_applied=target.signals_applied + [
                        f"coordination→{target_case}"
                    ],
                )

            # Left resolved, right ambiguous → right inherits left's case
            if len(left_cases) == 1 and len(right_cases) > 1:
                results[right_pos] = _propagate(left, right, next(iter(left_cases)))

            # Right resolved, left ambiguous → left inherits right's case
            elif len(right_cases) == 1 and len(left_cases) > 1:
                results[left_pos] = _propagate(right, left, next(iter(right_cases)))

        return results

    # =========================================================================
    # Step 7 — Public entry point
    # =========================================================================

    def disambiguate(self, sentence: str) -> list[TokenResult]:
        """
        Tokenize, analyse, score, and disambiguate a Hindi sentence.

        Returns a list of TokenResult objects in token order.
        Tokens with no analysis are included with empty candidate lists
        and has_analysis=False.

        Pipeline
        --------
        1. Tokenize on whitespace, NFC-normalise each token.
        2. Run noun and verb analyzers on every token.
        3. Build context signal map (compounds, postpositions, genitives,
           quantifiers, negation, per-position verb agreement).
        4. Score noun candidates with all signals + bigram prior.
           Track prev_paradigm for bigram state across the sentence.
        5. Filter verb candidates with negation mood constraint.
        6. Post-processing: propagate case across और-coordination.
        """
        tokens = self._tokenize(sentence)

        noun_cands_by_pos: dict[int, list[NounCandidate]] = {}
        verb_cands_by_pos: dict[int, list[VerbCandidate]] = {}
        for i, tok in enumerate(tokens):
            noun_cands_by_pos[i] = self._analyze_token_nouns(tok)
            verb_cands_by_pos[i] = self._analyze_token_verbs(tok)

        signal_map = self._build_signal_map(tokens, verb_cands_by_pos)

        results:       list[TokenResult] = []
        prev_paradigm: Optional[str]     = None  # bigram state

        for i, tok in enumerate(tokens):
            noun_cands = noun_cands_by_pos[i]
            verb_cands = verb_cands_by_pos[i]

            # Gender model (must run before scoring, may synthesize candidates)
            noun_cands = self._apply_gender_model(tok, noun_cands)

            # Score noun candidates with all signals + bigram prior
            noun_cands, noun_signals = self._score_noun_candidates(
                i, noun_cands, signal_map, prev_paradigm
            )

            # Filter verb candidates with negation mood constraint
            verb_cands, verb_signals = self._disambiguate_verb(
                verb_cands, signal_map
            )

            applied = noun_signals + [f"verb:{s}" for s in verb_signals]

            noun_ambiguous = len({_reading_key(c) for c in noun_cands}) > 1
            verb_ambiguous = len({
                (c.lemma, c.gender or "", c.number or "",
                 c.aspect or "", c.tense or "")
                for c in verb_cands
            }) > 1

            results.append(TokenResult(
                token=tok,
                position=i,
                noun_candidates=noun_cands,
                verb_candidates=verb_cands,
                ambiguous=noun_ambiguous or verb_ambiguous,
                signals_applied=applied,
            ))

            # Update bigram state.
            # Only update when this token resolved to a single paradigm class.
            # If still ambiguous, keep the previous resolved paradigm rather
            # than resetting — a nearby resolved noun is still informative.
            if noun_cands:
                paradigms = {c.paradigm for c in noun_cands}
                if len(paradigms) == 1:
                    prev_paradigm = next(iter(paradigms))

        # Coordination post-processing
        results = self._apply_coordination(results)

        return results


# ===========================================================================
# Convenience function
# ===========================================================================

def disambiguate_sentence(
    sentence: str,
    noun_lexicon_path: str,
    verb_data_dir: str,
    gender_model: Optional[GenderModel] = None,
    bigram_model:  Optional[BigramModel]  = None,
    verbose: bool = True,
) -> list[TokenResult]:
    """One-shot helper: build a Disambiguator, run it, and optionally print."""
    d = Disambiguator(
        noun_lexicon_path=noun_lexicon_path,
        verb_data_dir=verb_data_dir,
        gender_model=gender_model,
        bigram_model=bigram_model,
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Hindi morphological disambiguator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Without bigram model:
  python disambiguator.py Nouns/data/noun_lexicon_expanded.tsv path/to/morphin/

  # With bigram model trained on HDTB:
  python disambiguator.py Nouns/data/noun_lexicon_expanded.tsv path/to/morphin/ \\
      --treebank path/to/hdtb/

  # Single sentence:
  python disambiguator.py lex.tsv morphin/ --sentence 'लड़की ने खाना खाया'
        """,
    )
    parser.add_argument("noun_lexicon", help="Path to noun_lexicon_expanded.tsv")
    parser.add_argument("verb_data_dir", help="Directory with MorpHIN verb files")
    parser.add_argument(
        "--treebank", default=None, metavar="PATH",
        help="HDTB .dat file or directory — trains the bigram model",
    )
    parser.add_argument(
        "--sentence", default=None,
        help="Analyse a single sentence and exit",
    )
    args = parser.parse_args()

    # Load lexicon (needed for bigram training too)
    lex_lookup, _, _ = load_lexicon(args.noun_lexicon)

    bigram: Optional[BigramModel] = None
    if args.treebank:
        print(f"Training bigram model from {args.treebank} …", flush=True)
        bigram = BigramModel.from_treebank(args.treebank, lex_lookup)
        total_pairs = sum(
            sum(counter.values())
            for counter in bigram._counts.values()
        )
        print(f"  {len(bigram._counts)} conditioning classes, "
              f"{total_pairs} bigram observations\n")

    d = Disambiguator(
        noun_lexicon_path=args.noun_lexicon,
        verb_data_dir=args.verb_data_dir,
        bigram_model=bigram,
    )

    if args.sentence:
        results = d.disambiguate(args.sentence)
        print(f"\nSentence: {args.sentence!r}")
        print("─" * 60)
        for r in results:
            print(r)
    else:
        print(f"Hindi Disambiguator  (lexicon: {args.noun_lexicon})")
        if bigram:
            print(f"Bigram model loaded  (treebank: {args.treebank})")
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