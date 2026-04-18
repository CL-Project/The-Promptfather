"""
analyzer.py — VerbAnalyzer: the stateful analysis engine
==========================================================
Wires together loaders, feature tables, and feature extraction into a single
public object.  This is the only module that holds in-memory state.

Public API
----------
    va = VerbAnalyzer(data_dir='path/to/morphin_files')
    analyses = va.analyze('खाया')          # → list[VerbAnalysis]
    print(va.summarize('खाएगा'))           # human-readable
    results  = va.analyze_batch(word_list) # → dict[str, list[VerbAnalysis]]

    # Drop-in factory for an existing analyzer.py:
    from verb_analyzer.analyzer import make_verb_analyzer
    va = make_verb_analyzer(DATA_DIR)
"""

from __future__ import annotations

import os
from typing import Optional
import unicodedata

from .models import VerbAnalysis, VerbFeatures
from .loaders import (
    load_irregular_mapping,
    load_suffix_rules,
    load_suffix_analysis,
    load_aux_verbs,
    load_flag_map,
)
from .features import extract_features, _infer_irregular_features


class VerbAnalyzer:
    """
    Hindi verb morphological analyzer.

    Parameters
    ----------
    data_dir : str
        Directory containing the MorpHIN data files.
    verb_lexicon : set[str] | None
        Optional set of known verb stems.  When supplied, analyses whose
        recovered lemma is NOT in the lexicon are still returned — the caller
        decides whether to filter.
    """

    # Default filenames (relative to data_dir)
    _FILES = {
        "irregular":       "IRREGULAR_VERB_MAPPING",
        "rules_main":      "newFormatSuffixReplacementRules",
        "rules_unique":    "newFormatUniqueSuffixReplacementRules",
        "suffix_analysis": "SUFFIX_ANALYSIS",
        "aux_list":        "VERB_AUXILIARY_LIST",
        "stem_flags":      "VERB_STEM_FLAG_MAP",
        "suffix_flags":    "VERB_SUFFIX_FLAG_MAP",
    }

    def __init__(
        self,
        data_dir: str = ".",
        verb_lexicon: Optional[set[str]] = None,
    ):
        d = data_dir
        p = lambda name: os.path.join(d, self._FILES[name])  # noqa: E731

        self.irregulars:      dict[str, str]             = load_irregular_mapping(p("irregular"))
        self.suffix_rules:    list[tuple[str, str, str]] = load_suffix_rules(
            p("rules_main"), p("rules_unique")
        )
        self.suffix_analysis: dict[str, list[dict]]      = load_suffix_analysis(p("suffix_analysis"))
        self.aux_verbs:       set[str]                   = load_aux_verbs(p("aux_list"))
        self.stem_flags:      dict[str, str]             = load_flag_map(p("stem_flags"),  "#verb")
        self.suffix_flags:    dict[str, str]             = load_flag_map(p("suffix_flags"), "#verb")
        self.verb_lexicon:    Optional[set[str]]         = verb_lexicon

        # Pre-sort for greedy longest-first matching
        self._aux_sorted        = sorted(self.aux_verbs,   key=len, reverse=True)
        self._stem_flags_sorted = sorted(self.stem_flags,  key=len, reverse=True)

    def _deduplicate(self, results: list[VerbAnalysis]) -> list[VerbAnalysis]:
        seen = set()
        out = []
        for r in results:
            key = (r.lemma, r.suffix, r.paradigm, str(r.features))
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out
        
    def _is_plausible_lemma(self, lemma: str, suffix: str) -> bool:
        # Reject empty or single-character lemmas
        if len(lemma) < 2:
            return False
        # If a verb stem lexicon is loaded, use it as a gate
        if self.verb_lexicon is not None and lemma not in self.verb_lexicon:
            return False
        return True

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, word: str) -> list[VerbAnalysis]:
        """
        Morphologically analyze a single Hindi word.

        Returns a list of VerbAnalysis objects.  Empty list means the word
        doesn't look like a verb according to this analyzer's resources.
        Results are ordered: irregulars first, then longest-suffix matches.
        """
        word = unicodedata.normalize('NFC', word.strip())

        word = word.strip()
        results: list[VerbAnalysis] = []
        seen:    set[tuple]         = set()   # (lemma, suffix, paradigm, features) dedup key

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
                confidence="high",
            ))
            # Don't return early — the same word might also match suffix rules
            # (rare but possible for highly regular irregulars)

        # ── Step 2: Suffix replacement rules ─────────────────────────────────
        for paradigm, suffix, add_back in self.suffix_rules:
            if not word.endswith(suffix):
                continue

            stem_base = word[: len(word) - len(suffix)]
            lemma     = stem_base + add_back     # add_back may be empty string

            if not self._is_plausible_lemma(lemma, suffix):
                continue

            if not lemma:
                continue

            # Build all feature interpretations for this suffix
            feat_list = extract_features(suffix, self.suffix_flags)

            for features in feat_list:
                key = (lemma, suffix, paradigm)
                if key in seen:
                    continue
                seen.add(key)

                aux = self._detect_aux(lemma)
                conf = "medium" if len(suffix) >= 3 else "low"
                results.append(VerbAnalysis(
                    surface=word, lemma=lemma,
                    suffix=suffix, paradigm=paradigm,
                    features=features, irregular=False,
                    auxiliary=aux[0] if aux else None,
                    aux_flag=aux[1]  if aux else None,
                    confidence=conf,
                ))

        results = self._deduplicate(results)
        results.sort(key=lambda a: {"high": 0, "medium": 1, "low": 2}[a.confidence])
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
        for morph in self._stem_flags_sorted:
            if stem == morph or stem.endswith(morph) or stem.startswith(morph):
                return (morph, self.stem_flags[morph])
        for aux in self._aux_sorted:
            if stem == aux or stem.endswith(aux):
                return (aux, "aux")
        return None


# ── Integration factory ───────────────────────────────────────────────────────

def make_verb_analyzer(data_dir: str, verb_lexicon: Optional[set[str]] = None) -> VerbAnalyzer:
    """
    Drop-in factory for an existing analyzer.py:

        from verb_analyzer.analyzer import make_verb_analyzer
        va = make_verb_analyzer(DATA_DIR)

        # In your main analyze() function:
        verb_results = va.analyze(token)
    """
    return VerbAnalyzer(data_dir=data_dir, verb_lexicon=verb_lexicon)
