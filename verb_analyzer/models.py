"""
models.py — Morphosyntactic data classes
=========================================
VerbFeatures  — feature bundle (aspect, tense, mood, person, number, gender, case)
VerbAnalysis  — one complete morphological reading of a surface form
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
        """Return a new VerbFeatures with fields from *other* overriding *self*."""
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
    confidence: str = "low"

    def __str__(self) -> str:
        tag = "[IRREG] " if self.irregular else ""
        aux = f"  compound({self.auxiliary})" if self.auxiliary else ""
        return (f"{tag}lemma={self.lemma!r:<12} "
                f"suf={self.suffix!r:<10} "
                f"{self.features}{aux}")
