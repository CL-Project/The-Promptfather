"""
verb_analyzer — Hindi Verb Morphological Analyzer
==================================================
Backed by MorpHIN data files.

Quick start
-----------
    from verb_analyzer import VerbAnalyzer

    va = VerbAnalyzer(data_dir='path/to/morphin_files')
    for analysis in va.analyze('खाया'):
        print(analysis)

    print(va.summarize('खाएगा'))
    results = va.analyze_batch(['खाया', 'खाएगा', 'खाता'])

Module layout
-------------
    models.py          VerbFeatures, VerbAnalysis dataclasses
    loaders.py         MorpHIN file-format parsers
    feature_tables.py  Static linguistic lookup tables and frozensets
    features.py        extract_features(), _infer_irregular_features()
    analyzer.py        VerbAnalyzer class, make_verb_analyzer() factory
    __main__.py        CLI demo  (python -m verb_analyzer <data_dir>)
"""

from .models   import VerbFeatures, VerbAnalysis
from .analyzer import VerbAnalyzer, make_verb_analyzer

__all__ = [
    "VerbFeatures",
    "VerbAnalysis",
    "VerbAnalyzer",
    "make_verb_analyzer",
]
