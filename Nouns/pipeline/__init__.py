"""
Nouns/pipeline/__init__.py
===========================
Makes the pipeline directory a proper package so that inter-script imports
work regardless of the working directory the caller uses.

Internal dependency order (each module only imports from modules above it):
    conflict_finder       — reads the two raw MorpHIN lexicon files
    conflict_resolver     — imports conflict_finder; merges + resolves conflicts
    excluded              — imports conflict_resolver; analyses ex_noun endings
    excluded_classifier   — imports conflict_resolver + analyzer; heuristic classification
    paradigm_class_mapper — imports conflict_resolver; maps to formal M1–F5 system
    extract_noun_classes  — standalone; reads newer_LEXICON directly
    classify_nouns        — standalone; reads new_Prop_Lexicon directly

Usage
-----
Previously each script assumed it was run from inside Nouns/pipeline/ and
used bare sibling imports like:

    from conflict_resolver import merged

After packaging, use:

    from Nouns.pipeline.conflict_resolver import merged

Or if running a script directly from anywhere, the package __init__ ensures
sys.path is set correctly so sibling imports still resolve:

    cd project_root
    python -m Nouns.pipeline.paradigm_class_mapper
"""

import sys
import os

# Insert the pipeline directory itself onto sys.path so that intra-package
# scripts that use bare imports (e.g. `from conflict_resolver import merged`)
# continue to work when the package is imported from any working directory.
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

# Also insert the Nouns/ parent so that pipeline scripts can reach
# noun_paradigm_templates and the noun analyzer without path games.
_NOUNS_DIR = os.path.dirname(_PIPELINE_DIR)
if _NOUNS_DIR not in sys.path:
    sys.path.insert(0, _NOUNS_DIR)