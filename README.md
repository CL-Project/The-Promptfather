# The Promptfather — Hindi Morphological Analyzer

**Team:** Pranjal Baranwal, Apoorv Joshi  
**Scope:** Rule-based paradigm morphological analyzer for Hindi nouns and verbs, built on top of MorpHIN (IIT Bombay) lexicon and rule data.

---

## Overview

This project implements a rule-based morphological analyzer for Hindi that takes an inflected surface form and returns its lemma along with a full grammatical feature bundle — gender, number, case (for nouns) and tense, aspect, mood, person, number, gender (for verbs).

The analyzer is split into two components:

- **Noun Analyzer** — covers 9 paradigm classes (M1–M4, F1–F5), backed by a lexicon of ~24,600 classified noun stems
- **Verb Analyzer** — covers finite, participle, infinitive, and imperative forms, backed by MorpHIN suffix-replacement rules and an irregular verb mapping

---

## Repository Structure

```
.
├── Nouns/
│   ├── analyzer.py                  # Core noun analyzer
│   ├── noun_paradigm_templates.py   # Paradigm tables for all 9 noun classes
│   ├── noun_devlog.md               # Full development log for noun component
│   └── pipeline/                    # Lexicon construction pipeline
│       ├── classify_nouns.py        # Initial extraction from MorpHIN
│       ├── extract_noun_classes.py  # Paradigm class distribution analysis
│       ├── conflict_finder.py       # Cross-lexicon conflict detection
│       ├── conflict_resolver.py     # Conflict resolution + merge
│       ├── paradigm_class_mapper.py # MorpHIN exemplar → M1–F5 mapping
│       ├── excluded_classifier.py   # Heuristic classification of ex_noun words
│       ├── excluded.py              # Distribution analysis of excluded words
│       └── __init__.py
│
├── verb_analyzer/                   # Verb analyzer package
│   ├── __init__.py
│   ├── __main__.py                  # CLI (single word + interactive REPL)
│   ├── analyzer.py                  # VerbAnalyzer class
│   ├── loaders.py                   # MorpHIN file parsers
│   ├── models.py                    # VerbFeatures + VerbAnalysis dataclasses
│   ├── features.py                  # Suffix feature extraction logic
│   ├── feature_tables.py            # Static linguistic lookup tables
│   └── verb_analyzer_report.md      # Problems found and fixes applied
```

---

## Data Sources

| Source | Used For |
|---|---|
| Hindi WordNet v1.5 (IIT Bombay CFILT) | Primary lemma source; MorpHIN lexicon and rules |
| MorpHIN `new_Prop_Lexicon` | Noun stems with exemplar-based paradigm labels |
| MorpHIN `newer_LEXICON` | Second noun lexicon pass; used to resolve conflicts |
| MorpHIN suffix-replacement rules | Verb analysis rule base |
| MorpHIN `IRREGULAR_VERB_MAPPING` | Suppletive/irregular verb forms (गया, किया, लिया …) |

---

## Noun Analyzer

### Paradigm Classes

The analyzer uses a 9-class system derived from gender and phonological shape:

| Class | Pattern | Example | Count in Lexicon |
|---|---|---|---|
| M1 | Masculine -ā final | लड़का | 1,846 |
| M2 | Masculine consonant-final | घर | 7,990 |
| M3 | Masculine -ī/-i final | आदमी | 807 |
| M4 | Masculine -ū/-u final | आलू | 196 |
| F1 | Feminine -ī final | लड़की | 2,459 |
| F2 | Feminine -ā final | आशा | 1,834 |
| F3 | Feminine consonant-final | रात | 4,931 |
| F4 | Feminine -i final | शान्ति | 790 |
| F5 | Feminine -u/-ū final | ऋतु | 100 |

### Lexicon

| Lexicon | Size |
|---|---|
| Classified (certain) | 20,953 nouns |
| Expanded (with heuristic additions) | 24,613 nouns |
| Excluded — gender unknown (consonant-final) | 4,491 words |
| Excluded — loanwords / irregular | 8,291 words |
| Unresolved annotation conflicts | 147 (logged to `conflict_log.txt`) |

### Usage

```python
from Nouns.analyzer import load_lexicon, analyze_verbose
from Nouns.noun_paradigm_templates import PARADIGM_TABLES

lex_lookup, lex_display, lex_confidence = load_lexicon('Nouns/data/noun_lexicon_expanded.tsv')

# Returns list of dicts with lemma, gender, number, case, paradigm, confidence
results = analyze_verbose('लड़कियों', lex_lookup, lex_display, lex_confidence, PARADIGM_TABLES)
# → लड़की + [F1, F, PL, OBL, conf=certain]
```

### Normalization

Two-step normalization is applied at analysis time:

1. **NFC** — canonical reordering of combining characters
2. **Nukta stripping** — ड़ → ड and ढ़ → ढ for lookup, to compensate for inconsistent nukta encoding in the MorpHIN source data. Display output restores the original spelling.

---

## Verb Analyzer

### Usage

```bash
# Single word
python -m verb_analyzer <data_dir> खाया

# Interactive REPL
python -m verb_analyzer <data_dir>
```

```python
from verb_analyzer import VerbAnalyzer

va = VerbAnalyzer(data_dir='path/to/morphin_files')
for analysis in va.analyze('खाया'):
    print(analysis)
# → lemma='खा'  suf='या'  aspect:perfective, gender:M, number:S

print(va.summarize('खाएगा'))
```

### Analysis Pipeline

1. **Irregular lookup** — checks `IRREGULAR_VERB_MAPPING` first; suppletive stems (गया → जा, किया → कर) are tagged `confidence=high`
2. **Suffix replacement** — iterates suffix rules longest-first; recovers candidate lemma, applies plausibility check (minimum length), extracts feature bundle
3. **Deduplication** — final sweep removes duplicate readings
4. **Confidence scoring** — `high` (irregular), `medium` (suffix length ≥ 3), `low` (short generic suffix)

### Feature Bundle

```
aspect:      imperfective | perfective
tense:       future | past | present
mood:        imperative | subjunctive | conditional
verbal_type: finite | participle | infinitive | conjunctive
person:      1 | 2 | 3
number:      S | P
gender:      M | F | N
case:        D | O
```

### Known Issues Fixed During Development

| Problem | Fix |
|---|---|
| Unicode mismatch caused valid words to return no results | NFC normalization at entry point of `analyze()` |
| Duplicate analyses returned for the same form | Fixed dedup key logic; added final-pass sweep |
| Perfective reading missing for खाए | Introduced `_AMBIGUOUS_SUFFIXES` table for genuinely ambiguous suffixes |
| Nonsense one-character lemmas being generated | Added minimum-length plausibility check on recovered stems |
| Bad rule producing पढ़ै as a stem | Added loader-level warning for suspicious `add_back` vowels |
| No way to distinguish reliable vs speculative results | Added `confidence` field; results sorted high → low |

---

## Known Limitations

- **4,491 consonant-final nouns excluded** — gender of consonant-final nouns cannot be inferred from phonological shape alone; these require manual annotation or a trained gender classifier.
- **8,291 loanwords excluded** — words tagged `ex_noun` in MorpHIN (primarily English/Arabic/Persian borrowings) are not in the classified lexicon even though many do inflect in Hindi.
- **Morphological syncretism** — several surface forms are ambiguous across feature bundles (e.g., M1 direct plural = oblique singular; M2/F3 three slots share one form). The analyzer returns all valid readings; disambiguation requires syntactic context.
- **Nukta inconsistency** — MorpHIN inconsistently encodes ड़ / ढ़. Nukta stripping resolves lookup failures but means the analyzer cannot distinguish ड from ड़ in input.
- **Heuristic confidence** — ~3,800 vowel-final nouns were classified by ending heuristic and marked `confidence=heuristic`. Exceptions exist among Sanskrit tatsama words and loanwords.
- **Verbs not evaluated against gold standard** — UD treebank evaluation is deferred to future work.

---

## File Outputs

| File | Description |
|---|---|
| `Nouns/data/noun_lexicon_final.tsv` | 20,953 nouns classified from MorpHIN with certainty |
| `Nouns/data/noun_lexicon_expanded.tsv` | 24,613 nouns including heuristic additions |
| `Nouns/data/noun_lexicon_merged.tsv` | Raw merged lexicon before class mapping |
| `Nouns/data/conflict_log.txt` | 147 unresolved cross-file classification conflicts |

---

## Dependencies

- Python 3.8+
- Standard library only (`unicodedata`, `csv`, `collections`, `os`, `sys`)
- MorpHIN data files from Hindi WordNet v1.5 (IIT Bombay CFILT)
