# The Promptfather — Hindi Morphological Analyzer

**Team:** Pranjal Baranwal, Apoorv Joshi  
**Scope:** Rule-based paradigm morphological analyzer for Hindi nouns and verbs, built on top of MorpHIN (IIT Bombay) lexicon and rule data.

---

## Overview

This project implements a rule-based morphological analyzer for Hindi that takes an inflected surface form and returns its lemma along with a full grammatical feature bundle — gender, number, case (for nouns) and tense, aspect, mood, person, number, gender (for verbs).

The system is split into three top-level components:

- **Noun Analyzer** (`Nouns/`) — covers 9 paradigm classes (M1–M4, F1–F5), backed by a lexicon of ~24,600 classified noun stems. Includes a pipeline for constructing the lexicon from raw MorpHIN data, an ML-based gender predictor for ambiguous words, and an HDTB evaluator.
- **Verb Analyzer** (`verb_analyzer/`) — covers finite, participle, infinitive, and imperative verb forms using MorpHIN suffix-replacement rules and an irregular verb mapping.
- **Disambiguator** (`disambiguator.py`) — context-sensitive scorer that combines both analyzers to select the best reading for each token in a full sentence.

---

## Repository Structure

```
THE-PROMPTFATHER/
│
├── Nouns/                              # Noun analyzer package
│   ├── data/                           # Generated lexicon files (pipeline output)
│   │   ├── conflict_log.txt            # 147 unresolved cross-file class conflicts
│   │   ├── eval_errors.tsv             # Full HDTB evaluation error log (TSV)
│   │   ├── noun_lexicon_abstain.tsv    # Words where gender predictor abstained (p < 0.55)
│   │   ├── noun_lexicon_complete.tsv   # Final lexicon: certain + heuristic + high/medium predictions
│   │   ├── noun_lexicon_expanded.tsv   # Certain + heuristic vowel-final additions (~24,613 entries)
│   │   ├── noun_lexicon_final.tsv      # Purely certain classifications from MorpHIN (~20,953 entries)
│   │   ├── noun_lexicon_merged.tsv     # Raw merged lexicon before paradigm-class mapping
│   │   ├── noun_lexicon_predicted.tsv  # High + medium confidence gender predictions only
│   │   └── noun_lexicon_review.tsv     # Low-confidence predictions (p < 0.65); needs manual review
│   │
│   ├── pipeline/                       # Lexicon construction scripts (run once, in order)
│   │   ├── __init__.py                 # Package init; sets up sys.path for sibling imports
│   │   ├── classify_nouns.py           # Step 1a: extract noun entries from new_Prop_Lexicon
│   │   ├── extract_noun_classes.py     # Step 1b: paradigm class distribution from newer_LEXICON
│   │   ├── conflict_finder.py          # Step 2: cross-lexicon conflict detection and merge
│   │   ├── conflict_resolver.py        # Step 3: conflict resolution rules + saves merged TSV
│   │   ├── paradigm_class_mapper.py    # Step 4: maps MorpHIN exemplar labels → M1–F5
│   │   ├── excluded.py                 # Step 5a: character distribution of ex_noun words
│   │   └── excluded_classifier.py      # Step 5b: heuristic vowel-final classification → expanded TSV
│   │
│   ├── __main__.py                     # CLI entry point for the noun analyzer (REPL + single-word)
│   ├── analyzer.py                     # Core analysis engine: load_lexicon(), analyze()
│   ├── evaluate.py                     # HDTB evaluation script for the noun analyzer
│   ├── gender_predictor.py             # ML gender classifier for consonant-final ex_noun words
│   └── noun_paradigm_templates.py      # Inflectional rule tables for all 9 paradigm classes
│
├── verb_analyzer/                      # Verb analyzer package
│   ├── __init__.py                     # Public API: VerbAnalyzer, VerbAnalysis, VerbFeatures
│   ├── __main__.py                     # CLI entry point (REPL + single-word)
│   ├── analyzer.py                     # VerbAnalyzer class: wires loaders + features together
│   ├── evaluate.py                     # HDTB evaluation script for the verb analyzer
│   ├── feature_tables.py               # Static linguistic lookup tables (suffix sets, GN maps)
│   ├── features.py                     # extract_features() and _infer_irregular_features()
│   ├── loaders.py                      # MorpHIN file parsers (one function per file format)
│   └── models.py                       # VerbFeatures and VerbAnalysis dataclasses
│
├── disambiguator.py                    # Context-based sentence-level disambiguator
├── Proposal.pdf                        # Project proposal document
├── .gitignore
└── README.md
```

---

## Data Sources

| Source | Used For |
|---|---|
| Hindi WordNet v1.5 (IIT Bombay CFILT) | Primary lemma source; MorpHIN lexicon and rules |
| HDTB (Hindi Dependency Treebank) | Gold-standard evaluation for both components |

---

## Noun Analyzer

### Paradigm Classes

| Class | Pattern | Example | Count (certain) |
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

### Lexicon Files

| File | Size | Description |
|---|---|---|
| `noun_lexicon_final.tsv` | 20,953 nouns | Classified with certainty from MorpHIN |
| `noun_lexicon_expanded.tsv` | 24,613 nouns | + heuristic vowel-final additions |
| `noun_lexicon_complete.tsv` | 24,613+ nouns | + high/medium gender predictions |
| `noun_lexicon_merged.tsv` | ~33,000 entries | Raw merged lexicon before class mapping |
| `noun_lexicon_predicted.tsv` | varies | High + medium confidence ML predictions only |
| `noun_lexicon_review.tsv` | varies | Low-confidence predictions needing human review |
| `noun_lexicon_abstain.tsv` | varies | Words the model abstained on (p < 0.55) |
| `conflict_log.txt` | 147 entries | Unresolved cross-file conflicts |

### Module Descriptions

#### `noun_paradigm_templates.py`
Defines `PARADIGM_TABLES` — a dictionary mapping each paradigm class (`M1`–`F5`) to a list of inflection rules. Each rule is a 5-tuple:

```
(suffix_to_strip, suffix_to_add_for_lemma, gender, number, case)
```

For example, the M1 class strips `-े` and adds `-ा` to recover the lemma, tagging the form as masculine plural direct or masculine singular oblique (syncretism). This file is the single source of truth for inflectional coverage; adding or fixing a rule requires only editing this file.

#### `analyzer.py`
The core analysis engine. Contains three public functions:

- **`normalize_hindi(text)`** — applies NFC normalization then strips nukta from ड़ → ड and ढ़ → ढ to compensate for inconsistent MorpHIN encoding.
- **`load_lexicon(tsv_path)`** — reads a lexicon TSV and returns three dicts keyed by normalized form: `lexicon_lookup` (paradigm class), `lexicon_display` (original spelling), `lexicon_confidence` (`certain` or `heuristic`).
- **`analyze(surface_form, lex_lookup, lex_display, lex_confidence, paradigm_tables)`** — iterates every rule in every paradigm class, strips the rule's suffix from the surface form, reconstructs a candidate lemma, and checks whether that lemma exists in the lexicon with the matching class. Returns a deduplicated list of result dicts containing `lemma`, `gender`, `number`, `case`, `paradigm`, and `confidence`.

### Pipeline Module Descriptions (`Nouns/pipeline/`)

The pipeline scripts are run once, in dependency order, to build the classified lexicon from raw MorpHIN files. They assume MorpHIN data is available at `./HindiWN_1_5/MorpHIN/Lexicon/` relative to the `Nouns/` directory.

| Script | Input | Output | Purpose |
|---|---|---|---|
| `classify_nouns.py` | `new_Prop_Lexicon` | (stdout) | Extracts noun entries; prints sample |
| `extract_noun_classes.py` | `newer_LEXICON` | (stdout) | Prints paradigm class distribution |
| `conflict_finder.py` | both lexicons | (stdout + module exports) | Detects words with conflicting classes across the two files |
| `conflict_resolver.py` | `conflict_finder` exports | `noun_lexicon_merged.tsv`, `conflict_log.txt` | Applies resolution rules: prefer specific over `ex_noun`, prefer `newer_LEXICON` on رات/घر conflicts, log the rest |
| `paradigm_class_mapper.py` | `noun_lexicon_merged.tsv` | `noun_lexicon_final.tsv` | Translates MorpHIN exemplar labels (e.g. `लड़का`) to the formal M1–F5 system; fixes typos/encoding errors |
| `excluded.py` | `conflict_resolver` exports | (stdout) | Analyses the character distribution of `ex_noun` words |
| `excluded_classifier.py` | `noun_lexicon_final.tsv` | `noun_lexicon_expanded.tsv` | Heuristically classifies vowel-final `ex_noun` words by ending character; marks them `confidence=heuristic` |

---

### Usage — Noun Analyzer CLI

```bash
# Interactive REPL (type any noun, 'q' to quit)
python -m Nouns

# Single word
python -m Nouns लड़कों
```

```python
from Nouns.analyzer import load_lexicon, analyze
from Nouns.noun_paradigm_templates import PARADIGM_TABLES

lex_lookup, lex_display, lex_confidence = load_lexicon('Nouns/data/noun_lexicon_expanded.tsv')
results = analyze('लड़कियों', lex_lookup, lex_display, lex_confidence, PARADIGM_TABLES)
# → [{'lemma': 'लड़की', 'gender': 'F', 'number': 'PL', 'case': 'OBL',
#      'paradigm': 'F1', 'confidence': 'certain'}]
```

---

### Usage — Noun Evaluator

Evaluates the noun analyzer against HDTB CoNLL `.dat` files. Reports coverage, lemma accuracy, full-match accuracy, breakdown by paradigm class, breakdown by confidence tier, failure type classification, and a stratified error sample.

```bash
# Evaluate against a directory of HDTB files
python -m Nouns.evaluate path/to/HDTB/dir/

# Single file with error log
python -m Nouns.evaluate path/to/file.dat --errors Nouns/data/eval_errors.tsv

# Compare two lexicons
python -m Nouns.evaluate path/to/dir/ \
    --lexicon  Nouns/data/noun_lexicon_expanded.tsv \
    --lexicon2 Nouns/data/noun_lexicon_complete.tsv
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--lexicon` | `noun_lexicon_expanded.tsv` | Primary lexicon |
| `--lexicon2` | None | Optional second lexicon for side-by-side comparison |
| `--errors PATH` | None | Save full error list as TSV |
| `--max-errors N` | 200 | Max entries in printed stratified sample |
| `--seed N` | 42 | Random seed for sampling |

**Evaluation metrics:**
- **Coverage** — any analysis returned at all
- **Lemma accuracy** — any returned analysis has the correct lemma (oracle)
- **Correct lemma, wrong features** — right lemma but wrong gender/number/case
- **Full match accuracy** — correct lemma + gender + number + case (oracle; dimensions where gold is `any` or unannotated are skipped)

**Failure types:**
- `no_analysis` — no suffix rule fired (OOV form)
- `oov` — gold lemma not in lexicon (data gap, not a rule error)
- `wrong_lemma` — wrong lemma recovered (suffix rule or lexicon class error)
- `wrong_features` — correct lemma, wrong feature bundle (paradigm table error)

---

### Usage — Gender Predictor

Trains a character n-gram logistic regression classifier on M2/F3 consonant-final nouns and applies it to the 4,491 excluded consonant-final `ex_noun` words. Run from the `Nouns/` directory.

```bash
cd Nouns/
python gender_predictor.py
```

**Workflow:**
1. Loads training data from `noun_lexicon_final.tsv` (M2 = masculine, F3 = feminine consonant-final nouns).
2. Extracts reversed grapheme-cluster strings as features and trains a character n-gram logistic regression (`max_ngram=4`, `C=1.0`) over 5-fold cross-validation.
3. Predicts gender for all consonant-final `ex_noun` words not already in `noun_lexicon_expanded.tsv`.
4. Splits predictions into four tiers:

| Tier | Threshold | Output |
|---|---|---|
| `predicted_high` | p ≥ 0.80 | Written to `noun_lexicon_predicted.tsv` and merged into `noun_lexicon_complete.tsv` |
| `predicted_medium` | p ≥ 0.65 | Same as above |
| `predicted_low` | p ≥ 0.55 | Written to `noun_lexicon_review.tsv` only; **not** merged |
| `abstain` | p < 0.55 | Written to `noun_lexicon_abstain.tsv`; no gender assigned |

> Cross-validation accuracy is measured on held-out M2/F3 words (already well-behaved). The excluded words are systematically harder (mostly loanwords). True accuracy on the target set is lower than reported CV accuracy. Treat all predictions as hypotheses for human review.

**Dependencies:** `scikit-learn`, `numpy`

```bash
pip install scikit-learn numpy
```

---

## Verb Analyzer

### Module Descriptions (`verb_analyzer/`)

#### `models.py`
Defines two dataclasses:
- **`VerbFeatures`** — the full morphosyntactic feature bundle: `aspect`, `tense`, `mood`, `verbal_type`, `person`, `number`, `gender`, `case`. All fields are `Optional[str]` and default to `None`.
- **`VerbAnalysis`** — one complete reading: `surface`, `lemma`, `suffix`, `paradigm`, `features`, `irregular` flag, optional `auxiliary` and `aux_flag`, and a `confidence` level (`high` / `medium` / `low`).

#### `loaders.py`
Stateless file parsers, one function per MorpHIN format:
- **`load_irregular_mapping`** — reads `IRREGULAR_VERB_MAPPING` into `{surface: lemma}` (e.g. `गया → जा`).
- **`load_suffix_rules`** — reads one or more `newFormat*` rule files into a sorted list of `(paradigm, suffix, add_back)` tuples, ordered longest-suffix-first for greedy matching. Filters out invalid `add_back` values (ै, ो).
- **`load_suffix_analysis`** — reads `SUFFIX_ANALYSIS` into `{morpheme: [feature_dict, …]}`.
- **`load_aux_verbs`** — reads `VERB_AUXILIARY_LIST` into a set of auxiliary/vector verb roots.
- **`load_flag_map`** — reads `VERB_STEM_FLAG_MAP` / `VERB_SUFFIX_FLAG_MAP` into `{morpheme: flag}`.

#### `feature_tables.py`
All read-only linguistic lookup tables consumed by `features.py`. Editing linguistic coverage (adding suffixes, fixing gender assignment) only requires touching this file:
- `_TERMINAL_GN` — suffix ending → gender/number mapping for perfective/imperfective forms.
- `_FUTURE_GN` — gender/number from future suffix endings.
- `_PERSON_PREFIXES` — suffix substring → person string.
- `_FUTURE_SUFFIXES`, `_IMPERFECTIVE_SUFFIXES`, `_PERFECTIVE_SUFFIXES`, `_IMPERATIVE_SUFFIXES`, `_SUBJUNCTIVE_SUFFIXES` — frozensets for aspect/mood assignment.
- `_INFINITIVE_MAP` — unambiguous infinitive suffixes (ना/ने/नी) → full feature bundle.
- `_AMBIGUOUS_SUFFIXES` — suffixes with multiple legitimate readings (े, ए, ें).
- `_FLAG_TO_FEATURE` — MorpHIN flag → partial feature dict.

#### `features.py`
Pure functions; no I/O or state.
- **`extract_features(suffix, suffix_flag_map)`** — returns a list of `VerbFeatures` for a given suffix. Priority order: (1) infinitive lookup; (2) explicit ambiguous-suffix table; (3) full-suffix frozenset membership (prevents sub-morpheme mis-tagging of compound suffixes); (4) morpheme flag scan; (5) terminal morpheme scan for gender/number.
- **`_infer_irregular_features(word)`** — pattern-matches an irregular surface form (future endings, imperative endings, perfective endings) to recover a feature bundle.

#### `analyzer.py`
The `VerbAnalyzer` class wires everything together. Constructor loads all data files from a `data_dir`. The `analyze(word)` method:
1. Checks the irregular mapping first.
2. Iterates suffix rules longest-first; reconstructs candidate lemma; applies plausibility check (length ≥ 2; optional lexicon gate).
3. Calls `extract_features()` for each matching suffix.
4. Deduplicates on `(lemma, suffix, paradigm)`.
5. Sorts results: `high` (irregular) → `medium` (suffix ≥ 3 chars) → `low`.

Also provides `analyze_batch(words)` and `summarize(word)`.

### Usage — Verb Analyzer CLI

```bash
# Interactive REPL
python -m verb_analyzer path/to/morphin_files/

# Single word
python -m verb_analyzer path/to/morphin_files/ खाया
```

```python
from verb_analyzer import VerbAnalyzer

va = VerbAnalyzer(data_dir='path/to/morphin_files')
for analysis in va.analyze('खाया'):
    print(analysis)
# → lemma='खा'   suf='या'   aspect:perfective, gender:M, number:S, verbal_type:participle

print(va.summarize('खाएगा'))
results = va.analyze_batch(['खाया', 'खाएगा', 'खाता'])
```

### Feature Bundle

| Field | Values |
|---|---|
| `aspect` | `imperfective` \| `perfective` |
| `tense` | `future` \| `past` \| `present` |
| `mood` | `imperative` \| `subjunctive` \| `conditional` |
| `verbal_type` | `finite` \| `participle` \| `infinitive` \| `conjunctive` |
| `person` | `1` \| `2` \| `3` |
| `number` | `S` \| `P` |
| `gender` | `M` \| `F` \| `N` |
| `case` | `D` \| `O` |

---

### Usage — Verb Evaluator

Evaluates the verb analyzer against HDTB verbal tokens (CPOS `VM` and `VAUX`). Tense/aspect/mood are decoded from the `tam` field in HDTB feature strings rather than explicit `asp`/`ten`/`mood` keys, which are not present on verbal tokens in HDTB.

```bash
# Evaluate against a directory
python -m verb_analyzer.evaluate path/to/morphin/ path/to/HDTB/dir/

# Single file with error log
python -m verb_analyzer.evaluate . path/to/file.dat --errors errors.tsv

# Save stratified error sample as readable text
python -m verb_analyzer.evaluate . path/to/HDTB/ --sample sample.txt
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--errors PATH` | None | Save full error list as TSV |
| `--sample PATH` | None | Save stratified error sample as text |
| `--max-errors N` | 200 | Max entries in printed sample |
| `--seed N` | 42 | Random seed |

Metrics mirror the noun evaluator: coverage, lemma accuracy, full match, breakdown by verbal type / confidence tier / CPOS, failure type breakdown, and a stratified error sample.

---

## Disambiguator

`disambiguator.py` provides context-based disambiguation for full Hindi sentences by scoring all candidate analyses from both the noun and verb analyzers using surrounding context signals.

### Module Contents

#### `BigramModel`
A noun paradigm bigram model trained from HDTB treebank data. Models P(paradigm_j | paradigm_i) over consecutive NN token pairs. Uses additive smoothing (α = 0.1). Contributes up to +2 weight per candidate in the scorer. Train with:

```python
bigram = BigramModel.from_treebank(path='path/to/hdtb/', lexicon_lookup=lex_lookup)
```

#### `Disambiguator`
The main class. Accepts a noun lexicon path, verb data directory, and optional `gender_model` and `bigram_model`.

**Disambiguation pipeline (called by `.disambiguate(sentence)`):**

1. **Tokenize** — whitespace split, NFC normalize.
2. **Analyze** — run noun and verb analyzers on every token.
3. **Build signal map** — two-pass context extraction:
   - Pass 1: compound postpositions (longest-match-first); governed noun flagged oblique.
   - Pass 2: single-token signals — oblique postpositions, genitive agreement (का/की/के → gender/number on right neighbour), quantifiers (singular/plural on right neighbour), negation (नहीं/मत → mood constraint on verbs), per-position verb agreement.
4. **Score noun candidates** using all signals + bigram prior:

   | Signal | Weight |
   |---|---|
   | Genitive gender match | +8 |
   | Genitive gender mismatch | −8 |
   | Oblique case match | +4 |
   | Oblique case mismatch | −4 |
   | Number match (quantifier) | +2 |
   | Nearest verb agreement | +1 |
   | Bigram prior | 0–2 |

5. **Filter verb candidates** — negation mood constraint (नहीं → exclude imperatives; मत → keep only imperatives).
6. **Gender model** — for OOV tokens with no noun candidates, a pluggable `GenderModel` can synthesize candidates.
7. **Coordination post-processing** — propagates resolved case across `और`-joined noun phrases.

Returns a list of `TokenResult` objects with `noun_candidates`, `verb_candidates`, `ambiguous` flag, and `signals_applied` list.

### Usage — Disambiguator

```python
from disambiguator import Disambiguator, BigramModel
from Nouns.analyzer import load_lexicon

lex_lookup, _, _ = load_lexicon('Nouns/data/noun_lexicon_expanded.tsv')

# Optional: train bigram model
bigram = BigramModel.from_treebank('path/to/hdtb/', lexicon_lookup=lex_lookup)

d = Disambiguator(
    noun_lexicon_path='Nouns/data/noun_lexicon_expanded.tsv',
    verb_data_dir='path/to/morphin_files',
    bigram_model=bigram,   # optional
)

results = d.disambiguate('लड़की ने खाना खाया')
for tok in results:
    print(tok)
```

**CLI:**

```bash
# Interactive REPL (no bigram model)
python disambiguator.py Nouns/data/noun_lexicon_expanded.tsv path/to/morphin/

# With bigram model trained on HDTB
python disambiguator.py Nouns/data/noun_lexicon_expanded.tsv path/to/morphin/ \
    --treebank path/to/hdtb/

# Single sentence
python disambiguator.py Nouns/data/noun_lexicon_expanded.tsv path/to/morphin/ \
    --sentence 'लड़की ने खाना खाया'
```

---

## Dependencies

- Python 3.8+
- Standard library only for the core analyzers (`unicodedata`, `csv`, `collections`, `os`, `sys`)
- `scikit-learn`, `numpy` — required only for `gender_predictor.py`
- MorpHIN data files from Hindi WordNet v1.5 (IIT Bombay CFILT)

```bash
pip install scikit-learn numpy   # only needed for gender_predictor.py
```
