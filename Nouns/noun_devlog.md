# Hindi Morphological Analyzer — Development Log
**Project:** Rule-Based Paradigm Morphological Analyzer for Hindi  
**Team:** The Promptfather (Pranjal Baranwal, Apoorv Joshi)  
**Scope:** Noun analyzer (verbs deferred to future work)

---

## Overview

This document records the complete development process for the noun component of a rule-based Hindi morphological analyzer. It covers data acquisition, lexicon construction, paradigm table design, analyzer implementation, and known limitations.

---

## Phase 1: Data Acquisition

### What We Did

Downloaded two datasets:

1. **Hindi WordNet v1.4** from IIT Bombay (CFILT Lab) — primary source for lemmas
2. **Hindi Universal Dependencies Treebank** — reserved for evaluation (Week 5)

### WordNet Directory Structure

After extraction, the WordNet contained:

```
database/       — lexical data files (Princeton WordNet format)
MorpHIN/        — morphological processor built by IIT Bombay
config/         — browser configuration
HWNBrowser.jar  — GUI browser (not used programmatically)
```

### Key Discovery: MorpHIN

The `MorpHIN/` directory contains a fully built morphological processor for Hindi, including:

- `MorpHIN/Lexicon/` — pre-classified noun and verb stems
- `MorpHIN/Rules/` — inflection rules

This significantly changed our plan. Rather than building paradigm class assignments from scratch using the raw WordNet, we could use MorpHIN's lexicon as our primary source, since it already classifies words into paradigm classes.

---

## Phase 2: Lexicon Construction

### Step 1: Initial Extraction from WordNet Database

The `database/` folder uses Princeton WordNet format. The index files (`idxnoun_txt`, `idxverb_txt`) list one lemma per line, with the lemma as the first whitespace-separated token.

**Extraction script logic:**
- Read each line, take the first token as the lemma
- Filter to entries containing at least one Devanagari character (Unicode block U+0900–U+097F)
- Replace underscores with spaces (WordNet uses underscores for multi-word expressions)

**Initial counts:**
- Nouns: ~60,000
- Verbs: ~6,778

**Problem:** Counts were far larger than expected (proposal estimated 10,000–15,000).

**Cause:** The WordNet contains compound expressions, proper nouns, numeric ordinals (100वाँ etc.), and domain-specific vocabulary beyond everyday usage.

**Fix:** Added filters for multi-word expressions (any lemma containing a space after underscore replacement), digits mixed with Devanagari, and minimum character length of 2.

**Post-filter counts:**
- Nouns: ~57,168 (barely changed — most entries were already single-word Devanagari)
- Verbs: ~2,500 (dropped significantly — many verb entries were phrasal verbs)

**Decision:** The WordNet noun list was too large and lacked gender information, which is essential for paradigm class assignment. Switched to MorpHIN lexicon as primary source.

---

### Step 2: Parsing the MorpHIN Lexicon

The MorpHIN lexicon (`new_prop_lexicon` and `newer_LEXICON`) uses an exemplar-based paradigm system. Each line follows the format:

```
<word>,<paradigm_class>,<pos>
```

Where the paradigm class is named after a representative Hindi word (e.g., `लड़का`, `रात`, `घर`). Some lines have leading slashes (`//` or `////`) which are stripped before parsing.

**Two lexicon files found:**

| File | Noun count |
|---|---|
| `new_prop_lexicon` | 29,206 |
| `newer_LEXICON` | 29,152 |

**Overlap analysis:**
- Words in both: 29,090
- Words only in `new_prop_lexicon`: 116
- Words only in `newer_LEXICON`: 62
- Conflicting class assignments: 4,532

---

### Step 3: Resolving Conflicts Between the Two Files

**Conflict breakdown:**
- `रात vs घर` conflicts: 4,034 (~89% of all conflicts)
- Other conflicts: 498

**Why so many `रात vs घर` conflicts?**  
Both classes are consonant-final nouns. The only difference between them is grammatical gender — `घर` is masculine, `रात` is feminine. Gender of consonant-final nouns cannot be reliably predicted from phonological shape alone, so the two annotation passes disagreed on approximately 4,000 words.

**Resolution strategy:**
1. If one file assigns `ex_noun` and the other assigns a specific class → take the specific class
2. For `रात vs घर` conflicts → prefer `newer_LEXICON` (the more recent annotation)
3. For all other conflicts → prefer `newer_LEXICON` as default, log the conflict

**Remaining unresolved conflicts after strategy:** 147 (logged to `conflict_log.txt`)

**Merged lexicon size:** 29,268 nouns

---

### Step 4: Mapping MorpHIN Classes to Formal Paradigm System

MorpHIN uses exemplar word labels. We mapped these onto a formal M1–F5 system based on gender and phonological shape:

| MorpHIN Exemplar | Formal Class | Description |
|---|---|---|
| लड़का, राजा, विधाता, लोहा | M1 | Masculine -ā final |
| घर, क्रोध, अपनापन, खर्च | M2 | Masculine consonant-final, invariant |
| आदमी, कवि, पानी | M3 | Masculine -ī/-i final |
| आलू, शत्रु, लहू, बालू | M4 | Masculine -ū/-u final |
| लड़की, गुड़िया | F1 | Feminine -ī final |
| आशा, ईर्ष्या | F2 | Feminine -ā final |
| रात, भीड़ | F3 | Feminine consonant-final |
| शान्ति, आपत्ति | F4 | Feminine -i final |
| वायु, ऋतु, बहू | F5 | Feminine -u/-ū final |

**Note:** The proposal originally defined 6 classes (M1–M3, F1–F3). The actual data revealed 9 classes are needed — specifically the -u/-ū final classes (M4, F5) and the feminine -i final class (F4) were not accounted for in the original proposal.

**Typo and encoding fixes applied before mapping:**

Several MorpHIN class labels were corrupted:

| Corrupted label | Correct label |
|---|---|
| `<क्रोध` | `क्रोध` |
| `ेex_noun` | `ex_noun` |
| `गुिड़या` | `गुड़िया` |
| `लडकी` | `लड़की` |
| `अाशा` | `आशा` |
| `शािन्त` | `शान्ति` |
| `भीडञ`, `भिड़` | `भीड़` |
| `रातन` | `रात` |

**Post-mapping classified lexicon:** 20,953 nouns  
**Skipped (ex_noun, proper nouns, irregular):** 8,315 nouns

**Class distribution:**

| Class | Count |
|---|---|
| M1 | 1,846 |
| M2 | 7,990 |
| M3 | 807 |
| M4 | 196 |
| F1 | 2,459 |
| F2 | 1,834 |
| F3 | 4,931 |
| F4 | 790 |
| F5 | 100 |

M2 and F3 being the largest classes reflects the high frequency of consonant-final nouns in Hindi, which absorbed many Sanskrit, Arabic, and Persian loanwords.

---

### Step 5: Heuristic Classification of ex_noun Words

The 8,315 skipped words were analyzed by their final character:

| Ending | Count | Assigned Class |
|---|---|---|
| ा | 1,977 | M1 (high confidence) |
| ी | 1,454 | F1 (high confidence) |
| ि | 233 | F4 (high confidence) |
| ई | 136 | F1 variant (high confidence) |
| Consonant-final | ~4,491 | Unresolvable — gender unknown |

The ~3,800 vowel-final words were added to the lexicon with a `heuristic` confidence flag. The ~4,491 consonant-final words were excluded — their gender cannot be determined without additional annotation.

**Final expanded lexicon:** 24,613 nouns (saved to `noun_lexicon_expanded.tsv`)

---

## Phase 3: Paradigm Tables

### Design

Each paradigm class defines four inflectional rules, one per cell in the number × case grid (SG/PL × DIR/OBL). Each rule is a tuple:

```
(surface_suffix, lemma_suffix, gender, number, case)
```

The analyzer strips `surface_suffix` from the input, adds `lemma_suffix` to recover the lemma, and returns the feature bundle.

### Complete Paradigm Tables

**M1 — Masculine -ā final (लड़का)**

| Form | Surface | Rule |
|---|---|---|
| SG DIR | लड़का | strip ा, add ा |
| PL DIR | लड़के | strip े, add ा |
| SG OBL | लड़के | strip े, add ा |
| PL OBL | लड़कों | strip ों, add ा |

**M2 — Masculine consonant-final invariant (घर)**

| Form | Surface | Rule |
|---|---|---|
| SG DIR | घर | no change |
| PL DIR | घर | no change |
| SG OBL | घर | no change |
| PL OBL | घरों | add ों |

**M3 — Masculine -ī/-i final (आदमी)**

| Form | Surface | Rule |
|---|---|---|
| SG DIR | आदमी | no change |
| PL DIR | आदमी | no change |
| SG OBL | आदमी | no change |
| PL OBL | आदमियों | strip ियों, add ी |

**M4 — Masculine -ū/-u final (आलू)**

| Form | Surface | Rule |
|---|---|---|
| SG DIR | आलू | no change |
| PL DIR | आलू | no change |
| SG OBL | आलू | no change |
| PL OBL | आलुओं | strip ुओं, add ू |

**F1 — Feminine -ī final (लड़की)**

| Form | Surface | Rule |
|---|---|---|
| SG DIR | लड़की | no change |
| PL DIR | लड़कियाँ | strip ियाँ, add ी |
| SG OBL | लड़की | no change |
| PL OBL | लड़कियों | strip ियों, add ी |

**F2 — Feminine -ā final (आशा)**

| Form | Surface | Rule |
|---|---|---|
| SG DIR | आशा | no change |
| PL DIR | आशाएँ | strip ाएँ, add ा |
| SG OBL | आशा | no change |
| PL OBL | आशाओं | strip ाओं, add ा |

**F3 — Feminine consonant-final (रात)**

| Form | Surface | Rule |
|---|---|---|
| SG DIR | रात | no change |
| PL DIR | रातें | add ें |
| SG OBL | रात | no change |
| PL OBL | रातों | add ों |

**F4 — Feminine -i final (शान्ति)**

| Form | Surface | Rule |
|---|---|---|
| SG DIR | शान्ति | no change |
| PL DIR | शान्तियाँ | strip ियाँ, add ि |
| SG OBL | शान्ति | no change |
| PL OBL | शान्तियों | strip ियों, add ि |

**F5 — Feminine -u/-ū final (ऋतु)**

| Form | Surface | Rule |
|---|---|---|
| SG DIR | ऋतु | no change |
| PL DIR | ऋतुएँ | strip ुएँ, add ु |
| SG OBL | ऋतु | no change |
| PL OBL | ऋतुओं | strip ुओं, add ु |

---

## Phase 4: Analyzer Implementation

### Core Logic

The analyzer takes a surface form, iterates over all paradigm table rules, and for each rule:

1. Checks if the surface form ends in the rule's surface suffix
2. Strips the surface suffix and adds back the lemma suffix to recover the candidate lemma
3. Looks up the candidate lemma in the lexicon
4. If found with a matching paradigm class, returns the analysis

Syncretic forms (where one surface form maps to multiple feature bundles) correctly return multiple analyses.

### Unicode Normalization

**Problem 1 — NFC normalization:** Hindi Devanagari characters can be encoded as precomposed characters or as base character + combining mark. Two strings that look identical may fail string equality if one uses NFC and the other does not. Applied `unicodedata.normalize('NFC', text)` to all text at load time and query time.

**Problem 2 — Missing nuktas in MorpHIN lexicon:** The MorpHIN source data inconsistently encodes nukta-bearing characters. For example, `खिड़की` (window) was stored as `खिडकी` (without the nukta under ड). NFC normalization cannot fix a missing nukta — it can only reorder combining characters that are already present.

**Fix:** Strip nuktas from both lexicon keys and input before lookup, then use a separate `lexicon_display` dictionary that maps stripped keys back to original spellings for output. This means the analyzer displays correctly spelled lemmas while still matching inconsistently encoded entries.

**Residual issue:** Some lexicon entries (like `खिडकी`) are stored without nukta in the source MorpHIN file and therefore display without nukta in output. These cannot be auto-corrected without manually reviewing all affected entries.

---

## Known Limitations and Open Problems

### 1. Consonant-Final Nouns Without Gender (4,491 words)
These words could not be assigned a paradigm class because their gender is unknown and cannot be inferred from phonological shape. They are excluded from the lexicon. Resolving this would require either manual annotation or a trained gender prediction model.

### 2. ex_noun Loanwords (8,291 words in MorpHIN)
Words tagged `ex_noun` in MorpHIN are primarily loanwords from English, Arabic, and Persian that do not fit regular Hindi paradigms. Many of these actually do inflect in Hindi (e.g., फ़िल्म behaves like F3), but their gender is often unspecified in the source data. These are excluded from the classified lexicon.

### 3. Inconsistent Nukta Encoding in MorpHIN Source
MorpHIN inconsistently uses nukta for characters like ड़ and ढ़. Some entries have the nukta, others do not. Our normalization strips nuktas for lookup purposes, which means the analyzer cannot distinguish ड from ड़ — a theoretically correct distinction that is practically harmless for the Hindi noun vocabulary.

### 4. Morphological Syncretism
Several inflectional slots share the same surface form. The analyzer correctly returns all valid analyses for syncretic forms, but downstream tasks that require a single analysis will need a disambiguation step (e.g., using syntactic context from the UD treebank).

Notable syncretisms:
- M1: PL DIR = SG OBL (e.g., लड़के)
- M2/F3: SG DIR = PL DIR = SG OBL (e.g., घर, रात)
- M3/F1/F4/F5: SG DIR = SG OBL

### 5. Verbs Not Implemented
The original proposal included verb morphology (tense, aspect, person, number, gender agreement). This has been deferred. The MorpHIN `Rules/` folder contains verb paradigm data that could be used as a starting point for future work.

### 6. Heuristic Classifications May Be Wrong
The ~3,800 words classified by phonological heuristic (vowel-final endings) are marked with a `heuristic` confidence flag in the lexicon. While ending-based gender assignment is accurate for the majority of Hindi nouns, exceptions exist — particularly among Sanskrit tatsama words and certain loanwords.

---

## File Inventory

| File | Description |
|---|---|
| `noun_lexicon_final.tsv` | 20,953 nouns classified from MorpHIN with certainty |
| `noun_lexicon_expanded.tsv` | 24,613 nouns including heuristic additions |
| `conflict_log.txt` | 147 unresolved cross-file classification conflicts |
| `noun_paradigm_templates.py` | Paradigm tables for all 9 noun classes |
| `analyzer.py` | Core analyzer with normalization, lookup, and analysis functions |

---

## Summary Statistics

| Metric | Value |
|---|---|
| Lexicon size (certain) | 20,953 |
| Lexicon size (with heuristics) | 24,613 |
| Paradigm classes | 9 (M1–M4, F1–F5) |
| Words excluded (no gender info) | 4,491 |
| Words excluded (ex_noun/irregular) | 8,291 |
| Unresolved annotation conflicts | 147 |
