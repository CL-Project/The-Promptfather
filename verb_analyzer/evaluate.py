"""
evaluate.py — HDTB Verb Evaluation
=====================================
Evaluates the verb analyzer against the Hindi Dependency Treebank (HDTB).

Mirrors the structure of Nouns/evaluate.py exactly.

HOW IT WORKS
------------
1. PARSE HDTB FILES
   Reads one or more HDTB CoNLL .dat files. Each non-blank, non-comment line
   is one token. Columns are whitespace-separated (0-indexed):

       [0]  tok_id   [1]  surface   [2]  lemma
       [3]  pos_cat  [4]  CPOS      [5]  feats   [6]  head   [7]  deprel

   Only tokens where CPOS ∈ {'VM', 'VAUX'} are kept (main verbs + auxiliaries).
   Tokens with fewer than 6 columns or lemma == '_' are counted as 'skipped'.

2. PARSE GOLD FEATURES
   The feats column uses pipe-separated key-value pairs with '-' as separator:
       cat-v|gen-m|num-sg|pers-3|case-|vib-या|tam-yA|chunkId-VGF|...

   Keys actually present on verbal tokens and how they are mapped:
       gen   →  m / f / any      →  'M' / 'F' / None
       num   →  sg / pl / any    →  'S' / 'P' / None
       pers  →  1/2/3/1h/2h/3h  →  '1' / '2' / '3'  (honorifics collapsed)
       tam   →  transliteration of the TAM suffix, decoded as follows:
                 wA   → aspect:imperfective
                 yA / yA1 / vA → aspect:perfective
                 gA   → tense:future
                 WA   → tense:past
                 hE   → tense:present
                 eM / Uz → mood:subjunctive
                 ao / aO → mood:imperative
                 nA / kara / 0 → (infinitive/conjunctive — no assertion)

   NOTE: HDTB does NOT have explicit `asp`, `ten`, or `mood` keys on verbal
   tokens. The original evaluator looked for these and always got None, making
   the full-match metric meaningless. This version decodes from `tam` instead.

3. LEMMA NORMALISATION
   Gold lemmas pass through unicodedata.normalize('NFC') before comparison,
   matching what VerbAnalyzer.analyze() applies internally.

4. ORACLE MATCHING
   A token is counted correct if ANY returned analysis matches the gold.
   This is standard for analyzers without a downstream disambiguator.

   Four match levels (same hierarchy as noun evaluator):
     - coverage:        any analysis returned at all
     - lemma_match:     any analysis has the correct lemma
     - lemma_only:      correct lemma but wrong features
     - full_match:      correct lemma + all non-None gold features agree

   Gold features that are None (unannotated / 'any') are skipped in comparison
   so the analyzer is not penalised for dimensions the treebank left blank.

5. FAILURE CLASSIFICATION
   For each full-match miss, the evaluator records WHY it failed:
     - no_analysis:    analyzer returned nothing (word not handled by any rule)
     - wrong_lemma:    analyses returned but none had the correct lemma
     - wrong_features: correct lemma recovered but feature bundle wrong

   Note: unlike the noun evaluator there is no 'oov' tier — the verb analyzer
   is lexicon-free (suffix rules + irregular mapping), so coverage is not
   gated by a fixed lexicon.

6. METRICS REPORTED
   Overall: coverage, lemma accuracy, lemma-only, full match
   By verbal_type  (finite | participle | infinitive | irregular)
   By confidence   (high | medium | low)
   By CPOS         (VM = main verb, VAUX = auxiliary)
   Failure type breakdown
   Stratified error sample

USAGE
-----
   # Evaluate against a directory of HDTB .dat files:
   python -m verb_analyzer.evaluate  <data_dir>  <hdtb_path>

   # Single file:
   python -m verb_analyzer.evaluate  <data_dir>  path/to/file.dat

   # Save full error log:
   python -m verb_analyzer.evaluate  <data_dir>  <hdtb_path>  --errors errors.tsv

   # Adjust error sample size:
   python -m verb_analyzer.evaluate  <data_dir>  <hdtb_path>  --max-errors 300

ARGUMENTS
---------
   data_dir   Directory containing MorpHIN data files (same arg as CLI).
   hdtb_path  Path to a single .dat file or a directory of .dat files.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Optional

# ── Make sure the package is importable from any cwd ─────────────────────────
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_PKG_DIR)
for _p in (_PKG_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verb_analyzer import VerbAnalyzer


# ==============================================================================
# SECTION 1: HDTB FILE PARSER
# ==============================================================================

# CPOS tags that identify verbal tokens in HDTB
_VERBAL_CPOS = {"VM", "VAUX"}

# HDTB feature mapping — decoded from the actual `tam` field in the corpus.
# `asp`, `ten`, `mood` keys do NOT exist in HDTB verbal tokens.
# Tense/aspect/mood is encoded entirely in the `tam` transliteration field.
#
#   tam=wA   → ता  → imperfective participle
#   tam=yA   → या  → perfective participle
#   tam=yA1  → या  → perfective participle (alternate annotation)
#   tam=vA   → वा  → perfective (rare variant)
#   tam=gA   → गा  → future
#   tam=WA   → था  → past/habitual auxiliary
#   tam=hE   → है  → present auxiliary
#   tam=eM   → एं  → subjunctive
#   tam=ao   → ओ   → imperative (low register)
#   tam=aO   → ओ   → imperative (high register)
#   tam=Uz   → ऊं  → subjunctive first person
#   tam=nA   → ना  → infinitive  (no tense/aspect/mood asserted)
#   tam=kara → कर  → conjunctive (no tense/aspect/mood asserted)
#   tam=0    → ∅   → zero marker / light verb (no assertion)
_TAM_TO_FEATURES: dict[str, dict] = {
    "wA":   {"aspect": "imperfective"},
    "yA":   {"aspect": "perfective"},
    "yA1":  {"aspect": "perfective"},
    "vA":   {"aspect": "perfective"},
    "gA":   {"tense":  "future"},
    "WA":   {"tense":  "past"},
    "hE":   {"tense":  "present"},
    "eM":   {"mood":   "subjunctive"},
    "ao":   {"mood":   "imperative"},
    "aO":   {"mood":   "imperative"},
    "Uz":   {"mood":   "subjunctive"},
    "nA":   {},   # infinitive  — no tense/aspect/mood to assert
    "kara": {},   # conjunctive — no tense/aspect/mood to assert
    "0":    {},   # zero marker — no assertion
}

_GEN_MAP: dict[str, Optional[str]] = {
    "m":   "M",
    "f":   "F",
    "any": None,
}
_NUM_MAP: dict[str, Optional[str]] = {
    "sg":  "S",
    "pl":  "P",
    "any": None,
}
# pers-1h / pers-2h / pers-3h are honorific variants — collapse to base person
_PER_MAP: dict[str, Optional[str]] = {
    "1":   "1",  "1h": "1",
    "2":   "2",  "2h": "2",
    "3":   "3",  "3h": "3",
    "any": None,
}


def _parse_feats(feat_str: str) -> dict[str, str]:
    """
    Parse a pipe-separated HDTB feature string into a plain dict.

    'cat-v|gen-m|num-sg|per-3|asp-perf|ten-past|...'
    → {'cat': 'v', 'gen': 'm', 'num': 'sg', 'per': '3', 'asp': 'perf', ...}

    Splits on the FIRST '-' only so multi-hyphen values survive intact.
    """
    result: dict[str, str] = {}
    for part in feat_str.split("|"):
        part = part.strip()
        if not part or part == "_":
            continue
        idx = part.find("-")
        if idx == -1:
            continue
        result[part[:idx]] = part[idx + 1:]
    return result


def _map_features(feat_dict: dict[str, str]) -> dict[str, Optional[str]]:
    """
    Convert an HDTB feature dict to VerbAnalyzer notation.

    HDTB does not have asp/ten/mood keys on verbal tokens.
    Tense, aspect, and mood are decoded entirely from the `tam` field.
    Person uses the key `pers` (not `per`), and honorific variants
    (1h, 2h, 3h) are collapsed to their base values.

    Returns None for any dimension that is absent or 'any', so the
    evaluator skips those slots during comparison rather than penalising
    the analyzer for unannotated dimensions.
    """
    raw_gen = feat_dict.get("gen",  "")
    raw_num = feat_dict.get("num",  "")
    raw_per = feat_dict.get("pers", "")   # HDTB uses 'pers', not 'per'
    raw_tam = feat_dict.get("tam",  "")

    # Decode tense/aspect/mood from tam; default to empty dict (all None)
    tam_feats = _TAM_TO_FEATURES.get(raw_tam, {})

    return {
        "gender": _GEN_MAP.get(raw_gen),
        "number": _NUM_MAP.get(raw_num),
        "person": _PER_MAP.get(raw_per),
        "aspect": tam_feats.get("aspect"),
        "tense":  tam_feats.get("tense"),
        "mood":   tam_feats.get("mood"),
    }


def parse_hdtb_file(filepath: str) -> tuple[list[dict], int]:
    """
    Parse a single HDTB CoNLL .dat file.

    Returns (verb_tokens, skipped_count).

    skipped_count counts VM/VAUX tokens that could not be used:
      - fewer than 6 columns
      - lemma == '_'  (unannotated in HDTB)
    """
    tokens:  list[dict] = []
    skipped: int        = 0
    fname = os.path.basename(filepath)

    with open(filepath, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.rstrip("\n")

            if not line.strip() or line.strip().startswith("#"):
                continue

            cols = line.split()

            if len(cols) < 5:
                continue
            cpos = cols[4].strip()
            if cpos not in _VERBAL_CPOS:
                continue

            # From here the token is a verbal candidate; any miss is counted
            if len(cols) < 6:
                skipped += 1
                continue

            surface  = unicodedata.normalize("NFC", cols[1].strip())
            lemma    = unicodedata.normalize("NFC", cols[2].strip())
            feat_str = cols[5].strip()

            if lemma == "_" or not lemma:
                skipped += 1
                continue

            feat_dict = _parse_feats(feat_str)
            mapped    = _map_features(feat_dict)

            tokens.append({
                "surface": surface,
                "lemma":   lemma,
                "gender":  mapped["gender"],
                "number":  mapped["number"],
                "person":  mapped["person"],
                "aspect":  mapped["aspect"],
                "tense":   mapped["tense"],
                "mood":    mapped["mood"],
                "cpos":    cpos,
                "file":    fname,
                "line_no": line_no,
            })

    return tokens, skipped


def collect_hdtb_files(path: str) -> list[str]:
    """Return [path] if file, or all .dat files under path if directory."""
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, fnames in os.walk(path):
        for fn in sorted(fnames):
            if fn.endswith(".dat"):
                files.append(os.path.join(root, fn))
    return sorted(files)


# ==============================================================================
# SECTION 2: MATCHING LOGIC
# ==============================================================================

def _lemma_matches(predicted_lemma: str, gold_lemma: str) -> bool:
    """
    NFC-normalised string equality.

    The verb analyzer normalises internally, so by the time a lemma comes
    back it is already NFC. We normalise the gold lemma here to avoid
    false negatives from encoding disagreements between HDTB and MorpHIN.
    """
    gold_nfc = unicodedata.normalize("NFC", gold_lemma)
    pred_nfc = unicodedata.normalize("NFC", predicted_lemma)
    return pred_nfc == gold_nfc


def _feature_dict_from_analysis(analysis) -> dict[str, Optional[str]]:
    """Extract a flat feature dict from a VerbAnalysis object."""
    f = analysis.features
    return {
        "gender": f.gender,
        "number": f.number,
        "person": f.person,
        "aspect": f.aspect,
        "tense":  f.tense,
        "mood":   f.mood,
    }


def _analyses_match_lemma(analyses: list, gold_lemma: str) -> bool:
    return any(_lemma_matches(a.lemma, gold_lemma) for a in analyses)


def _analyses_match_full(
    analyses:    list,
    gold_lemma:  str,
    gold_feats:  dict[str, Optional[str]],
) -> bool:
    """
    Oracle full match: any analysis with correct lemma AND all non-None
    gold features matching.

    Dimensions where gold is None (unannotated / 'any') are not checked,
    consistent with the noun evaluator's treatment of gen-any.
    """
    for a in analyses:
        if not _lemma_matches(a.lemma, gold_lemma):
            continue
        pred = _feature_dict_from_analysis(a)
        match = True
        for key, gold_val in gold_feats.items():
            if gold_val is None:
                continue   # skip unannotated dimension
            if pred.get(key) != gold_val:
                match = False
                break
        if match:
            return True
    return False


def _classify_failure(analyses: list, gold_lemma: str) -> str:
    """
    Classify WHY a token failed full match.

    no_analysis    — analyzer returned nothing (unhandled form)
    wrong_lemma    — analyses returned but none had the right lemma
    wrong_features — correct lemma, but feature bundle mismatch
    """
    if not analyses:
        return "no_analysis"
    if not _analyses_match_lemma(analyses, gold_lemma):
        return "wrong_lemma"
    return "wrong_features"


# ==============================================================================
# SECTION 3: EVALUATION LOOP
# ==============================================================================

def _best_verbal_type(analyses: list) -> str:
    """
    Pick the most informative verbal_type label for a token's analyses,
    used for the by-verbal_type breakdown table.
    """
    if any(a.irregular for a in analyses):
        return "irregular"
    types = [a.features.verbal_type for a in analyses if a.features.verbal_type]
    if not types:
        return "unknown"
    # Priority: finite > participle > infinitive > conjunctive
    for vt in ("finite", "participle", "infinitive", "conjunctive"):
        if vt in types:
            return vt
    return types[0]


def _best_confidence(analyses: list) -> str:
    order = {"high": 0, "medium": 1, "low": 2}
    if not analyses:
        return "none"
    return min(analyses, key=lambda a: order.get(a.confidence, 99)).confidence


def _run_evaluation(
    all_tokens: list[dict],
    va:         VerbAnalyzer,
    max_errors: int = 200,
) -> dict:
    """Core evaluation loop."""
    total         = 0
    covered       = 0
    lemma_correct = 0
    lemma_only    = 0
    full_correct  = 0

    by_vtype:      dict[str, Counter] = defaultdict(Counter)
    by_confidence: dict[str, Counter] = defaultdict(Counter)
    by_cpos:       dict[str, Counter] = defaultdict(Counter)
    failure_types: Counter            = Counter()

    all_errors: list[dict] = []

    for tok in all_tokens:
        total += 1
        surface    = tok["surface"]
        gold_lemma = tok["lemma"]
        gold_feats = {
            "gender": tok["gender"],
            "number": tok["number"],
            "person": tok["person"],
            "aspect": tok["aspect"],
            "tense":  tok["tense"],
            "mood":   tok["mood"],
        }

        analyses = va.analyze(surface)

        has_analysis = bool(analyses)
        lemma_ok     = _analyses_match_lemma(analyses, gold_lemma)
        full_ok      = _analyses_match_full(analyses, gold_lemma, gold_feats)

        if has_analysis: covered       += 1
        if lemma_ok:     lemma_correct += 1
        if full_ok:      full_correct  += 1
        if lemma_ok and not full_ok:
            lemma_only += 1

        vtype = _best_verbal_type(analyses) if analyses else "unknown"
        conf  = _best_confidence(analyses)
        cpos  = tok["cpos"]

        by_vtype[vtype]["total"]    += 1
        by_vtype[vtype]["covered"]  += int(has_analysis)
        by_vtype[vtype]["lemma_ok"] += int(lemma_ok)
        by_vtype[vtype]["full_ok"]  += int(full_ok)

        by_confidence[conf]["total"]    += 1
        by_confidence[conf]["covered"]  += int(has_analysis)
        by_confidence[conf]["full_ok"]  += int(full_ok)

        by_cpos[cpos]["total"]    += 1
        by_cpos[cpos]["covered"]  += int(has_analysis)
        by_cpos[cpos]["lemma_ok"] += int(lemma_ok)
        by_cpos[cpos]["full_ok"]  += int(full_ok)

        if not full_ok:
            ftype = _classify_failure(analyses, gold_lemma)
            failure_types[ftype] += 1
            all_errors.append({
                "surface":      surface,
                "gold_lemma":   gold_lemma,
                "gold_feats":   gold_feats,
                "analyses":     analyses,
                "vtype":        vtype,
                "confidence":   conf,
                "cpos":         cpos,
                "failure_type": ftype,
                "file":         tok["file"],
                "line_no":      tok["line_no"],
            })

    errors = _stratified_sample(all_errors, max_errors, key="vtype")

    return {
        "total":          total,
        "covered":        covered,
        "lemma_correct":  lemma_correct,
        "lemma_only":     lemma_only,
        "full_correct":   full_correct,
        "by_vtype":       dict(by_vtype),
        "by_confidence":  dict(by_confidence),
        "by_cpos":        dict(by_cpos),
        "failure_types":  dict(failure_types),
        "errors":         errors,
        "all_errors":     all_errors,
    }


def _stratified_sample(
    errors:    list[dict],
    n:         int,
    key:       str = "vtype",
) -> list[dict]:
    """
    Return up to n errors sampled proportionally across strata.
    Every stratum with errors gets at least 1 slot.
    Mirrors Nouns/evaluate.py _stratified_sample exactly.
    """
    if len(errors) <= n:
        return list(errors)

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for e in errors:
        by_stratum[e[key]].append(e)

    total_errors = len(errors)
    sample: list[dict] = []

    for stratum, stratum_errors in sorted(by_stratum.items()):
        share = len(stratum_errors) / total_errors
        quota = max(1, round(n * share))
        shuffled = list(stratum_errors)
        random.shuffle(shuffled)
        sample.extend(shuffled[:quota])

    random.shuffle(sample)
    return sample[:n]


def evaluate(
    data_dir:   str,
    hdtb_path:  str,
    max_errors: int = 200,
    seed:       int = 42,
) -> dict:
    """
    Run full evaluation. Returns the results dict (also printed by print_report).
    """
    random.seed(seed)

    print(f"Loading VerbAnalyzer from data_dir: {data_dir}")
    va = VerbAnalyzer(data_dir=data_dir)
    print(f"  Irregular mapping: {len(va.irregulars):,} entries")
    print(f"  Suffix rules:      {len(va.suffix_rules):,} rules")

    hdtb_files = collect_hdtb_files(hdtb_path)
    if not hdtb_files:
        print(f"ERROR: no .dat files found at {hdtb_path}")
        sys.exit(1)
    print(f"\nFound {len(hdtb_files)} HDTB file(s) — parsing...")

    all_tokens:    list[dict] = []
    total_skipped: int        = 0
    for fp in hdtb_files:
        toks, skipped = parse_hdtb_file(fp)
        all_tokens.extend(toks)
        total_skipped += skipped

    print(f"  Verbal tokens extracted: {len(all_tokens):,}")
    if total_skipped:
        print(f"  Skipped (no lemma / malformed): {total_skipped:,}")

    cpos_dist = Counter(t["cpos"] for t in all_tokens)
    print(f"  VM (main verb): {cpos_dist.get('VM', 0):,}   "
          f"VAUX (auxiliary): {cpos_dist.get('VAUX', 0):,}")
    print()

    results = _run_evaluation(all_tokens, va, max_errors)
    results["skipped"]   = total_skipped
    results["data_dir"]  = data_dir
    results["hdtb_path"] = hdtb_path

    return results


# ==============================================================================
# SECTION 4: REPORTING
# ==============================================================================

def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "N/A"


def _print_summary(results: dict) -> None:
    total      = results["total"]
    skipped    = results.get("skipped", 0)
    covered    = results["covered"]
    lemma_ok   = results["lemma_correct"]
    lemma_only = results["lemma_only"]
    full_ok    = results["full_correct"]

    W = 64
    print("=" * W)
    print("  HDTB VERB EVALUATION RESULTS")
    print("=" * W)
    print(f"\n{'Total verbal tokens evaluated:':<44} {total:>6,}")
    if skipped:
        print(f"{'Skipped (no lemma / malformed):':<44} {skipped:>6,}")
    print(f"{'Coverage (any analysis returned):':<44} {covered:>6,}  ({_pct(covered, total)})")
    print(f"{'Lemma accuracy (oracle):':<44} {lemma_ok:>6,}  ({_pct(lemma_ok, total)})")
    print(f"{'Correct lemma, wrong features:':<44} {lemma_only:>6,}  ({_pct(lemma_only, total)})")
    print(f"{'Full match accuracy (oracle):':<44} {full_ok:>6,}  ({_pct(full_ok, total)})")
    print(f"  (full match = correct lemma + all non-None gold features)")
    print(f"  (oracle = correct if ANY returned analysis matches gold)")
    print(f"  (gen/num/per/asp/ten/mood skipped when gold is 'any' or blank)")


def _print_vtype_table(by_vtype: dict) -> None:
    W = 64
    print(f"\n{'─' * W}")
    print("  BREAKDOWN BY VERBAL TYPE")
    print(f"{'─' * W}")
    print(f"  {'Type':<14} {'Total':>7} {'Covered':>9} {'Lemma OK':>10} "
          f"{'Full OK':>9} {'Accuracy':>10}")
    print(f"  {'─'*14} {'─'*7} {'─'*9} {'─'*10} {'─'*9} {'─'*10}")
    order = ["finite", "participle", "infinitive", "conjunctive", "irregular", "unknown"]
    printed: set[str] = set()
    for vt in order + sorted(by_vtype.keys()):
        if vt in printed or vt not in by_vtype:
            continue
        printed.add(vt)
        c = by_vtype[vt]
        t = c["total"]
        print(f"  {vt:<14} {t:>7,} {c['covered']:>8,}  "
              f"{c.get('lemma_ok', 0):>9,}  "
              f"{c['full_ok']:>8,}  "
              f"{_pct(c['full_ok'], t):>9}")


def _print_cpos_table(by_cpos: dict) -> None:
    W = 64
    print(f"\n{'─' * W}")
    print("  BREAKDOWN BY CPOS TAG")
    print(f"{'─' * W}")
    print(f"  {'CPOS':<10} {'Total':>7} {'Covered':>9} {'Lemma OK':>10} "
          f"{'Full OK':>9} {'Accuracy':>10}")
    print(f"  {'─'*10} {'─'*7} {'─'*9} {'─'*10} {'─'*9} {'─'*10}")
    for cpos in sorted(by_cpos.keys()):
        c = by_cpos[cpos]
        t = c["total"]
        note = "  ← main verb" if cpos == "VM" else "  ← auxiliary"
        print(f"  {cpos:<10} {t:>7,} {c['covered']:>8,}  "
              f"{c.get('lemma_ok', 0):>9,}  "
              f"{c['full_ok']:>8,}  "
              f"{_pct(c['full_ok'], t):>9}{note}")


def _print_confidence_table(by_confidence: dict) -> None:
    W = 64
    print(f"\n{'─' * W}")
    print("  BREAKDOWN BY ANALYZER CONFIDENCE")
    print(f"{'─' * W}")
    print(f"  {'Tier':<10} {'Total':>7} {'Covered':>9} {'Full OK':>9} {'Accuracy':>10}")
    print(f"  {'─'*10} {'─'*7} {'─'*9} {'─'*9} {'─'*10}")
    tier_order = ["high", "medium", "low", "none"]
    printed: set[str] = set()
    for tier in tier_order + sorted(by_confidence.keys()):
        if tier in printed or tier not in by_confidence:
            continue
        printed.add(tier)
        c    = by_confidence[tier]
        t    = c["total"]
        note = "  ← no analysis" if tier == "none" else ""
        print(f"  {tier:<10} {t:>7,} {c['covered']:>8,}  "
              f"{c['full_ok']:>8,}  {_pct(c['full_ok'], t):>9}{note}")


def _print_failure_breakdown(failure_types: dict, total_failures: int) -> None:
    W = 64
    print(f"\n{'─' * W}")
    print("  FAILURE TYPE BREAKDOWN")
    print(f"{'─' * W}")
    meanings = {
        "no_analysis":    "no suffix rule or irregular entry matched",
        "wrong_lemma":    "analyses returned but lemma recovery was wrong",
        "wrong_features": "correct lemma, but feature bundle mismatch",
    }
    print(f"  {'Type':<20} {'Count':>7} {'% of misses':>13}  Meaning")
    print(f"  {'─'*20} {'─'*7} {'─'*13}  {'─'*42}")
    for ftype in ["no_analysis", "wrong_lemma", "wrong_features"]:
        n = failure_types.get(ftype, 0)
        if n == 0:
            continue
        print(f"  {ftype:<20} {n:>7,} {_pct(n, total_failures):>12}   {meanings[ftype]}")


def _print_error_sample(errors: list[dict]) -> None:
    W = 64
    print(f"\n{'─' * W}")
    print(f"  ERROR SAMPLE ({len(errors)} tokens, stratified by verbal type)")
    print(f"{'─' * W}")
    for i, e in enumerate(errors, 1):
        gf = e["gold_feats"]
        gold_str = (
            f"lemma={e['gold_lemma']}  "
            f"asp={gf['aspect']}  ten={gf['tense']}  mood={gf['mood']}  "
            f"gen={gf['gender']}  num={gf['number']}  per={gf['person']}"
        )
        ftype = e.get("failure_type", "?")
        print(f"\n  {i:>3}. [{ftype}]  surface={e['surface']}  "
              f"vtype={e['vtype']}/conf={e['confidence']}  cpos={e['cpos']}")
        print(f"       {e['file']} line {e['line_no']}")
        print(f"       GOLD:      {gold_str}")
        if e["analyses"]:
            for a in e["analyses"][:3]:
                print(f"       PREDICTED: {a}")
        else:
            print(f"       PREDICTED: (no analysis returned)")


def print_report(results: dict) -> None:
    _print_summary(results)
    _print_vtype_table(results["by_vtype"])
    _print_cpos_table(results["by_cpos"])
    _print_confidence_table(results["by_confidence"])

    total_failures = results["total"] - results["full_correct"]
    _print_failure_breakdown(results["failure_types"], total_failures)

    print(f"\n{'=' * 64}")
    if results["errors"]:
        print(f"  (use --sample <path> to save the stratified error sample to a file)")


def save_errors(results: dict, path: str) -> None:
    """Save the full error list (not just the printed sample) as TSV."""
    import csv
    all_errors = results.get("all_errors", results["errors"])
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "surface", "gold_lemma",
            "gold_asp", "gold_ten", "gold_mood",
            "gold_gen", "gold_num", "gold_per",
            "vtype", "confidence", "cpos", "failure_type",
            "n_predicted", "pred_lemmas",
            "file", "line_no",
        ])
        for e in all_errors:
            gf = e["gold_feats"]
            pred_lemmas = "|".join(a.lemma for a in e["analyses"]) or ""
            writer.writerow([
                e["surface"], e["gold_lemma"],
                gf.get("aspect") or "", gf.get("tense") or "", gf.get("mood") or "",
                gf.get("gender") or "", gf.get("number") or "", gf.get("person") or "",
                e["vtype"], e["confidence"], e["cpos"],
                e.get("failure_type", ""),
                len(e["analyses"]), pred_lemmas,
                e["file"], e["line_no"],
            ])
    print(f"\nFull error log ({len(all_errors):,} entries) saved to: {path}")



def save_sample(results: dict, path: str) -> None:
    """Save the stratified error sample as a readable text file."""
    errors = results["errors"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("VERB ANALYZER — STRATIFIED ERROR SAMPLE\n")
        f.write(f"{len(errors)} tokens, stratified by verbal type\n")
        f.write("=" * 64 + "\n")
        for i, e in enumerate(errors, 1):
            gf = e["gold_feats"]
            gold_str = (
                f"lemma={e['gold_lemma']}  "
                f"asp={gf['aspect']}  ten={gf['tense']}  mood={gf['mood']}  "
                f"gen={gf['gender']}  num={gf['number']}  per={gf['person']}"
            )
            ftype = e.get("failure_type", "?")
            f.write(f"\n{i:>3}. [{ftype}]  surface={e['surface']}  vtype={e['vtype']}/conf={e['confidence']}  cpos={e['cpos']}\n")
            f.write(f"     {e['file']} line {e['line_no']}\n")
            f.write(f"     GOLD:      {gold_str}\n")
            if e["analyses"]:
                for a in e["analyses"][:3]:
                    f.write(f"     PREDICTED: {a}\n")
            else:
                f.write("     PREDICTED: (no analysis returned)\n")
    print(f"\nError sample ({len(errors):,} entries) saved to: {path}")

# ==============================================================================
# SECTION 5: CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Hindi verb analyzer against HDTB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m verb_analyzer.evaluate  path/to/morphin/  HDTB/InterChunk/CoNLL/utf/
  python -m verb_analyzer.evaluate  .                 path/to/file.dat
  python -m verb_analyzer.evaluate  .                 HDTB/  --errors errors.tsv
  python -m verb_analyzer.evaluate  .                 HDTB/  --max-errors 300
        """,
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing MorpHIN data files (same as CLI first arg).",
    )
    parser.add_argument(
        "hdtb_path",
        help="Path to a single HDTB .dat file or a directory of .dat files.",
    )
    parser.add_argument(
        "--errors",
        default=None,
        metavar="PATH",
        help="Save the full error log as TSV to this path.",
    )
    parser.add_argument(
        "--sample",
        default=None,
        metavar="PATH",
        help="Save the stratified error sample (up to --max-errors entries) as a readable text file.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=200,
        metavar="N",
        help="Max error examples in the printed sample (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified error sampling (default: 42).",
    )
    args = parser.parse_args()

    results = evaluate(
        data_dir=args.data_dir,
        hdtb_path=args.hdtb_path,
        max_errors=args.max_errors,
        seed=args.seed,
    )
    print_report(results)

    if args.errors:
        save_errors(results, args.errors)

    if args.sample:
        save_sample(results, args.sample)


if __name__ == "__main__":
    main()