"""
evaluate.py — HDTB Noun Evaluation
=====================================
Evaluates the noun analyzer against the Hindi Dependency Treebank (HDTB).

HOW IT WORKS
------------
1. PARSE HDTB FILES
   Reads one or more HDTB CoNLL files. Each non-blank, non-comment line is
   one token. Columns are whitespace-separated:

       tok_id  surface  lemma  pos_cat  CPOS  feats  head  deprel  ...

   Column indices (0-based, confirmed from actual HDTB files):
       [0]  sentence-local token id
       [1]  surface form  (Devanagari)
       [2]  lemma         (Devanagari)
       [3]  pos_cat       ('n', 'v', 'adj' ...)
       [4]  CPOS          ('NN', 'VM', 'JJ' ...)
       [5]  feats         pipe-separated key-value pairs
       [6]  head
       [7]  deprel

   Only tokens where CPOS == 'NN' are kept (common nouns).
   Tokens with fewer than 6 columns or with lemma == '_' are skipped and
   counted in a 'skipped' counter so nothing silently disappears.

2. PARSE GOLD FEATURES
   The feats column uses pipe-separated key-value pairs with '-' as separator:
       cat-n|gen-m|num-sg|pers-3|case-d|vib-0|tam-0|...

   Relevant keys extracted:
       gen   ->  m / f / any
       num   ->  sg / pl
       case  ->  d (direct) / o (oblique, regardless of vib postposition)

   Mapped to your system's notation:
       gen-m  -> 'M',   gen-f  -> 'F',   gen-any -> None (skipped in comparison)
       num-sg -> 'SG',  num-pl -> 'PL'
       case-d -> 'DIR', case-o -> 'OBL'

3. LEMMA NORMALISATION
   Before comparing gold lemma against predicted lemma, the gold lemma is
   passed through the same normalize_hindi() the analyzer uses (NFC +
   nukta stripping). This prevents false negatives where the two sources
   spell the lemma identically except for nukta encoding disagreement
   between HDTB and MorpHIN.

4. ORACLE MATCHING
   A token is counted correct if ANY returned analysis matches the gold.
   This is standard for analyzers without a disambiguator.

   Four match levels:
     - coverage:          any analysis returned at all
     - lemma_match:       any analysis has the correct lemma
     - lemma_only_match:  correct lemma but wrong features (NEW)
     - full_match:        correct lemma + gender + number + case
       (gender/case skipped when gold value is None or 'any')

5. FAILURE CLASSIFICATION
   For each miss, the evaluator records WHY it failed:
     - no_analysis:    analyzer returned nothing at all
     - oov:            gold lemma not in lexicon (data gap, not a rule error)
     - wrong_lemma:    wrong lemma recovered (suffix rule or lexicon error)
     - wrong_features: right lemma, wrong gender/number/case (paradigm table error)

6. METRICS REPORTED
   Overall: skipped count, coverage, lemma accuracy, lemma-only, full match
   By paradigm class (M1-F5, OOV)
   By confidence tier (certain / heuristic / OOV)
   Failure type breakdown (separates rule errors from data gaps)
   Stratified error sample (proportional across paradigm classes)
   Optional lexicon comparison (--lexicon2)

USAGE
-----
   python -m Nouns.evaluate path/to/HDTB/dir/
   python -m Nouns.evaluate path/to/file.dat --errors Nouns/data/errors.tsv
   python -m Nouns.evaluate path/to/dir/ \\
       --lexicon  Nouns/data/noun_lexicon_expanded.tsv \\
       --lexicon2 Nouns/data/noun_lexicon_complete.tsv

FILE STRUCTURE
--------------
   Nouns/
   ├── evaluate.py               <- this file
   ├── analyzer.py
   ├── noun_paradigm_templates.py
   └── data/
       ├── noun_lexicon_expanded.tsv
       └── noun_lexicon_complete.tsv   (optional, from gender_predictor.py)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Optional

# -- Make sure sibling imports work regardless of cwd -------------------------
_NOUNS_DIR = os.path.dirname(os.path.abspath(__file__))
if _NOUNS_DIR not in sys.path:
    sys.path.insert(0, _NOUNS_DIR)

from analyzer import load_lexicon, analyze, normalize_hindi
from noun_paradigm_templates import PARADIGM_TABLES

# -- Default lexicon path -----------------------------------------------------
_DEFAULT_LEXICON = os.path.join(_NOUNS_DIR, "data", "noun_lexicon_expanded.tsv")


# ==============================================================================
# SECTION 1: HDTB FILE PARSER
# ==============================================================================

def _parse_feats(feat_str: str) -> dict[str, str]:
    """
    Parse a pipe-separated HDTB feature string into a dict.

    Input:  'cat-n|gen-m|num-sg|pers-3|case-d|vib-0|tam-0|chunkId-NP|...'
    Output: {'cat': 'n', 'gen': 'm', 'num': 'sg', 'case': 'd', ...}

    Splits each field on the FIRST '-' only so values like 'vib-0_men'
    are stored as key='vib', val='0_men' without breaking anything.
    """
    result: dict[str, str] = {}
    for part in feat_str.split("|"):
        part = part.strip()
        if not part or part == "_":
            continue
        idx = part.find("-")
        if idx == -1:
            continue
        key = part[:idx]
        val = part[idx + 1:]
        result[key] = val
    return result


def _map_features(feat_dict: dict[str, str]) -> dict[str, Optional[str]]:
    """
    Convert HDTB feature dict to the analyzer's notation.

    gen-any  -> None  (gender unknown; skipped in comparison)
    case-o*  -> OBL   (matches 'o', 'o_men', 'o_ko', etc.)
    """
    gender_map = {"m": "M", "f": "F"}
    number_map = {"sg": "SG", "pl": "PL"}

    raw_gen  = feat_dict.get("gen",  "")
    raw_num  = feat_dict.get("num",  "")
    raw_case = feat_dict.get("case", "")

    return {
        "gender": gender_map.get(raw_gen),
        "number": number_map.get(raw_num),
        "case":   "DIR" if raw_case == "d"
                  else ("OBL" if raw_case.startswith("o") else None),
    }


def parse_hdtb_file(filepath: str) -> tuple[list[dict], int]:
    """
    Parse a single HDTB CoNLL file. Returns (noun_tokens, skipped_count).

    skipped_count counts NN tokens that could not be used:
      - fewer than 6 columns
      - lemma == '_'  (unannotated in HDTB)

    HDTB column order (confirmed, 0-indexed):
        [0] tok_id  [1] surface  [2] lemma
        [3] pos_cat [4] CPOS     [5] feats  [6] head  [7] deprel
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

            # CPOS is at index 4
            if len(cols) < 5:
                continue
            cpos = cols[4].strip()
            if cpos != "NN":
                continue

            # From here the token is a candidate NN -- any skip is counted
            if len(cols) < 6:
                skipped += 1
                continue

            surface  = unicodedata.normalize("NFC", cols[1].strip())
            lemma    = unicodedata.normalize("NFC", cols[2].strip())
            feat_str = cols[5].strip()

            # Skip tokens with no lemma annotation
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
                "case":    mapped["case"],
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

def _lemma_matches(predicted_lemma: str, gold_lemma: str, gold_lemma_norm: str) -> bool:
    """
    Two-stage lemma comparison:
      1. Strict NFC equality
      2. Normalised fallback (nukta-stripped)

    The fallback prevents false negatives when HDTB and MorpHIN disagree
    on nukta encoding for the same word (e.g. khidki with vs without nukta).
    Both are checked so a genuinely wrong lemma still fails.
    """
    pred_norm = normalize_hindi(predicted_lemma)
    return predicted_lemma == gold_lemma or pred_norm == gold_lemma_norm


def _analyses_match_lemma(
    analyses: list[dict], gold_lemma: str, gold_lemma_norm: str
) -> bool:
    return any(
        _lemma_matches(a["lemma"], gold_lemma, gold_lemma_norm)
        for a in analyses
    )


def _analyses_match_full(
    analyses:        list[dict],
    gold_lemma:      str,
    gold_lemma_norm: str,
    gold_gender:     Optional[str],
    gold_number:     Optional[str],
    gold_case:       Optional[str],
) -> bool:
    """
    Oracle full match: any analysis with correct lemma AND all non-None features.

    When gold_gender is None (gen-any in HDTB), gender is not checked.
    This avoids penalising the analyzer for dimensions the treebank left
    underspecified.
    """
    for a in analyses:
        if not _lemma_matches(a["lemma"], gold_lemma, gold_lemma_norm):
            continue
        if gold_gender is not None and a.get("gender") != gold_gender:
            continue
        if gold_number is not None and a.get("number") != gold_number:
            continue
        if gold_case   is not None and a.get("case")   != gold_case:
            continue
        return True
    return False


def _classify_failure(
    analyses:        list[dict],
    gold_lemma:      str,
    gold_lemma_norm: str,
    in_lexicon:      bool,
) -> str:
    """
    Classify WHY a token failed full match:

      no_analysis   -- analyzer returned nothing (OOV or no matching suffix rule)
      oov           -- analyzer returned something but gold lemma not in lexicon
                       (the rules may be correct but we cannot confirm)
      wrong_lemma   -- analyses returned but none had the right lemma (rule error)
      wrong_features-- lemma correct but feature bundle wrong (paradigm table error)
    """
    if not analyses:
        return "no_analysis"
    if not in_lexicon:
        return "oov"
    if not _analyses_match_lemma(analyses, gold_lemma, gold_lemma_norm):
        return "wrong_lemma"
    return "wrong_features"


# ==============================================================================
# SECTION 3: EVALUATION LOOP
# ==============================================================================

def _run_evaluation(
    all_tokens:     list[dict],
    lex_lookup:     dict,
    lex_display:    dict,
    lex_confidence: dict,
    max_errors:     int = 200,
) -> dict:
    """
    Core evaluation loop. Separated from evaluate() so it can be called
    twice when --lexicon2 is supplied without re-parsing the HDTB files.
    """
    total         = 0
    covered       = 0
    lemma_correct = 0
    lemma_only    = 0   # correct lemma but wrong features
    full_correct  = 0

    by_paradigm:   dict[str, Counter] = defaultdict(Counter)
    by_confidence: dict[str, Counter] = defaultdict(Counter)
    failure_types: Counter            = Counter()

    all_errors: list[dict] = []

    for tok in all_tokens:
        total += 1
        surface         = tok["surface"]
        gold_lemma      = tok["lemma"]
        gold_lemma_norm = normalize_hindi(gold_lemma)
        gold_gender     = tok["gender"]
        gold_number     = tok["number"]
        gold_case       = tok["case"]

        analyses = analyze(
            surface, lex_lookup, lex_display, lex_confidence, PARADIGM_TABLES
        )

        # -- Match flags ------------------------------------------------------
        has_analysis = bool(analyses)
        lemma_ok     = _analyses_match_lemma(analyses, gold_lemma, gold_lemma_norm)
        full_ok      = _analyses_match_full(
            analyses, gold_lemma, gold_lemma_norm,
            gold_gender, gold_number, gold_case
        )

        if has_analysis:  covered       += 1
        if lemma_ok:      lemma_correct += 1
        if full_ok:       full_correct  += 1
        if lemma_ok and not full_ok:
            lemma_only += 1

        # -- Paradigm / confidence lookup ------------------------------------
        paradigm   = lex_lookup.get(gold_lemma_norm, "OOV")
        confidence = lex_confidence.get(gold_lemma_norm, "OOV")
        in_lexicon = paradigm != "OOV"

        by_paradigm[paradigm]["total"]    += 1
        by_paradigm[paradigm]["covered"]  += int(has_analysis)
        by_paradigm[paradigm]["lemma_ok"] += int(lemma_ok)
        by_paradigm[paradigm]["full_ok"]  += int(full_ok)

        by_confidence[confidence]["total"]   += 1
        by_confidence[confidence]["covered"] += int(has_analysis)
        by_confidence[confidence]["full_ok"] += int(full_ok)

        # -- Failure classification -------------------------------------------
        if not full_ok:
            ftype = _classify_failure(
                analyses, gold_lemma, gold_lemma_norm, in_lexicon
            )
            failure_types[ftype] += 1
            all_errors.append({
                "surface":      surface,
                "gold_lemma":   gold_lemma,
                "gold_gender":  gold_gender,
                "gold_number":  gold_number,
                "gold_case":    gold_case,
                "predicted":    analyses,
                "paradigm":     paradigm,
                "confidence":   confidence,
                "failure_type": ftype,
                "file":         tok["file"],
                "line_no":      tok["line_no"],
            })

    errors = _stratified_sample(all_errors, max_errors)

    return {
        "total":          total,
        "covered":        covered,
        "lemma_correct":  lemma_correct,
        "lemma_only":     lemma_only,
        "full_correct":   full_correct,
        "by_paradigm":    dict(by_paradigm),
        "by_confidence":  dict(by_confidence),
        "failure_types":  dict(failure_types),
        "errors":         errors,
        "all_errors":     all_errors,
    }


def _stratified_sample(errors: list[dict], n: int) -> list[dict]:
    """
    Return up to n errors sampled proportionally across paradigm classes.
    Every class with errors gets at least 1 slot, so small classes like
    M4 and F5 are not crowded out by large ones like M2 and F3.
    """
    if len(errors) <= n:
        return list(errors)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for e in errors:
        by_class[e["paradigm"]].append(e)

    total_errors = len(errors)
    sample: list[dict] = []

    for cls, cls_errors in sorted(by_class.items()):
        share = len(cls_errors) / total_errors
        quota = max(1, round(n * share))
        shuffled = list(cls_errors)
        random.shuffle(shuffled)
        sample.extend(shuffled[:quota])

    random.shuffle(sample)
    return sample[:n]


def evaluate(
    hdtb_path:     str,
    lexicon_path:  str           = _DEFAULT_LEXICON,
    lexicon2_path: Optional[str] = None,
    max_errors:    int           = 200,
    seed:          int           = 42,
) -> dict:
    """
    Run full evaluation. If lexicon2_path is given, runs a second pass
    with that lexicon and returns comparison results under key 'comparison'.
    """
    random.seed(seed)

    # -- Load primary lexicon -------------------------------------------------
    print(f"Loading lexicon from: {lexicon_path}")
    lex_lookup, lex_display, lex_confidence = load_lexicon(lexicon_path)
    print(f"  {len(lex_lookup):,} lemmas loaded")

    # -- Collect and parse HDTB files -----------------------------------------
    hdtb_files = collect_hdtb_files(hdtb_path)
    if not hdtb_files:
        print(f"ERROR: no .dat files found at {hdtb_path}")
        sys.exit(1)
    print(f"\nFound {len(hdtb_files)} HDTB file(s) -- parsing...")

    all_tokens: list[dict] = []
    total_skipped = 0
    for fp in hdtb_files:
        toks, skipped = parse_hdtb_file(fp)
        all_tokens.extend(toks)
        total_skipped += skipped

    print(f"  Common noun tokens extracted: {len(all_tokens):,}")
    if total_skipped:
        print(f"  Skipped (no lemma / malformed): {total_skipped:,}")
    print()

    # -- Primary evaluation ---------------------------------------------------
    results = _run_evaluation(
        all_tokens, lex_lookup, lex_display, lex_confidence, max_errors
    )
    results["skipped"]      = total_skipped
    results["lexicon_path"] = lexicon_path

    # -- Optional second-lexicon comparison -----------------------------------
    if lexicon2_path:
        print(f"Loading comparison lexicon from: {lexicon2_path}")
        lex2_lookup, lex2_display, lex2_confidence = load_lexicon(lexicon2_path)
        print(f"  {len(lex2_lookup):,} lemmas loaded\n")
        results2 = _run_evaluation(
            all_tokens, lex2_lookup, lex2_display, lex2_confidence, max_errors
        )
        results2["lexicon_path"] = lexicon2_path
        results["comparison"]   = results2

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
    print("  HDTB NOUN EVALUATION RESULTS")
    print("=" * W)
    print(f"\n{'Total NN tokens evaluated:':<44} {total:>6,}")
    if skipped:
        print(f"{'Skipped (no lemma / malformed):':<44} {skipped:>6,}")
    print(f"{'Coverage (any analysis returned):':<44} {covered:>6,}  ({_pct(covered, total)})")
    print(f"{'Lemma accuracy (oracle):':<44} {lemma_ok:>6,}  ({_pct(lemma_ok, total)})")
    print(f"{'Correct lemma, wrong features:':<44} {lemma_only:>6,}  ({_pct(lemma_only, total)})")
    print(f"{'Full match accuracy (oracle):':<44} {full_ok:>6,}  ({_pct(full_ok, total)})")
    print(f"  (full match = correct lemma + gender + number + case)")
    print(f"  (oracle = correct if ANY returned analysis matches gold)")
    print(f"  (gender/case skipped when gold is 'any' or unannotated)")


def _print_paradigm_table(by_paradigm: dict) -> None:
    W = 64
    print(f"\n{'─' * W}")
    print("  BREAKDOWN BY PARADIGM CLASS")
    print(f"{'─' * W}")
    print(f"  {'Class':<8} {'Total':>7} {'Covered':>9} {'Lemma OK':>10} "
          f"{'Full OK':>9} {'Accuracy':>10}")
    print(f"  {'─'*8} {'─'*7} {'─'*9} {'─'*10} {'─'*9} {'─'*10}")
    for cls in sorted(by_paradigm.keys()):
        c = by_paradigm[cls]
        t = c["total"]
        print(f"  {cls:<8} {t:>7,} {c['covered']:>8,}  "
              f"{c.get('lemma_ok', 0):>9,}  "
              f"{c['full_ok']:>8,}  "
              f"{_pct(c['full_ok'], t):>9}")


def _print_confidence_table(by_confidence: dict) -> None:
    W = 64
    print(f"\n{'─' * W}")
    print("  BREAKDOWN BY LEXICON CONFIDENCE")
    print(f"{'─' * W}")
    print(f"  {'Tier':<18} {'Total':>7} {'Covered':>9} {'Full OK':>9} {'Accuracy':>10}")
    print(f"  {'─'*18} {'─'*7} {'─'*9} {'─'*9} {'─'*10}")
    tier_order = ["certain", "heuristic", "OOV"]
    printed: set[str] = set()
    for tier in tier_order + sorted(by_confidence.keys()):
        if tier in printed or tier not in by_confidence:
            continue
        printed.add(tier)
        c    = by_confidence[tier]
        t    = c["total"]
        note = "  <- lexicon gap" if tier == "OOV" else ""
        print(f"  {tier:<18} {t:>7,} {c['covered']:>8,}  "
              f"{c['full_ok']:>8,}  {_pct(c['full_ok'], t):>9}{note}")


def _print_failure_breakdown(failure_types: dict, total_failures: int) -> None:
    W = 64
    print(f"\n{'─' * W}")
    print("  FAILURE TYPE BREAKDOWN")
    print(f"{'─' * W}")
    meanings = {
        "no_analysis":    "OOV or no matching suffix rule fired",
        "oov":            "gold lemma not in lexicon (data gap, not rule error)",
        "wrong_lemma":    "wrong lemma recovered (suffix rule or lexicon class error)",
        "wrong_features": "correct lemma, wrong gender/number/case (paradigm table error)",
    }
    print(f"  {'Type':<20} {'Count':>7} {'% of misses':>13}  Meaning")
    print(f"  {'─'*20} {'─'*7} {'─'*13}  {'─'*40}")
    for ftype in ["no_analysis", "oov", "wrong_lemma", "wrong_features"]:
        n = failure_types.get(ftype, 0)
        if n == 0:
            continue
        print(f"  {ftype:<20} {n:>7,} {_pct(n, total_failures):>12}   {meanings[ftype]}")


def _print_error_sample(errors: list[dict]) -> None:
    W = 64
    print(f"\n{'─' * W}")
    print(f"  ERROR SAMPLE ({len(errors)} tokens, stratified across paradigm classes)")
    print(f"{'─' * W}")
    for i, e in enumerate(errors, 1):
        gold_str = (f"lemma={e['gold_lemma']}  gen={e['gold_gender']}  "
                    f"num={e['gold_number']}  case={e['gold_case']}")
        ftype = e.get("failure_type", "?")
        print(f"\n  {i:>3}. [{ftype}]  surface={e['surface']}  "
              f"paradigm={e['paradigm']}/{e['confidence']}")
        print(f"       {e['file']} line {e['line_no']}")
        print(f"       GOLD:      {gold_str}")
        if e["predicted"]:
            for a in e["predicted"][:3]:
                print(f"       PREDICTED: lemma={a['lemma']}  "
                      f"gen={a.get('gender')}  num={a.get('number')}  "
                      f"case={a.get('case')}  paradigm={a.get('paradigm')}")
        else:
            print(f"       PREDICTED: (no analysis returned)")


def print_report(results: dict) -> None:
    _print_summary(results)
    _print_paradigm_table(results["by_paradigm"])
    _print_confidence_table(results["by_confidence"])

    total_failures = results["total"] - results["full_correct"]
    _print_failure_breakdown(results["failure_types"], total_failures)
    _print_error_sample(results["errors"])

    # -- Comparison run -------------------------------------------------------
    if "comparison" in results:
        comp = results["comparison"]
        W    = 64
        print(f"\n{'=' * W}")
        print("  LEXICON COMPARISON")
        print(f"{'=' * W}")
        print(f"\n  {'Metric':<38} {'Lexicon 1':>12} {'Lexicon 2':>12} {'Delta':>8}")
        print(f"  {'─'*38} {'─'*12} {'─'*12} {'─'*8}")
        t = results["total"]
        for label, k in [
            ("Coverage",       "covered"),
            ("Lemma accuracy", "lemma_correct"),
            ("Full match",     "full_correct"),
        ]:
            v1   = results[k]
            v2   = comp[k]
            diff = v2 - v1
            sign = "+" if diff >= 0 else ""
            print(f"  {label:<38} {_pct(v1,t):>12} {_pct(v2,t):>12} "
                  f"{sign}{diff:>6,}")
        print(f"\n  Lexicon 1: {results['lexicon_path']}")
        print(f"  Lexicon 2: {comp['lexicon_path']}")

    print(f"\n{'=' * 64}")


def save_errors(results: dict, path: str) -> None:
    """Save the FULL error list (not just the printed sample) as TSV."""
    import csv
    all_errors = results.get("all_errors", results["errors"])
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "surface", "gold_lemma", "gold_gender", "gold_number", "gold_case",
            "paradigm", "confidence", "failure_type",
            "n_predicted", "pred_lemmas", "file", "line_no",
        ])
        for e in all_errors:
            pred_lemmas = "|".join(a["lemma"] for a in e["predicted"]) or ""
            writer.writerow([
                e["surface"], e["gold_lemma"],
                e["gold_gender"] or "", e["gold_number"] or "",
                e["gold_case"]   or "",
                e["paradigm"], e["confidence"],
                e.get("failure_type", ""),
                len(e["predicted"]), pred_lemmas,
                e["file"], e["line_no"],
            ])
    print(f"\nFull error log ({len(all_errors):,} entries) saved to: {path}")


# ==============================================================================
# SECTION 5: CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Hindi noun analyzer against HDTB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m Nouns.evaluate HDTB_pre_release_version-0.05/InterChunk/CoNLL/utf/
  python -m Nouns.evaluate path/to/file.dat --errors Nouns/data/errors.tsv
  python -m Nouns.evaluate path/to/dir/ \\
      --lexicon  Nouns/data/noun_lexicon_expanded.tsv \\
      --lexicon2 Nouns/data/noun_lexicon_complete.tsv
        """,
    )
    parser.add_argument(
        "hdtb_path",
        help="Path to a single HDTB .dat file or a directory of .dat files",
    )
    parser.add_argument(
        "--lexicon",
        default=_DEFAULT_LEXICON,
        help="Primary lexicon TSV (default: noun_lexicon_expanded.tsv)",
    )
    parser.add_argument(
        "--lexicon2",
        default=None,
        help="Optional second lexicon for comparison (e.g. noun_lexicon_complete.tsv)",
    )
    parser.add_argument(
        "--errors",
        default=None,
        help="Save full error log as TSV to this path",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=200,
        help="Max error examples in printed sample, stratified (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified error sampling (default: 42)",
    )
    args = parser.parse_args()

    results = evaluate(
        hdtb_path=args.hdtb_path,
        lexicon_path=args.lexicon,
        lexicon2_path=args.lexicon2,
        max_errors=args.max_errors,
        seed=args.seed,
    )
    print_report(results)

    if args.errors:
        save_errors(results, args.errors)


if __name__ == "__main__":
    main()