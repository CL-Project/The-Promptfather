import csv
import unicodedata
from noun_paradigm_templates import PARADIGM_TABLES


def normalize_hindi(text):
    """
    Two-step normalization:
    1. NFC — canonical ordering of combining characters
    2. Nukta stripping — removes nukta (0x93c) from ड and ढ
       so that खिड़की and खिडकी match each other during lookup.
       Hindi speakers treat these as the same word in many contexts,
       and the MorpHIN lexicon is inconsistent about nukta usage.
    """
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\u0921\u093c', '\u0921')  # ड़ → ड
    text = text.replace('\u0922\u093c', '\u0922')  # ढ़ → ढ
    return text


def analyze(surface_form, lexicon_lookup, lexicon_display, lexicon_confidence, paradigm_tables):
    """
    Given a surface form, return all valid (lemma, features) analyses.

    Fix 1: normalize_hindi is now applied here, not just in analyze_verbose.
           This means all callers (including the evaluator) get correct behavior.

    Fix 2: each result now includes a 'confidence' field ('certain' or 'heuristic')
           so the evaluator can separate errors on gold-classified words from
           errors on heuristically-classified ones.

    Strategy:
    1. For each paradigm class and each rule in that class,
       check if the surface form ends in the rule's suffix
    2. If it does, recover the candidate lemma by stripping
       the surface suffix and adding back the lemma suffix
    3. Check if that candidate lemma exists in the lexicon
       with the matching paradigm class
    4. If yes, it's a valid analysis — add to results
    """
    # FIX 1: normalize at the entry point of analyze() itself
    surface_form = normalize_hindi(surface_form)

    results = []

    for paradigm_class, rules in paradigm_tables.items():
        for (surface_suffix, lemma_suffix, gender, number, case) in rules:

            if surface_suffix:
                if not surface_form.endswith(surface_suffix):
                    continue
                stem = surface_form[:-len(surface_suffix)]
            else:
                stem = surface_form

            candidate_lemma = stem + lemma_suffix

            if candidate_lemma in lexicon_lookup and \
               lexicon_lookup[candidate_lemma] == paradigm_class:
                results.append({
                    'lemma': lexicon_display[candidate_lemma],
                    'gender': gender,
                    'number': number,
                    'case': case,
                    'paradigm': paradigm_class,
                    # FIX 2: propagate confidence so evaluator can stratify errors
                    'confidence': lexicon_confidence[candidate_lemma],
                })

    seen = set()
    unique_results = []
    for r in results:
        key = (r['lemma'], r['gender'], r['number'], r['case'])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    return unique_results


def analyze_verbose(surface_form, lexicon_lookup, lexicon_display, lexicon_confidence, paradigm_tables):
    """
    Pretty-print wrapper around analyze() for testing.
    normalize_hindi no longer needs to be called here — analyze() handles it.
    """
    results = analyze(surface_form, lexicon_lookup, lexicon_display, lexicon_confidence, paradigm_tables)
    print(f"\nInput: {surface_form}")
    if not results:
        print("  No analysis found")
    else:
        for r in results:
            print(f"  {r['lemma']} + "
                  f"[{r['paradigm']}, "
                  f"{r['gender']}, "
                  f"{r['number']}, "
                  f"{r['case']}, "
                  f"conf={r['confidence']}]")
    return results


def load_lexicon(tsv_path):
    """
    Load noun_lexicon_expanded.tsv and return three dicts keyed by
    nukta-stripped NFC form:
      lexicon_lookup     → paradigm class
      lexicon_display    → original spelling (with nuktas)
      lexicon_confidence → 'certain' or 'heuristic'
    """
    lexicon_lookup = {}
    lexicon_display = {}
    lexicon_confidence = {}

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            original = unicodedata.normalize('NFC', row['word'])
            stripped = normalize_hindi(original)
            lexicon_lookup[stripped] = row['paradigm_class']
            lexicon_display[stripped] = original
            # FIX 2: read confidence column
            lexicon_confidence[stripped] = row.get('confidence', 'certain')

    return lexicon_lookup, lexicon_display, lexicon_confidence


# ---------------------------------------------------------------------------
# Test harness (only runs when executing this file directly)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os

    tsv_path = os.path.join(os.path.dirname(__file__), 'data', 'noun_lexicon_expanded.tsv')
    lexicon_lookup, lexicon_display, lexicon_confidence = load_lexicon(tsv_path)
    print(f"Loaded {len(lexicon_lookup)} lemmas")

    test_words = [
        'लड़का', 'लड़के', 'लड़कों',
        'घर', 'घरों',
        'आदमियों',
        'आलू', 'आलुओं',
        'लड़की', 'लड़कियाँ', 'लड़कियों',
        'आशाएँ', 'आशाओं',
        'रातें', 'रातों',
        'शान्ति', 'शान्तियाँ', 'शान्तियों',
        'ऋतु', 'ऋतुएँ', 'ऋतुओं',
        'किताबों',
    ]

    additional_tests = [
        'कमरों', 'किसानों', 'नदियाँ', 'सड़कों', 'खिड़कियों', 'बातें',
    ]

    print("=== Core test suite ===")
    for word in test_words:
        analyze_verbose(word, lexicon_lookup, lexicon_display, lexicon_confidence, PARADIGM_TABLES)

    print("\n=== Additional tests ===")
    for word in additional_tests:
        analyze_verbose(word, lexicon_lookup, lexicon_display, lexicon_confidence, PARADIGM_TABLES)