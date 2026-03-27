import csv
import unicodedata
from noun_paradigm_templates import PARADIGM_TABLES

def analyze(surface_form, lexicon_lookup, lexicon_display, paradigm_tables):
    """
    Given a surface form, return all valid (lemma, features) analyses.
    
    Strategy:
    1. For each paradigm class and each rule in that class,
       check if the surface form ends in the rule's suffix
    2. If it does, recover the candidate lemma by stripping
       the surface suffix and adding back the lemma suffix
    3. Check if that candidate lemma exists in the lexicon
       with the matching paradigm class
    4. If yes, it's a valid analysis — add to results
    """
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
                    'lemma': lexicon_display[candidate_lemma],  # original form
                    'gender': gender,
                    'number': number,
                    'case': case,
                    'paradigm': paradigm_class
                })
    
    seen = set()
    unique_results = []
    for r in results:
        key = (r['lemma'], r['gender'], r['number'], r['case'])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    return unique_results

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
    # Strip nukta after ड and ढ to normalize spelling variants
    text = text.replace('\u0921\u093c', '\u0921')  # ड़ → ड
    text = text.replace('\u0922\u093c', '\u0922')  # ढ़ → ढ
    return text

# And normalize every input before analysis
def analyze_verbose(surface_form, lexicon_lookup, lexicon_display, paradigm_tables):
    """
    Pretty-print wrapper around analyze() for testing.
    """
    original_input = surface_form  # save original for display
    surface_form = normalize_hindi(surface_form)
    results = analyze(surface_form, lexicon_lookup, lexicon_display, paradigm_tables)
    print(f"\nInput: {original_input}")  # print original, not stripped
    # rest stays the same
    if not results:
        print("  No analysis found")
    else:
        for r in results:
            print(f"  {r['lemma']} + "
                  f"[{r['paradigm']}, "
                  f"{r['gender']}, "
                  f"{r['number']}, "
                  f"{r['case']}]")
    return results



# Load lexicon with two versions
lexicon_lookup = {}   # nukta-stripped keys → paradigm class
lexicon_display = {}  # nukta-stripped keys → original form with nuktas

with open('noun_lexicon_expanded.tsv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        original = unicodedata.normalize('NFC', row['word'])
        stripped = normalize_hindi(original)
        lexicon_lookup[stripped] = row['paradigm_class']
        lexicon_display[stripped] = original

print(f"Loaded {len(lexicon_lookup)} lemmas")

# Test suite — one word from each class, testing multiple forms
test_words = [
    # M1 tests
    'लड़का',    # M1 SG DIR
    'लड़के',    # M1 PL DIR / SG OBL (should return 2 analyses)
    'लड़कों',   # M1 PL OBL
    
    # M2 tests
    'घर',       # M2 SG DIR / PL DIR / SG OBL (3 analyses)
    'घरों',     # M2 PL OBL
    
    # M3 tests
    'आदमियों',  # M3 PL OBL

    # M4 tests (आलू class — masculine -ū final)
    'आलू',      # M4 SG DIR / PL DIR / SG OBL (3 analyses — invariant)
    'आलुओं',    # M4 PL OBL
    
    # F1 tests
    'लड़की',    # F1 SG DIR / SG OBL (2 analyses)
    'लड़कियाँ', # F1 PL DIR
    'लड़कियों', # F1 PL OBL
    
    # F2 tests
    'आशाएँ',   # F2 PL DIR
    'आशाओं',   # F2 PL OBL
    
    # F3 tests
    'रातें',    # F3 PL DIR
    'रातों',    # F3 PL OBL

    # F4 tests (शान्ति class — feminine -i final)
    'शान्ति',   # F4 SG DIR / SG OBL (2 analyses)
    'शान्तियाँ', # F4 PL DIR
    'शान्तियों', # F4 PL OBL

    # F5 tests (ऋतु class — feminine -u final)
    'ऋतु',      # F5 SG DIR / SG OBL (2 analyses)
    'ऋतुएँ',    # F5 PL DIR
    'ऋतुओं',    # F5 PL OBL
    
    # Unknown word test
    'किताबों',  # should find no analysis (ex_noun in lexicon)
]

additional_tests = [
    'कमरों',    # कमरा — M1, should give PL OBL
    'किसानों',  # किसान — M2, should give PL OBL  
    'नदियाँ',   # नदी — F1, should give PL DIR
    'सड़कों',   # सड़क — F3, should give PL OBL
    'खिड़कियों', # खिड़की — F1, should give PL OBL
    'बातें',    # बात — F3, should give PL DIR
]


print("=== Core test suite ===")
for word in test_words:
    analyze_verbose(word, lexicon_lookup, lexicon_display, PARADIGM_TABLES)

print("\n=== Additional tests ===")
for word in additional_tests:
    analyze_verbose(word, lexicon_lookup, lexicon_display, PARADIGM_TABLES)

# # Check what's actually stored in the lexicon for खिड़की variants
# target = unicodedata.normalize('NFC', 'खिड़की')
# print("Looking for:", [hex(ord(c)) for c in target])

# # Search for anything resembling खिड़की in the lexicon
# for word in lexicon:
#     if 'खि' in word and 'की' in word:
#         print(f"Found: {word} → {[hex(ord(c)) for c in word]} → {lexicon[word]}")

