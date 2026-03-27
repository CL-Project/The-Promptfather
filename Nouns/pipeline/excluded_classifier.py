import csv
from conflict_resolver import merged
from analyzer import normalize_hindi

# Heuristic classification for ex_noun words
ENDING_TO_CLASS = {
    'ा': 'M1',    # लड़का pattern
    'ी': 'F1',    # लड़की pattern  
    'ि': 'F4',    # शान्ति pattern
    'ई': 'F1',    # treated as F1 variant
}

heuristic_additions = {}
genuinely_unknown = []

for word, cls in merged.items():
    if cls == 'ex_noun' and word:
        last_char = word[-1]
        if last_char in ENDING_TO_CLASS:
            heuristic_additions[word] = ENDING_TO_CLASS[last_char]
        else:
            genuinely_unknown.append(word)

print(f"Heuristically classifiable: {len(heuristic_additions)}")
print(f"Genuinely ambiguous (consonant-final): {len(genuinely_unknown)}")

# Merge heuristic additions into final lexicon
# But mark them separately so you know their confidence level
heuristic_lexicon = {}
with open('./data/noun_lexicon_final.tsv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        heuristic_lexicon[row['word']] = (row['paradigm_class'], 'certain')

for word, cls in heuristic_additions.items():
    normalized = normalize_hindi(word)
    if normalized not in heuristic_lexicon:
        heuristic_lexicon[normalized] = (cls, 'heuristic')

# Save expanded lexicon
with open('./data/noun_lexicon_expanded.tsv', 'w', encoding='utf-8') as f:
    f.write("word\tparadigm_class\tconfidence\n")
    for word, (cls, confidence) in sorted(heuristic_lexicon.items()):
        f.write(f"{word}\t{cls}\t{confidence}\n")

print(f"\nExpanded lexicon total: {len(heuristic_lexicon)}")

