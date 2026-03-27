def parse_lexicon(filepath):
    lexicon = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lstrip('/')
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 3:
                continue
            word = parts[0].strip('<>').strip()
            paradigm = parts[1].strip('<>').strip()
            pos = parts[2].strip('<>').strip()
            if pos == 'noun' and word:
                lexicon[word] = paradigm
    return lexicon

lex1 = parse_lexicon('./HindiWN_1_5/MorpHIN/Lexicon/new_Prop_Lexicon')
lex2 = parse_lexicon('./HindiWN_1_5/MorpHIN/Lexicon/newer_LEXICON')

# Find conflicts
conflicts = {}
for word in set(lex1) & set(lex2):  # words in both
    if lex1[word] != lex2[word]:
        conflicts[word] = (lex1[word], lex2[word])

# Merge — prefer newer_LEXICON where conflicts exist
merged = {**lex1, **lex2}  # lex2 overwrites lex1 on conflicts

print(f"new_prop_lexicon: {len(lex1)} nouns")
print(f"newer_LEXICON: {len(lex2)} nouns")
print(f"Words in both: {len(set(lex1) & set(lex2))}")
print(f"Words only in new_prop: {len(set(lex1) - set(lex2))}")
print(f"Words only in newer: {len(set(lex2) - set(lex1))}")
print(f"Conflicts (same word, different class): {len(conflicts)}")
print(f"Merged total: {len(merged)}")

print(f"\nSample conflicts:")
for word, (c1, c2) in list(conflicts.items())[:20]:
    print(f"  {word}: {c1} vs {c2}")

