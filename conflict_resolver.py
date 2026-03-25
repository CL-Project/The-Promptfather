from collections import Counter
from conflict_finder import conflicts, lex1, lex2

# Check what proportion of conflicts are specifically रात vs घर
raat_ghar = sum(1 for w, (c1, c2) in conflicts.items() 
                if set([c1,c2]) == {'रात', 'घर'})
other = len(conflicts) - raat_ghar

print(f"रात vs घर conflicts: {raat_ghar}")
print(f"Other conflicts: {other}")

# See what the other conflicts actually are
print("\nNon रात/घर conflicts:")
for word, (c1, c2) in conflicts.items():
    if set([c1,c2]) != {'रात', 'घर'}:
        print(f"  {word}: {c1} vs {c2}")



def resolve_conflicts(lex1, lex2):
    merged = {}
    conflict_log = []
    
    all_words = set(lex1) | set(lex2)
    
    for word in all_words:
        c1 = lex1.get(word)
        c2 = lex2.get(word)
        
        # Word only in one file
        if c1 is None:
            merged[word] = c2
            continue
        if c2 is None:
            merged[word] = c1
            continue
        
        # No conflict
        if c1 == c2:
            merged[word] = c1
            continue
        
        # Pattern 1: one is ex_noun, other is specific → take specific
        if c1 == 'ex_noun' and c2 != 'ex_noun':
            merged[word] = c2
            continue
        if c2 == 'ex_noun' and c1 != 'ex_noun':
            merged[word] = c1
            continue
        
        # Pattern 2: रात vs घर → prefer newer_LEXICON (c2)
        if set([c1, c2]) == {'रात', 'घर'}:
            merged[word] = c2
            continue
        
        # Everything else: prefer newer_LEXICON but log it
        merged[word] = c2
        conflict_log.append((word, c1, c2))
    
    return merged, conflict_log

merged, remaining_conflicts = resolve_conflicts(lex1, lex2)

print(f"Final merged lexicon: {len(merged)} nouns")
print(f"Remaining unresolved conflicts (logged): {len(remaining_conflicts)}")

# Show class distribution of final merged lexicon
dist = Counter(cls for cls in merged.values())
print("\nFinal paradigm class distribution:")
for cls, count in dist.most_common():
    print(f"  {cls}: {count}")

# Save the conflict log for your error analysis section
with open('conflict_log.txt', 'w', encoding='utf-8') as f:
    for word, c1, c2 in remaining_conflicts:
        f.write(f"{word}\t{c1}\t{c2}\n")

# Save the final merged lexicon
with open('noun_lexicon_merged.tsv', 'w', encoding='utf-8') as f:
    f.write("word\tparadigm_class\n")
    for word, cls in sorted(merged.items()):
        f.write(f"{word}\t{cls}\n")

print("\nSaved to noun_lexicon_merged.tsv and conflict_log.txt")