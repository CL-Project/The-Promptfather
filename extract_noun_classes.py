from collections import Counter

noun_lexicon = []
classes = Counter()

with open('./HindiWN_1_5/MorpHIN/Lexicon/newer_LEXICON', 'r', encoding='utf-8') as f:
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
        
        if pos == 'noun':
            noun_lexicon.append((word, paradigm))
            classes[paradigm] += 1

print(f"Total nouns: {len(noun_lexicon)}")
print(f"\nAll paradigm classes:")
for cls, count in classes.most_common():
    print(f"  {cls}: {count} words")


