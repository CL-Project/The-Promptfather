noun_lexicon = []  # will hold (word, paradigm_class) tuples

with open('./HindiWN_1_5/MorpHIN/Lexicon/new_Prop_Lexicon', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        line = line.strip().lstrip('/')
        if not line:
            continue
        parts = line.split(',')
        
        if len(parts) == 3:
            word = parts[0].strip('<>').strip()
            paradigm = parts[1].strip('<>').strip()
            pos = parts[2].strip('<>').strip()
            
            if pos == 'noun':
                noun_lexicon.append((word, paradigm))

print(f"Total nouns in MorpHIN lexicon: {len(noun_lexicon)}")
print("\nSample entries:")
for word, paradigm in noun_lexicon[:20]:
    print(f"  {word} → {paradigm}")