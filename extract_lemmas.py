import re
import os

def extract_lemmas(filepath, single_word_only=True, remove_digits=True, min_length=2):
    lemmas = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            tokens = line.split()
            lemma = tokens[0].replace('_', ' ')
            
            # Filter 1: must contain Devanagari
            if not re.search(r'[\u0900-\u097F]', lemma):
                continue
            
            # Filter 2: remove multi-word expressions (compounds, phrases)
            # These can't be assigned a single paradigm class meaningfully
            if single_word_only and ' ' in lemma:
                continue

            # if single_word_only and '-' in lemma:
                # continue
            
            # Filter 3: remove anything still containing digits (like 100वाँ)
            if remove_digits and re.search(r'\d', lemma):
                continue
            
            # Filter 4: skip very short entries that are likely artifacts
            if len(lemma) < min_length:
                continue
            
            lemmas.append(lemma)
    
    return lemmas

# --- Run extraction for nouns ---
noun_file = './Project/HindiWN_1_5/database/idxnoun_txt'   # adjust path if needed
verb_file = './Project/HindiWN_1_5/database/idxverb_txt'

noun_lemmas = extract_lemmas(noun_file)
verb_lemmas = extract_lemmas(verb_file)

print(f"Extracted {len(noun_lemmas)} noun lemmas")
print(f"Extracted {len(verb_lemmas)} verb lemmas")
print("\nFirst 10 nouns:", noun_lemmas[:10])
print("First 10 verbs:", verb_lemmas[:10])

# Save to text files for inspection
with open('noun_lemmas.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(noun_lemmas))

with open('verb_lemmas.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(verb_lemmas))

print("\nSaved to noun_lemmas.txt and verb_lemmas.txt")