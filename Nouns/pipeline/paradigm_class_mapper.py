from collections import Counter
from conflict_resolver import merged
# Normalize garbage/typo classes to their correct equivalents
TYPO_FIX = {
    '<क्रोध': 'क्रोध',
    'ेex_noun': 'ex_noun',
    'गुिड़या': 'गुड़िया',
    'लडकी': 'लड़की',
    'अाशा': 'आशा',
    'शािन्त': 'शान्ति',
    'भीडञ': 'भीड़',
    'भिड़': 'भीड़',
    'रातन': 'रात',
    'आ': 'ex_noun',      # single character, likely artifact
    'बच्चा': 'लड़का',    # same M1 pattern
    'चिड़िया': 'गुड़िया', # same F2 subclass pattern
    'रेडिओ': 'ex_noun',  # loanword
    'फोटो': 'ex_noun',   # loanword
    'गेहूॅ': 'ex_noun',  # encoding issue
    'proper_noun': 'ex_noun',  # skip proper nouns
}

# Now map all MorpHIN classes to your formal M1-F5 system
# Based on gender and phonological pattern of the exemplar word
PARADIGM_MAP = {
    # MASCULINE CLASSES
    'लड़का': 'M1',     # masculine -ā final: लड़का → लड़के (obl/pl)
    'राजा': 'M1',      # masculine -ā final animate: same pattern
    'विधाता': 'M1',    # masculine -ā final: same pattern
    'लोहा': 'M1',      # masculine -ā final mass noun: same pattern
    'घर': 'M2',        # masculine consonant-final: invariant
    'क्रोध': 'M2',     # masculine consonant-final: invariant
    'खर्च': 'M2',      # masculine consonant-final: invariant
    'अपनापन': 'M2',    # masculine -pan suffix: invariant
    'आदमी': 'M3',      # masculine -ī final: invariant
    'कवि': 'M3',       # masculine -i final: invariant
    'पानी': 'M3',      # masculine -ī final mass noun: invariant
    'आलू': 'M4',       # masculine -ū/-u final
    'शत्रु': 'M4',     # masculine -u final
    'लहू': 'M4',       # masculine -ū final
    'बालू': 'M4',      # masculine -ū final

    # FEMININE CLASSES
    'लड़की': 'F1',     # feminine -ī final: लड़की → लड़कियाँ (dir pl), लड़कियों (obl pl)
    'गुड़िया': 'F1',   # feminine -iyā final: same pattern
    'आशा': 'F2',       # feminine -ā final: आशा → आशाएँ (pl)
    'ईर्ष्या': 'F2',   # feminine -yā final: same pattern
    'रात': 'F3',       # feminine consonant-final: रात → रातें (pl)
    'भीड़': 'F3',      # feminine consonant-final: same pattern
    'शान्ति': 'F4',    # feminine -i/-ī final: शान्ति → शान्तियाँ (pl)
    'आपत्ति': 'F4',    # feminine -i final: same pattern
    'वायु': 'F5',      # feminine -u/-ū final
    'ऋतु': 'F5',       # feminine -u final
    'बहू': 'F5',       # feminine -ū final

    # IRREGULAR / SKIP
    'माँ': 'IRREG',
    'सरसों': 'IRREG',
    'लौ': 'IRREG',
    'कुऑ': 'IRREG',
    'ex_noun': 'UNKNOWN',
}

# Apply typo fixes then paradigm mapping
final_lexicon = {}
skipped = []

for word, cls in merged.items():
    # Fix typos first
    cls = TYPO_FIX.get(cls, cls)
    # Map to formal system
    mapped = PARADIGM_MAP.get(cls, 'UNKNOWN')
    if mapped != 'UNKNOWN' and mapped != 'IRREG':
        final_lexicon[word] = mapped
    else:
        skipped.append((word, cls, mapped))

# Final distribution
dist = Counter(cls for cls in final_lexicon.values())
print(f"Final classified lexicon: {len(final_lexicon)} nouns")
print(f"Skipped (unknown/irregular): {len(skipped)}")
print("\nDistribution across your paradigm classes:")
for cls, count in sorted(dist.items()):
    print(f"  {cls}: {count}")

# Save final lexicon
with open('./data/noun_lexicon_final.tsv', 'w', encoding='utf-8') as f:
    f.write("word\tparadigm_class\n")
    for word, cls in sorted(final_lexicon.items()):
        f.write(f"{word}\t{cls}\n")

print("\nSaved to noun_lexicon_final.tsv")