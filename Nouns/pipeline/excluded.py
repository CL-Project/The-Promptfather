from collections import Counter
from conflict_resolver import merged

endings = Counter()
for word, cls in merged.items():
    if cls == 'ex_noun' and word:
        endings[word[-1]] += 1

print("Final character distribution of ex_noun words:")
for char, count in endings.most_common(20):
    print(f"  {char} ({hex(ord(char))}): {count}")